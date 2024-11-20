use burn::prelude::*;

use crate::utils::*;

pub(crate) fn calculate_lame_params<B: Backend>(
    youngs_modulus: Tensor<B, 1>,
    poisson_ratio: Tensor<B, 1>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let ones = youngs_modulus.clone().ones_like();
    let lambda = youngs_modulus.clone() * poisson_ratio.clone()
        / ((ones.clone() + poisson_ratio.clone()) * (ones.clone() - poisson_ratio.clone() * 2.0));
    let mu = youngs_modulus / ((ones + poisson_ratio) * 2.0);
    (lambda, mu)
}

pub(crate) fn cauchy_strain<B: Backend>(deformation_gradient: Tensor<B, 5>) -> Tensor<B, 5> {
    let dims = deformation_gradient.shape();

    let id_mat: Tensor<B, 5> = Tensor::eye(3, &deformation_gradient.device())
        .unsqueeze_dims::<5>(&[0, 0, 0])
        .expand(dims);

    // Calculate (F^T + F)/2 - I
    let strain =
        (deformation_gradient.clone().swap_dims(3, 4) + deformation_gradient) * 0.5 - id_mat;

    strain
}

pub(crate) fn linear_elastic_energy<B: Backend>(
    mu: Tensor<B, 3>,
    lambda: Tensor<B, 3>,
    deformation_gradient: Tensor<B, 5>,
) -> Tensor<B, 3> {
    let dims = deformation_gradient.shape();
    let batch_dims = Shape::from(dims.dims[..dims.num_dims() - 2].to_vec());

    let eps = cauchy_strain(deformation_gradient.clone());

    let eps_reshaped = eps.clone().reshape([batch_dims.num_elements(), 3, 3]);

    let trace_eps = calculate_trace(eps_reshaped).reshape::<3, _>(batch_dims.clone());

    let eps_outerprod =
        eps.clone()
            .swap_dims(3, 4)
            .matmul(eps)
            .reshape([batch_dims.num_elements(), 3, 3]);

    let trace_outerprod = calculate_trace(eps_outerprod).reshape::<3, _>(batch_dims);

    mu * trace_outerprod + (lambda * 0.5) * trace_eps.clone() * trace_eps
}

#[allow(dead_code)]
pub(crate) fn linear_elastic_gradient<B: Backend>(
    mu: Tensor<B, 3>,
    lambda: Tensor<B, 3>,
    deformation_gradient: Tensor<B, 5>,
) -> Tensor<B, 5> {
    // Stress (/Linear Deformation Gradeint(F))
    // σ = 2μE + λtr(E)I
    // σ = µ(F + F.T − 2I) + λtr(F− I)I.

    let dims: Shape = deformation_gradient.shape();

    let batch_dims = Shape::from(dims.dims[..dims.num_dims() - 2].to_vec());
    let id_mat: Tensor<B, 5> = Tensor::eye(3, &deformation_gradient.device())
        .unsqueeze_dims::<5>(&[0, 0, 0])
        .expand(dims);

    let f_m_i =
        (deformation_gradient.clone() - id_mat.clone()).reshape([batch_dims.num_elements(), 3, 3]);

    let trace_f_m_i = calculate_trace(f_m_i).reshape::<3, _>(batch_dims.clone());

    let g1 = mu.unsqueeze_dims::<5>(&[-1, -1])
        * (deformation_gradient.clone().swap_dims(3, 4) // F.T
              + deformation_gradient // F
              - id_mat.clone() * 2.0); // 2I

    let g2 = (lambda * trace_f_m_i).unsqueeze_dims::<5>(&[-1, -1]) * id_mat;

    g1 + g2
}

pub(crate) fn neohookean_energy<B: Backend>(
    mu: Tensor<B, 3>,
    lambda: Tensor<B, 3>,
    deformation_gradient: Tensor<B, 5>,
) -> Tensor<B, 3> {
    let dims = deformation_gradient.shape();
    let batch_dims = Shape::from(dims.dims[..dims.num_dims() - 2].to_vec());

    // Calculate F^T F (Cauchy-Green deformation tensor)
    let f_t_f = deformation_gradient
        .clone()
        .swap_dims(3, 4)
        .matmul(deformation_gradient.clone())
        .reshape([batch_dims.num_elements(), 3, 3]);

    let deformation_gradient = deformation_gradient.reshape([batch_dims.num_elements(), 3, 3]);

    // I2 = trace(F^T F)
    let i2 = calculate_trace(f_t_f).reshape::<3, _>(batch_dims.clone());

    // Calculate J = det(F)
    let j = calculate_determinant(deformation_gradient).reshape::<3, _>(batch_dims);

    // Calculate energy W = (mu/2) * (I2 - 3) + (lambda/2) * (J - 1)^2 - mu * (J - 1)
    let three = Tensor::full_like(&i2, 3.0);
    let one = Tensor::full_like(&j, 1.0);
    let c1 = mu.clone() * 0.5;
    let d1 = lambda * 0.5;

    let w = c1 * (i2 - three) + d1 * (j.clone() - one.clone()) * (j.clone() - one.clone())
        - mu * (j - one);

    w
}
