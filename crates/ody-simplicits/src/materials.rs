use burn::prelude::*;

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
    let device = &deformation_gradient.device();

    let eps = cauchy_strain(deformation_gradient.clone());

    let eps_reshaped = eps.clone().reshape([batch_dims.num_elements(), 3, 3]);

    let trace_eps = (0..batch_dims.num_elements())
        .map(|i| {
            let index = Tensor::<B, 1, Int>::from_ints([i], device);
            let batch: Tensor<B, 2> = eps_reshaped.clone().select(0, index).squeeze::<2>(0);
            let trace = (batch * Tensor::<B, 2>::eye(3, device)).sum();
            trace
        })
        .collect::<Vec<_>>();
    let trace_eps = Tensor::cat(trace_eps, 0).reshape::<3, _>(batch_dims.clone());

    let eps_outerprod = eps.clone().swap_dims(3, 4).matmul(eps);
    let eps_outerprod = eps_outerprod.reshape([batch_dims.num_elements(), 3, 3]);

    let trace_outerprod = (0..batch_dims.num_elements())
        .map(|i| {
            let index = Tensor::<B, 1, Int>::from_ints([i], device);
            let batch = eps_outerprod.clone().select(0, index).squeeze::<2>(0);
            let trace = (batch * Tensor::<B, 2>::eye(3, device)).sum();
            trace
        })
        .collect::<Vec<_>>();

    let trace_outerprod = Tensor::cat(trace_outerprod, 0).reshape::<3, _>(batch_dims);

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

    let device = &deformation_gradient.device();
    let trace_f_m_i = (0..batch_dims.num_elements())
        .map(|i| {
            let index = Tensor::<B, 1, Int>::from_ints([i], device);
            let batch = f_m_i.clone().select(0, index).squeeze::<2>(0);
            let trace = (batch * Tensor::<B, 2>::eye(3, device)).sum();
            trace
        })
        .collect::<Vec<_>>();
    let trace_f_m_i = Tensor::cat(trace_f_m_i, 0).reshape::<3, _>(batch_dims);

    let g1 = mu.unsqueeze_dims::<5>(&[3, 4])
        * (deformation_gradient.clone().swap_dims(3, 4) // F.T
              + deformation_gradient // F
              - id_mat.clone() * 2.0); // 2I

    let g2 = (lambda * trace_f_m_i).unsqueeze_dims::<5>(&[3, 4]) * id_mat;

    g1 + g2
}

fn determinant<B: Backend>(matrix: Tensor<B, 2>) -> Tensor<B, 1> {
    let device = &matrix.device();
    let a = matrix.into_data().into_vec::<f32>().unwrap();

    // Calculate determinant using cofactor expansion
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) -  // col0
              a[1] * (a[3] * a[8] - a[5] * a[6]) + // col1
              a[2] * (a[3] * a[7] - a[4] * a[6]); // col2

    Tensor::<B, 1>::from_floats([det], device)
}
pub(crate) fn neohookean_energy<B: Backend>(
    mu: Tensor<B, 3>,
    lambda: Tensor<B, 3>,
    deformation_gradient: Tensor<B, 5>,
) -> Tensor<B, 3> {
    let dims = deformation_gradient.shape();
    let batch_dims = Shape::from(dims.dims[..dims.num_dims() - 2].to_vec());
    let device = &deformation_gradient.device();

    // Calculate F^T F (Cauchy-Green deformation tensor)
    let f_t_f = deformation_gradient
        .clone()
        .swap_dims(3, 4)
        .matmul(deformation_gradient.clone())
        .reshape([batch_dims.num_elements(), 3, 3]);

    // I2 = trace(F^T F)
    let i2 = (0..batch_dims.num_elements())
        .map(|i| {
            let index = Tensor::<B, 1, Int>::from_ints([i], device);
            let batch = f_t_f
                .clone()
                .reshape([batch_dims.num_elements(), 3, 3])
                .select(0, index)
                .squeeze::<2>(0);
            let trace = (batch * Tensor::<B, 2>::eye(3, device)).sum();
            trace
        })
        .collect::<Vec<_>>();
    let i2 = Tensor::cat(i2, 0).reshape::<3, _>(batch_dims.clone());

    // Calculate J = det(F)
    let deformation_gradient = deformation_gradient.reshape([batch_dims.num_elements(), 3, 3]);
    let j = (0..batch_dims.num_elements())
        .map(|i| {
            let index = Tensor::<B, 1, Int>::from_ints([i], device);
            let batch = deformation_gradient
                .clone()
                .select(0, index)
                .squeeze::<2>(0);
            determinant(batch)
        })
        .collect::<Vec<_>>();
    let j = Tensor::cat(j, 0).reshape::<3, _>(batch_dims);

    // Calculate energy W = (mu/2) * (I2 - 3) + (lambda/2) * (J - 1)^2 - mu * (J - 1)
    let three = Tensor::full_like(&i2, 3.0);
    let one = Tensor::full_like(&j, 1.0);
    let c1 = mu.clone() * 0.5;
    let d1 = lambda * 0.5;

    let w = c1 * (i2 - three) + d1 * (j.clone() - one.clone()) * (j.clone() - one.clone())
        - mu * (j - one);

    w
}
