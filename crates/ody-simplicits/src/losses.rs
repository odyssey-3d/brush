use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::{Backend, Tensor},
    tensor::Int,
};

use crate::{
    materials::{calculate_lame_params, linear_elastic_energy, neohookean_energy},
    model::SimplicitsModel,
    sampling::randomly_sample_points,
    skinning::{finite_difference_jacobian, weighted_linear_blend_skinning},
};

pub fn compute_losses<B: Backend>(
    model: &SimplicitsModel<B>,
    normalized_pts: &Tensor<B, 2>,
    yms: &Tensor<B, 1>,
    prs: &Tensor<B, 1>,
    rhos: &Tensor<B, 1>,
    energy_interp: f64,
    batch_size: usize,
    num_handles: usize,
    appx_vol: f64,
    num_samples: usize,
    le_coeff: f64,
    lo_coeff: f64,
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let num_points = normalized_pts.shape().dims[0];

    let batch_transforms: Tensor<B, 4> = Tensor::random(
        [batch_size, num_handles, 3, 4],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    ) * 0.1;

    let (sampled_points, sampled_yms, sampled_prs, sampled_rhos) = randomly_sample_points(
        num_samples,
        num_points,
        device,
        normalized_pts,
        yms,
        prs,
        rhos,
    );

    let weights = model.forward(sampled_points.clone());
    let le = loss_elastic(
        model,
        &sampled_points,
        &sampled_yms,
        &sampled_prs,
        &sampled_rhos,
        batch_transforms,
        appx_vol,
        energy_interp,
    ) * le_coeff;

    let lo = loss_ortho(weights, device) * lo_coeff;
    (le, lo)
}

fn loss_elastic<B: Backend>(
    model: &SimplicitsModel<B>,
    pts: &Tensor<B, 2>,
    yms: &Tensor<B, 1>,
    prs: &Tensor<B, 1>,
    _rhos: &Tensor<B, 1>,
    transforms: Tensor<B, 4>,
    appx_vol: f64,
    energy_interp: f64,
) -> Tensor<B, 1> {
    let device = &pts.device();

    let (lambda, mu) = calculate_lame_params(yms.clone(), prs.clone());

    let pt_wise_fs = finite_difference_jacobian(
        |x| weighted_linear_blend_skinning(x, transforms.clone(), &model),
        pts.clone(),
        1e-6,
    );

    let pt_wise_fs = pt_wise_fs.select(2, Tensor::<B, 1, Int>::from_ints([0], device));

    let n = pt_wise_fs.shape().dims[0];
    let b = pt_wise_fs.shape().dims[1];

    let mu = mu
        .unsqueeze_dim::<2>(1)
        .expand([n, b])
        .unsqueeze_dim::<3>(2);

    let lambda = lambda
        .unsqueeze_dim::<2>(1)
        .expand([n, b])
        .unsqueeze_dim::<3>(2);

    // # ramps up from 100% linear elasticity to 100% neohookean elasticity
    let linear_elastic = linear_elastic_energy(mu.clone(), lambda.clone(), pt_wise_fs.clone())
        * (1.0 - energy_interp);
    let neo_elastic =
        neohookean_energy(mu.clone(), lambda.clone(), pt_wise_fs.clone()) * energy_interp;

    (linear_elastic + neo_elastic).sum() / (appx_vol / pts.shape().dims[0] as f64)
}

fn loss_ortho<B: Backend>(weights: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 1> {
    MseLoss::new().forward(
        weights.clone().transpose().matmul(weights.clone()),
        Tensor::eye(weights.shape().dims[1], device),
        Reduction::Mean,
    )
}
