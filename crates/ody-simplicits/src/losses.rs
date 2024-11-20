use super::model::SimplicitsModel;
use crate::sampling::randomly_sample_points;

use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::*,
};

use crate::materials::{calculate_lame_params, linear_elastic_energy, neohookean_energy};

fn finite_difference_jacobian<B: Backend, F: Fn(Tensor<B, 2>) -> Tensor<B, 4>>(
    f: F,
    x: Tensor<B, 2>,
    epsilon: f64,
) -> Tensor<B, 5> {
    let device = &x.device();
    let n = x.shape().dims[0];

    let delta: Tensor<B, 1> = Tensor::sqrt(Tensor::from_floats([epsilon as f32], device));
    let h = delta.clone().expand([1, x.shape().dims[1]]) * Tensor::eye(x.shape().dims[1], device);
    let h0 = h
        .clone()
        .select(0, Tensor::<B, 1, Int>::from_ints([0], device));
    let h1 = h
        .clone()
        .select(0, Tensor::<B, 1, Int>::from_ints([1], device));
    let h2 = h
        .clone()
        .select(0, Tensor::<B, 1, Int>::from_ints([2], device));

    let finite_diff_bounds = Tensor::cat(
        vec![
            x.clone() + h0.clone(),
            x.clone() + h1.clone(),
            x.clone() + h2.clone(),
            x.clone() - h0.clone(),
            x.clone() - h1.clone(),
            x.clone() - h2.clone(),
        ],
        0,
    );

    let jacobian = f(finite_diff_bounds);
    let shape = jacobian.shape().dims.to_vec();
    let jacobian = jacobian.reshape([2, 3, n, shape[1], shape[2], shape[3]]);

    let j0 = jacobian
        .clone()
        .select(0, Tensor::<B, 1, Int>::from_ints([0], device))
        .squeeze(0);
    let j1 = jacobian
        .clone()
        .select(0, Tensor::<B, 1, Int>::from_ints([1], device))
        .squeeze(0);

    let jacobian = (j0 - j1)
        .permute([1, 2, 3, 4, 0])
        .div((delta * 2.0).expand([1, 1, 1, 1, 1]));

    jacobian
}

fn linear_blend_skinning<B: Backend>(
    x0: Tensor<B, 2>,
    transforms: Tensor<B, 4>,
    w_x0: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let n = x0.shape().dims[0];
    let b = transforms.shape().dims[0];
    let h = transforms.shape().dims[1];
    let bh = b * h;

    let device = &x0.device();

    let x0_i = x0.clone().unsqueeze_dim::<3>(1);

    let x03 = Tensor::cat(vec![x0_i.clone(), Tensor::ones([n, 1, 1], device)], 2)
        .swap_dims(1, 2)
        .unsqueeze_dim::<4>(1)
        .expand([n, bh, 4, 1]);

    let transforms = transforms
        .reshape([bh, 3, 4])
        .unsqueeze::<4>()
        .expand([n, bh, 3, 4]);

    let w_map_x0 = w_x0
        .unsqueeze_dim::<3>(2)
        .unsqueeze_dim::<4>(1)
        .unsqueeze_dim::<5>(3)
        .expand([n, b, h, 3, 1])
        .reshape([n, bh, 3, 1]);

    let x_map_x0 = w_map_x0
        .mul(transforms)
        .matmul(x03)
        .reshape([n, b, h, 3, 1])
        .sum_dim(2)
        .squeeze::<4>(2)
        .swap_dims(2, 3);

    let x0_i = x0_i.unsqueeze_dim::<4>(1).expand([n, b, 1, 3]);

    x_map_x0 + x0_i
}

fn weighted_linear_blend_skinning<B: Backend>(
    x0: Tensor<B, 2>,
    transforms: Tensor<B, 4>,
    model: &SimplicitsModel<B>,
) -> Tensor<B, 4> {
    let w_x0 = model.forward(x0.clone());
    linear_blend_skinning(x0, transforms, w_x0)
}

pub(crate) fn loss_elastic<B: Backend>(
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

pub(crate) fn loss_ortho<B: Backend>(weights: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 1> {
    MseLoss::new().forward(
        weights.clone().transpose().matmul(weights.clone()),
        Tensor::eye(weights.shape().dims[1], device),
        Reduction::Mean,
    )
}

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

