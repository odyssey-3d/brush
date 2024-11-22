use burn::prelude::{Backend, Tensor, Int};

use crate::model::SimplicitsModel;

pub fn finite_difference_jacobian<B: Backend, F: Fn(Tensor<B, 2>) -> Tensor<B, 4>>(
    f: F,
    x: Tensor<B, 2>,
    epsilon: f64,
) -> Tensor<B, 5> {
    let device = &x.device();
    let n = x.shape().dims[0];

    let delta: Tensor<B, 1> = Tensor::sqrt(Tensor::from_floats([epsilon as f32], device));
    let h = delta.clone().expand([1, x.shape().dims[1]]) * Tensor::eye(x.shape().dims[1], device);
    let h = h.iter_dim(0).map(|x| x.clone()).collect::<Vec<_>>();

    let finite_diff_bounds = Tensor::cat(
        vec![
            x.clone() + h[0].clone(),
            x.clone() + h[1].clone(),
            x.clone() + h[2].clone(),
            x.clone() - h[0].clone(),
            x.clone() - h[1].clone(),
            x.clone() - h[2].clone(),
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

pub fn weighted_linear_blend_skinning<B: Backend>(
    x0: Tensor<B, 2>,
    transforms: Tensor<B, 4>,
    model: &SimplicitsModel<B>,
) -> Tensor<B, 4> {
    let w_x0 = model.forward(x0.clone());
    linear_blend_skinning(x0, transforms, w_x0)
}
