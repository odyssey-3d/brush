use burn::prelude::{Backend, Tensor};

use crate::{model::SimplicitsModel, sampling::randomly_sample_points};

pub fn do_physics_test<B: Backend>(
    model: &SimplicitsModel<B>,
    points: Vec<f32>,
    youngs_modulus: Vec<f32>,
    poisson_ratio: Vec<f32>,
    density_rho: Vec<f32>,
    device: &B::Device,
) {
    let points: Tensor<B, 1> = Tensor::from_floats(&*points, device);
    let points = points.reshape([-1, 3]);

    let youngs_modulus: Tensor<B, 1> = Tensor::from_floats(&*youngs_modulus, device);
    let poisson_ratio: Tensor<B, 1> = Tensor::from_floats(&*poisson_ratio, device);
    let density_rho: Tensor<B, 1> = Tensor::from_floats(&*density_rho, device);
    let num_samples = 6;

    do_physics_pass(
        model,
        &points,
        &youngs_modulus,
        &poisson_ratio,
        &density_rho,
        num_samples,
    );
}

pub fn do_physics_pass<B: Backend>(
    model: &SimplicitsModel<B>,
    normalized_pts: &Tensor<B, 2>,
    yms: &Tensor<B, 1>,
    prs: &Tensor<B, 1>,
    rhos: &Tensor<B, 1>,
    num_samples: usize,
) {
    let device = &normalized_pts.device();
    let num_points = normalized_pts.shape().dims[0];
    let (sampled_points, sampled_yms, sampled_prs, sampled_rhos) = randomly_sample_points(
        num_samples,
        num_points,
        device,
        normalized_pts,
        yms,
        prs,
        rhos,
    );

    let sim_weights = model.forward(sampled_points.clone());
    let rigid = Tensor::ones(
        [sampled_points.shape().dims[0], 1],
        &sampled_points.device(),
    );
    let sim_weights = Tensor::cat(vec![sim_weights, rigid], 1);
    println!("sim_weights:{:?}", sim_weights.shape());

    let model_plus_rigid_fn = |points: Tensor<B, 2>| {
        let simplicits = model.forward(points.clone());
        let ones = Tensor::ones([points.shape().dims[0], 1], &points.device());
        Tensor::cat(vec![simplicits, ones], 1)
    };

    // init simulation DOFs (Z)
    let z = Tensor::<B, 2>::zeros([sim_weights.shape().dims[1] * 12, 1], &sim_weights.device());
    let z_prev = z.clone().detach();
    let z_dot = z.zeros_like();
    let x0_flat = sampled_points.clone().flatten::<1>(0, 1);

    println!("z:{:?}", z.shape());
    println!("x0_flat:{:?}", x0_flat.shape());

    let (m, inv_m) = lumped_mass_matrix(sampled_rhos, 1.0, 3);
    println!("m:{:?}", m.shape());
    println!("inv_m:{:?}", inv_m.shape());
    println!("m: {}", m);
    println!("inv_m: {}", inv_m);

    let b = linear_blending_skinning_matrix(sampled_points.clone(), sim_weights.clone());
}

pub fn lumped_mass_matrix<B: Backend>(
    density_rho: Tensor<B, 1>,
    total_volume: f64,
    spatial_dims: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let vol_per_sample = total_volume / density_rho.shape().dims[0] as f64;
    let point_masses = density_rho * vol_per_sample;

    // repeat the point masses for each spatial dimension and flatten
    let point_masses = point_masses
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, spatial_dims)
        .flatten::<1>(0, 1)
        .unsqueeze_dim::<2>(1);

    // create a diagonal matrix from the point masses
    let eye = Tensor::<B, 2>::eye(point_masses.shape().dims[0], &point_masses.device());
    let diag = eye.clone() * point_masses.clone();
    let recip_diag = eye * point_masses.recip();
    (diag, recip_diag)
}

pub fn linear_blending_skinning_matrix<B: Backend>(points: Tensor<B, 2>, weights: Tensor<B, 2>) {
    //x_i = sum(w(x^0_i)_j * T_j

    let num_samples = points.shape().dims[0]; //N
    let num_handles = weights.shape().dims[1]; //H

    let ones = Tensor::<B, 2>::ones([num_samples, 1], &points.device());

    let x03 = Tensor::cat(vec![points.clone(), ones], 1);
    let x03 = x03.unsqueeze_dim::<3>(2).repeat_dim(1, 3).reshape([-1, 12]);
    let x03 = x03
        .unsqueeze_dim::<3>(2)
        .repeat_dim(1, 3 * num_handles)
        .reshape([-1, 12 * num_handles as i32]);
    println!("x03:{:?}", x03.shape());
    println!("x03:{}", x03);

    let w = weights
        .unsqueeze_dim::<3>(2)
        .repeat_dim(1, 12)
        .reshape([-1, 12 * num_samples as i32]);
    println!("w:{:?}", w.shape());
    println!("w:{}", w);
}
