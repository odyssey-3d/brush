use async_fn_stream::try_fn_stream;
use burn::{
    backend::Autodiff,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use burn_wgpu::{Wgpu, WgpuDevice};
use tokio_stream::Stream;

use ody_simplicits::{
    losses::compute_losses,
    model::{create_model, save_simplicits_model, SimplicitsModel},
};

type Backend = Wgpu;
type Device = WgpuDevice;

use crate::viewer::ViewerMessage;

pub(crate) fn simplicits_training(
    device: Device,
    points: Vec<f32>,
) -> impl Stream<Item = anyhow::Result<ViewerMessage>> {
    try_fn_stream(|emitter| async move {
        println!("simplicits_training");

        let lr = 1e-3;
        let le_coeff = 1e-1;
        let lo_coeff = 1e6;
        let num_batch = 8;
        let num_handles = 10;
        let num_samples = 16;
        let num_steps = 10000;
        let log_every_n = 1000;

        // Physics material parameters
        // youngs_modulus = 1e5
        // poisson_ratio = 0.45
        // rho = 500  # kg/m^3
        // approx_volume = 0.5  # m^3

        let approx_volume = 0.5;
        let youngs_modulus = vec![1e5; points.len()];
        let poisson_ratio = vec![0.45; points.len()];
        let density_rho = vec![500.0; points.len()];

        let model = train_simplicits(
            num_handles,
            &device,
            points,
            youngs_modulus,
            poisson_ratio,
            density_rho,
            approx_volume,
            num_steps,
            num_batch,
            num_samples,
            le_coeff,
            lo_coeff,
            lr,
            log_every_n,
            emitter,
        )
        .await;

        // Save model in MessagePack format with full precision
        let model_path = "model.mpk";
        save_simplicits_model(&model, model_path);

        Ok(())
    })
}

async fn train_simplicits(
    num_handles: usize,
    device: &Device,
    points: Vec<f32>,
    youngs_modulus: Vec<f32>,
    poisson_ratio: Vec<f32>,
    density_rho: Vec<f32>,
    approx_volume: f32,
    num_steps: u32,
    num_batch: usize,
    num_samples: usize,
    le_coeff: f64,
    lo_coeff: f64,
    lr: f64,
    log_every_n: u32,
    emitter: async_fn_stream::TryStreamEmitter<ViewerMessage, anyhow::Error>,
) -> SimplicitsModel<Autodiff<Backend>> {
    let mut model = create_model::<Autodiff<Backend>>(num_handles, device);
    println!("{}", model);

    let opt_config = AdamConfig::new().with_epsilon(1e-3);
    let mut optim = opt_config.init::<Autodiff<Backend>, SimplicitsModel<Autodiff<Backend>>>();

    let points: Tensor<Autodiff<Backend>, 1> = Tensor::from_floats(&*points, device);
    let points: Tensor<Autodiff<Backend>, 2> = points.reshape([-1, 3]);

    let youngs_modulus: Tensor<Autodiff<Backend>, 1> =
        Tensor::from_floats(&*youngs_modulus, device);
    let poisson_ratio: Tensor<Autodiff<Backend>, 1> = Tensor::from_floats(&*poisson_ratio, device);
    let density_rho: Tensor<Autodiff<Backend>, 1> = Tensor::from_floats(&*density_rho, device);

    let bb_max = points.clone().max_dim(0);
    let bb_min = points.clone().min_dim(0);

    let norm_bb_max =
        ((points.clone() - bb_min.clone()) / (bb_max.clone() - bb_min.clone())).max_dim(0);
    let norm_bb_min =
        ((points.clone() - bb_min.clone()) / (bb_max.clone() - bb_min.clone())).min_dim(0);

    let bb_max = bb_max.into_data().into_vec::<f32>().unwrap();
    let bb_min = bb_min.into_data().into_vec::<f32>().unwrap();
    let norm_bb_max = norm_bb_max.into_data().into_vec::<f32>().unwrap();
    let norm_bb_min = norm_bb_min.into_data().into_vec::<f32>().unwrap();

    let bb_vol = (bb_max[0] - bb_min[0]) * (bb_max[1] - bb_min[1]) * (bb_max[2] - bb_min[2]);
    let norm_bb_vol = (norm_bb_max[0] - norm_bb_min[0])
        * (norm_bb_max[1] - norm_bb_min[1])
        * (norm_bb_max[2] - norm_bb_min[2]);

    let norm_appx_vol = approx_volume.clone() * (norm_bb_vol / bb_vol);

    let mut last_loss: Tensor<Autodiff<Backend>, 1> = Tensor::zeros([1], device);

    for i in 0..num_steps {
        let losses = compute_losses(
            &model,
            &points,
            &youngs_modulus,
            &poisson_ratio,
            &density_rho,
            i as f64 / num_steps as f64,
            num_batch,
            num_handles,
            norm_appx_vol as f64,
            num_samples,
            le_coeff,
            lo_coeff,
            device,
        );
        let losses = losses.0 + losses.1;

        last_loss = losses.clone();
        let grads = losses.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        if i % log_every_n == 0 {
            let loss = last_loss.clone().to_data().to_vec::<f32>().unwrap()[0];
            emitter
                .emit(ViewerMessage::Simplicits { iter: i, loss })
                .await;
        }
    }
    println!("Last loss: {}", last_loss);
    model
}
