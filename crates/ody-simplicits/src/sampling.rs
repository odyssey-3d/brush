use burn::{
    prelude::Backend,
    tensor::{Tensor, Int},
};

fn random_sample_indices<B: Backend>(
    num_samples: usize,
    num_points: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let u = Tensor::<B, 1>::random(
        [num_samples],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        device,
    ) * ((num_points - 1) as f32);
    u.int()
}
pub fn randomly_sample_points<B: Backend>(
    num_samples: usize,
    num_points: usize,
    device: &<B as Backend>::Device,
    normalized_pts: &Tensor<B, 2>,
    yms: &Tensor<B, 1>,
    prs: &Tensor<B, 1>,
    rhos: &Tensor<B, 1>,
) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let sample_indices = random_sample_indices(num_samples, num_points, device);
    let sampled_points = normalized_pts.clone().select(0, sample_indices.clone());
    let sampled_yms = yms.clone().select(0, sample_indices.clone());
    let sampled_prs = prs.clone().select(0, sample_indices.clone());
    let sampled_rhos = rhos.clone().select(0, sample_indices.clone());
    (sampled_points, sampled_yms, sampled_prs, sampled_rhos)
}
