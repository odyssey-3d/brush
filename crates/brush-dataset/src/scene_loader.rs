use ::tokio::sync::mpsc;
use ::tokio::sync::mpsc::Receiver;
use brush_render::Backend;
use brush_train::image::image_to_tensor;
use brush_train::scene::Scene;
use brush_train::train::SceneBatch;
use burn::tensor::Tensor;
use rand::{seq::SliceRandom, SeedableRng};
use tokio_with_wasm::alias as tokio;

pub struct SceneLoader<B: Backend> {
    receiver: Receiver<SceneBatch<B>>,
}

impl<B: Backend> SceneLoader<B> {
    pub fn new(scene: &Scene, batch_size: usize, seed: u64, device: &B::Device) -> Self {
        let scene = scene.clone();
        // The bounded size == number of batches to prefetch.
        let (tx, rx) = mpsc::channel(5);
        let device = device.clone();
        let scene_extent = scene.bounds(0.0, 0.0).extent.max_element() as f64;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let fut = async move {
            let mut shuf_indices = vec![];

            loop {
                let (selected_tensors, gt_views) = (0..batch_size)
                    .map(|_| {
                        let index = shuf_indices.pop().unwrap_or_else(|| {
                            shuf_indices = (0..scene.views.len()).collect();
                            shuf_indices.shuffle(&mut rng);
                            shuf_indices.pop().unwrap()
                        });
                        let view = scene.views[index].clone();
                        (image_to_tensor(&view.image, &device), view)
                    })
                    .unzip();

                let batch_tensor = Tensor::stack(selected_tensors, 0);

                let scene_batch = SceneBatch {
                    gt_images: batch_tensor,
                    gt_views,
                    scene_extent,
                };

                if tx.send(scene_batch).await.is_err() {
                    break;
                }
            }
        };

        tokio::task::spawn(fut);
        Self { receiver: rx }
    }

    pub async fn next_batch(&mut self) -> SceneBatch<B> {
        self.receiver
            .recv()
            .await
            .expect("Somehow lost data loading channel!")
    }
}
