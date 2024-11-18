#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::sync::Arc;

use brush_render::{
    bounding_box::BoundingBox,
    camera::{focal_to_fov, fov_to_focal, Camera},
    gaussian_splats::{RandomSplatsConfig, Splats},
};
use brush_train::{
    image::image_to_tensor,
    scene::SceneView,
    train::{SceneBatch, SplatTrainer, TrainConfig},
};
use brush_ui::burn_texture::BurnTexture;
use burn::{
    backend::{
        wgpu::{WgpuDevice, WgpuSetup},
        Autodiff, Wgpu,
    },
    lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    module::AutodiffModule,
};
use eframe::egui_wgpu::WgpuConfiguration;
use egui::{load::SizedTexture, ImageSource, TextureHandle, TextureOptions};
use glam::{Quat, Vec2, Vec3};
use rand::SeedableRng;
use tokio::sync::mpsc::{Receiver, Sender};

type Backend = Wgpu;

struct TrainStep {
    splats: Splats<Backend>,
    step: u32,
}

fn spawn_train_loop(
    view: SceneView,
    config: TrainConfig,
    device: WgpuDevice,
    ctx: egui::Context,
    sender: Sender<TrainStep>,
) {
    // Spawn a task that iterates over the training stream.
    tokio::task::spawn(async move {
        let seed = 42;

        <Wgpu as burn::prelude::Backend>::seed(seed);
        let mut rng = rand::rngs::StdRng::from_seed([seed as u8; 32]);

        let init_bounds = BoundingBox::from_min_max(-Vec3::ONE * 5.0, Vec3::ONE * 5.0);

        let mut splats: Splats<Autodiff<Backend>> = Splats::from_random_config(
            RandomSplatsConfig::new()
                .with_sh_degree(0)
                .with_init_count(32),
            init_bounds,
            &mut rng,
            &device,
        );

        let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &device);

        // One batch of training data, it's the same every step so can just cosntruct it once.
        let batch = SceneBatch {
            gt_images: image_to_tensor(&view.image, &device).unsqueeze(),
            gt_views: vec![view],
            scene_extent: 1.0,
        };

        loop {
            let (new_splats, _) = trainer.step(batch.clone(), splats).await.unwrap();
            splats = new_splats;

            ctx.request_repaint();

            if sender
                .send(TrainStep {
                    splats: splats.valid(),
                    step: trainer.iter,
                })
                .await
                .is_err()
            {
                break;
            }
        }
    });
}

struct App {
    view: SceneView,
    tex_handle: TextureHandle,
    backbuffer: BurnTexture,
    receiver: Receiver<TrainStep>,
    last_step: Option<TrainStep>,
}

impl App {
    fn new(
        view: SceneView,
        setup: &WgpuSetup,
        ctx: egui::Context,
        events: Receiver<TrainStep>,
    ) -> Self {
        let color_img = egui::ColorImage::from_rgb(
            [view.image.width() as usize, view.image.height() as usize],
            &view.image.to_rgb8().into_vec(),
        );

        let handle = ctx.load_texture("nearest_view_tex", color_img, TextureOptions::default());

        Self {
            view,
            tex_handle: handle,
            backbuffer: BurnTexture::new(setup.device.clone(), setup.queue.clone()),
            receiver: events,
            last_step: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        while let Ok(step) = self.receiver.try_recv() {
            self.last_step = Some(step);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(msg) = self.last_step.as_ref() else {
                return;
            };

            let image = &self.view.image;

            let (img, _) = msg.splats.render(
                &self.view.camera,
                glam::uvec2(image.width(), image.height()),
                true,
            );

            let renderer = frame.wgpu_render_state().unwrap().renderer.clone();
            let size = egui::vec2(image.width() as f32, image.height() as f32);

            ui.horizontal(|ui| {
                let texture_id = self.backbuffer.update_texture(img, renderer);
                ui.image(ImageSource::Texture(SizedTexture::new(texture_id, size)));
                ui.image(ImageSource::Texture(SizedTexture::new(
                    self.tex_handle.id(),
                    size,
                )));
            });

            ui.label(format!("Splats: {}", msg.splats.num_splats()));
            ui.label(format!("Step: {}", msg.step));
        });
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let setup = brush_train::create_wgpu_setup().await;
    let wgpu_options = WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::Existing {
            instance: setup.instance.clone(),
            adapter: setup.adapter.clone(),
            device: setup.device.clone(),
            queue: setup.queue.clone(),
        },
        ..Default::default()
    };

    // NB: Load carrying icon. egui at head fails when no icon is included
    // as the built-in one is git-lfs which cargo doesn't clone properly.
    let icon = eframe::icon_data::from_png_bytes(
        &include_bytes!("../../brush-desktop/assets/icon-256.png")[..],
    )
    .unwrap();

    let native_options = eframe::NativeOptions {
        // Build app display.
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::Vec2::new(1100.0, 500.0))
            .with_active(true)
            .with_icon(std::sync::Arc::new(icon)),

        // Need a slightly more careful wgpu init to support burn.
        wgpu_options,
        ..Default::default()
    };

    let device = WgpuDevice::DefaultDevice;

    let lr_max = 1.5e-4;
    let decay = 1.0;

    let image = image::open("./crab.jpg").unwrap();

    let fov_x = 0.5 * std::f64::consts::PI;
    let fov_y = focal_to_fov(fov_to_focal(fov_x, image.width()), image.height());

    let center_uv = Vec2::ONE * 0.5;

    let camera = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        Quat::IDENTITY,
        fov_x,
        fov_y,
        center_uv,
    );

    let view = SceneView {
        name: "crabby".to_owned(),
        camera,
        image: Arc::new(image),
    };

    let config = TrainConfig::new(ExponentialLrSchedulerConfig::new(lr_max, decay))
        .with_max_refine_step(u32::MAX) // Just keep refining
        .with_warmup_steps(100) // Don't really need a warmup for simple 2D
        .with_reset_alpha_every_refine(u32::MAX); // Don't use alpha reset.

    eframe::run_native(
        "Brush",
        native_options,
        Box::new(move |cc| {
            let (sender, receiver) = tokio::sync::mpsc::channel(32);

            spawn_train_loop(view.clone(), config, device, cc.egui_ctx.clone(), sender);

            Ok(Box::new(App::new(
                view,
                &setup,
                cc.egui_ctx.clone(),
                receiver,
            )))
        }),
    )
    .unwrap();
}
