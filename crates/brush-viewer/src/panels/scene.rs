use brush_dataset::splat_export;
use brush_ui::burn_texture::BurnTexture;
use burn_wgpu::Wgpu;
use egui::epaint::mutex::RwLock as EguiRwLock;

use ody_simplicits::model::{load_simplicits_model, SimplicitsModel};

use std::sync::Arc;

use brush_render::gaussian_splats::Splats;
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect};
use glam::Affine3A;
use tokio_stream::StreamExt;
use tokio_with_wasm::alias as tokio;

use tracing::trace_span;
use web_time::Instant;

use crate::{
    simplicits_training::simplicits_training,
    train_loop::TrainMessage,
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};

type Backend = Wgpu;

pub(crate) struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,
    pub(crate) last_message: Option<ViewerMessage>,
    pub(crate) simplicits: Option<SimplicitsModel<Backend>>,

    is_loading: bool,
    is_training: bool,
    live_update: bool,
    paused: bool,

    last_cam_trans: Affine3A,
    dirty: bool,

    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    renderer: Arc<EguiRwLock<Renderer>>,
}

impl ScenePanel {
    pub(crate) fn new(
        queue: Arc<wgpu::Queue>,
        device: Arc<wgpu::Device>,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        Self {
            backbuffer: BurnTexture::new(device.clone(), queue.clone()),
            last_draw: None,
            last_message: None,
            live_update: true,
            paused: false,
            dirty: true,
            last_cam_trans: Affine3A::IDENTITY,
            is_loading: false,
            is_training: false,
            queue,
            device,
            renderer,
            simplicits: None,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut ViewerContext,
        rect: Rect,
        splats: &Splats<Backend>,
    ) {
        // If this viewport is re-rendering.
        if ui.ctx().has_requested_repaint() {
            let _span = trace_span!("Render splats").entered();
            let size = glam::uvec2(1024, 1024);
            let (img, _) = splats.render(&context.camera, size, true);
            self.backbuffer.update_texture(img, self.renderer.clone());
            self.dirty = false;
        }

        if let Some(id) = self.backbuffer.id() {
            ui.scope(|ui| {
                if context
                    .dataset
                    .train
                    .views
                    .first()
                    .map(|view| view.image.color().has_alpha())
                    .unwrap_or(false)
                {
                    // if training views have alpha, show a background checker.
                    brush_ui::draw_checkerboard(ui, rect);
                } else {
                    // If a scene is opaque, it assumes a black background.
                    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                };

                ui.painter().image(
                    id,
                    rect,
                    Rect {
                        min: egui::pos2(0.0, 0.0),
                        max: egui::pos2(1.0, 1.0),
                    },
                    Color32::WHITE,
                );
            });
        }
    }

    fn show_splat_options(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut ViewerContext,
        splats: &Splats<Backend>,
    ) -> egui::InnerResponse<()> {
        ui.horizontal(|ui| {
            if self.is_training {
                ui.add_space(15.0);

                let label = if self.paused {
                    "⏸ paused"
                } else {
                    "⏵ training"
                };

                if ui.selectable_label(!self.paused, label).clicked() {
                    self.paused = !self.paused;
                    context.send_train_message(TrainMessage::Paused(self.paused));
                }

                ui.add_space(15.0);

                ui.scope(|ui| {
                    ui.style_mut().visuals.selection.bg_fill = Color32::DARK_RED;
                    if ui
                        .selectable_label(self.live_update, "🔴 Live update splats")
                        .clicked()
                    {
                        self.live_update = !self.live_update;
                    }
                });

                ui.add_space(15.0);

                if ui.button("⬆ Export").clicked() {
                    let splats = splats.clone();

                    let fut = async move {
                        let file = rrfd::save_file("export.ply").await;

                        // Not sure where/how to show this error if any.
                        match file {
                            Err(e) => {
                                log::error!("Failed to save file: {e}");
                            }
                            Ok(file) => {
                                let data = splat_export::splat_to_ply(splats).await;

                                let data = match data {
                                    Ok(data) => data,
                                    Err(e) => {
                                        log::error!("Failed to serialize file: {e}");
                                        return;
                                    }
                                };

                                if let Err(e) = file.write(&data).await {
                                    log::error!("Failed to write file: {e}");
                                }
                            }
                        }
                    };

                    tokio::task::spawn(fut);
                }
            }
        })
    }
}

fn train_simplicits_loop(context: &mut ViewerContext, splats: &Splats<Backend>) {
    let device = context.device.clone();
    let positions = splats
        .means
        .val()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();

    let fut = async move {
        let stream = simplicits_training(device, positions);
        let mut stream = std::pin::pin!(stream);
        while let Some(message) = stream.next().await {
            match message {
                Ok(ViewerMessage::Simplicits { iter, loss }) => {
                    println!("SIMPLICITS::{} - loss:{}", iter, loss);
                }
                _ => {
                    println!("SIMPLICITS::DONE");
                    break;
                }
            }
        }
    };
    tokio::task::spawn(fut);
}

impl ViewerPanel for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: ViewerMessage, _: &mut ViewerContext) {
        match message.clone() {
            ViewerMessage::NewSource => {
                self.last_message = None;
                self.paused = false;
                self.is_loading = false;
                self.is_training = false;
            }
            ViewerMessage::DoneLoading { training: _ } => {
                self.is_loading = false;
            }
            ViewerMessage::StartLoading {
                training,
                filename: _,
            } => {
                self.is_training = training;
                self.last_message = None;
                self.is_loading = true;
            }
            ViewerMessage::Splats { iter: _, splats: _ } => {
                if self.live_update {
                    self.dirty = true;
                    self.last_message = Some(message);
                }
            }
            ViewerMessage::Error(_) => {
                self.last_message = Some(message);
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        // Empty scene, nothing to show.
        if !self.is_loading && context.dataset.train.views.is_empty() && self.last_message.is_none()
        {
            ui.heading("Load a ply file or dataset to get started.");
            ui.add_space(5.0);
            ui.label(
                r#"
Load a pretrained .ply file to view it

Or load a dataset to train on. These are zip files with:
    - a transform_train.json and images, like the synthetic NeRF dataset format.
    - COLMAP data, containing the `images` & `sparse` folder."#,
            );

            ui.add_space(10.0);

            #[cfg(target_family = "wasm")]
            ui.scope(|ui| {
                ui.visuals_mut().override_text_color = Some(Color32::YELLOW);
                ui.heading("Note: Running in browser is still experimental");

                ui.label(
                    r#"
In browser training is slower, and lower quality than the native app.

For bigger training runs consider using the native app."#,
                );
            });

            return;
        }

        if self.last_cam_trans
            != glam::Affine3A::from_rotation_translation(
                context.camera.rotation,
                context.camera.position,
            )
            || context.controls.is_animating()
        {
            self.dirty = true;
        }

        if let Some(message) = self.last_message.clone() {
            match message {
                ViewerMessage::Error(e) => {
                    ui.label("Error: ".to_owned() + &e.to_string());
                }
                ViewerMessage::Splats { iter: _, splats } => {
                    let mut size = ui.available_size();
                    let focal = context.camera.focal(glam::uvec2(1, 1));
                    let aspect_ratio = focal.y / focal.x;
                    if size.x / size.y > aspect_ratio {
                        size.x = size.y * aspect_ratio;
                    } else {
                        size.y = size.x / aspect_ratio;
                    }

                    let cur_time = Instant::now();
                    ui.horizontal(|ui| {
                        if ui.button("Train simplicits").clicked() {
                            train_simplicits_loop(context, &splats);
                        }
                        if ui.button("Load simplicits").clicked() {
                            self.simplicits =
                                Some(load_simplicits_model("model.mpk", &context.device));
                        }
                    });

                    if let Some(last_draw) = self.last_draw {
                        let delta_time = cur_time - last_draw;
                        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);
                        let rect = context.controls.handle_user_input(
                            ui,
                            size,
                            &mut context.camera,
                            delta_time,
                        );
                        self.draw_splats(ui, context, rect, &splats);

                        self.show_splat_options(ui, context, &splats);
                    }
                    self.last_draw = Some(cur_time);

                    // Also redraw next frame, need to check if we're still animating.
                    if self.dirty {
                        ui.ctx().request_repaint();
                    }
                }
                _ => {}
            }
        }
    }
}
