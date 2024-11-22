use brush_dataset::splat_export;
use brush_ui::burn_texture::BurnTexture;
use burn_wgpu::Wgpu;
use core::f32;
use egui::epaint::mutex::RwLock as EguiRwLock;

use std::{sync::Arc, time::Duration};

use brush_render::{
    camera::{focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect};
use tokio_with_wasm::alias as tokio;
use tracing::trace_span;
use web_time::Instant;

use crate::{
    train_loop::TrainMessage,
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};

type Backend = Wgpu;

pub(crate) struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    view_splats: Vec<Splats<Wgpu>>,
    frame: f32,
    err: Option<Arc<anyhow::Error>>,

    is_loading: bool,

    is_training: bool,
    live_update: bool,
    paused: bool,

    last_size: glam::UVec2,
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
            frame: 0.0,
            backbuffer: BurnTexture::new(device.clone(), queue.clone()),
            last_draw: None,
            err: None,
            view_splats: vec![],
            live_update: true,
            paused: false,
            dirty: true,
            last_size: glam::UVec2::ZERO,
            is_loading: false,
            is_training: false,
            queue,
            device,
            renderer,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut ViewerContext,
        rect: Rect,
        splats: &Splats<Wgpu>,
    ) {
        // If this viewport is re-rendering.
        if ui.ctx().has_requested_repaint() && self.dirty {
            let _span = trace_span!("Render splats").entered();
            let size = glam::uvec2(1024, 1024);
            let (img, _) = splats.render(&context.camera, size, true);
            self.backbuffer.update_texture(img, self.renderer.clone());
            self.dirty = false;
            self.last_size = size;
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
        delta_time: Duration,
    ) -> egui::InnerResponse<()> {
        ui.horizontal(|ui| {
            if self.is_loading {
                ui.horizontal(|ui| {
                    ui.label("Loading... Please wait.");
                    ui.spinner();
                });
            }

            if self.view_splats.len() > 1 {
                self.dirty = true;

                if !self.is_loading {
                    let label = if self.paused {
                        "⏸ paused"
                    } else {
                        "⏵ playing"
                    };

                    if ui.selectable_label(!self.paused, label).clicked() {
                        self.paused = !self.paused;
                    }

                    if !self.paused {
                        self.frame += delta_time.as_secs_f32();
                        self.dirty = true;
                    }
                }
            }
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

impl ViewerPanel for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: &ViewerMessage, context: &mut ViewerContext) {
        if self.live_update {
            self.dirty = true;
        }

        match message {
            ViewerMessage::NewSource => {
                self.view_splats = vec![];
                self.paused = false;
                self.is_loading = false;
                self.is_training = false;
                self.err = None;
            }
            ViewerMessage::DoneLoading { training: _ } => {
                self.is_loading = false;
            }
            ViewerMessage::StartLoading {
                training,
                filename: _,
            } => {
                self.is_training = *training;
                self.is_loading = true;
            }
            ViewerMessage::ViewSplats {
                up_axis,
                splats,
                frame,
            } => {
                context.set_up_axis(*up_axis);

                if self.live_update {
                    self.view_splats.truncate(*frame);
                    log::info!("Received splat at {frame}");
                    self.view_splats.push(*splats.clone());
                    self.frame = *frame as f32 - 0.5;
                }
            }
            ViewerMessage::TrainStep {
                splats,
                stats: _,
                iter: _,
                timestamp: _,
            } => {
                if self.live_update {
                    self.view_splats = vec![*splats.clone()];
                }
            }
            ViewerMessage::Error(e) => {
                self.err = Some(e.clone());
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        let cur_time = Instant::now();
        let delta_time = self
            .last_draw
            .map(|last| cur_time - last)
            .unwrap_or(Duration::from_millis(10));

        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show.
        if !self.is_loading && self.view_splats.is_empty() && self.err.is_none() {
            ui.heading("Load a ply file or dataset to get started.");
            ui.add_space(5.0);
            ui.label(
                r#"
Load a pretrained .ply file to view it

Or load a dataset to train on. These are zip files with:
    - a transforms.json and images, like the nerfstudio dataset format.
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

        if let Some(err) = self.err.as_ref() {
            ui.label("Error: ".to_owned() + &err.to_string());
        } else if !self.view_splats.is_empty() {
            const FPS: usize = 24;
            let frame = ((self.frame * FPS as f32).floor() as usize) % self.view_splats.len();
            let splats = self.view_splats[frame].clone();

            let mut size = ui.available_size();
            // Always keep some margin at the bottom
            size.y -= 50.0;

            if self.is_training {
                let focal = context.camera.focal(glam::uvec2(1, 1));
                let aspect_ratio = focal.y / focal.x;
                if size.x / size.y > aspect_ratio {
                    size.x = size.y * aspect_ratio;
                } else {
                    size.y = size.x / aspect_ratio;
                }
            } else {
                let focal_y = fov_to_focal(context.camera.fov_y, size.y as u32) as f32;
                context.camera.fov_x = focal_to_fov(focal_y as f64, size.x as u32);
            }

            let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);
            let rect = context.controls.handle_user_input(ui, size, delta_time);

            let total_transform = context.model_transform * context.controls.transform;
            context.camera.position = total_transform.translation.into();
            context.camera.rotation = glam::Quat::from_mat3a(&total_transform.matrix3);
    
            context.controls.dirty = false;
    
            self.dirty |= self.last_size != size;
            self.draw_splats(ui, context, rect, &splats);
            self.show_splat_options(ui, context, &splats, delta_time);
        }

        if self.dirty {
            ui.ctx().request_repaint();
        }
    }
}
