use brush_dataset::splat_export;

use std::ops::Range;
use std::sync::Arc;

use burn::tensor::{Distribution, Tensor};

use brush_render::gaussian_splats::Splats;
use brush_ui::burn_texture::BurnTexture;

use eframe::egui_wgpu::Renderer;
use egui::epaint::mutex::RwLock as EguiRwLock;
use egui::{Color32, Rect};
use glam::Affine3A;
use tracing::trace_span;
use web_time::Instant;

use crate::{
    train_loop::TrainMessage,
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};

pub(crate) struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,
    pub(crate) last_message: Option<ViewerMessage>,

    is_loading: bool,
    is_training: bool,
    live_update: bool,
    paused: bool,

    last_cam_trans: Affine3A,
    dirty: bool,

    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    renderer: Arc<EguiRwLock<Renderer>>,

    splats: Option<Splats<brush_render::PrimaryBackend>>,
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
            splats: None,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut ViewerContext,
        rect: Rect,
        background: glam::Vec3,
    ) {
        let splats = self.splats.as_ref().unwrap();
        // If this viewport is re-rendering.
        if ui.ctx().has_requested_repaint() && self.dirty {
            let _span = trace_span!("Render splats").entered();
            let size = glam::uvec2(1024, 1024);
            let (img, _) = splats.render(&context.camera, size, background, true);
            self.backbuffer.update_texture(img, self.renderer.clone());
            self.dirty = false;
        }

        if let Some(id) = self.backbuffer.id() {
            ui.scope(|ui| {
                ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
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
    ) -> egui::InnerResponse<()> {
        let splats = self.splats.as_mut().unwrap();
        ui.horizontal(|ui| {
            ui.add_space(10.0);
            if ui.button("Select All").clicked() {
                let s = Tensor::ones_like(&splats.selected);
                splats.set_selected(s);
            }
            if ui.button("Select None").clicked() {
                let s = Tensor::zeros_like(&splats.selected);
                splats.set_selected(s);
            }
            if ui.button("Select Multi").clicked() {
                let s = Tensor::zeros_like(&splats.selected);
                let num_splats = splats.num_splats() / 1000;
                let s = s.slice_assign(
                    [
                        Range {
                            start: 0,
                            end: num_splats,
                        },
                    ],
                    Tensor::ones([num_splats], &context.device) * 2.0,
                );
                let s = s.slice_assign(
                    [
                        Range {
                            start: num_splats,
                            end: 2*num_splats,
                        },
                    ],
                    Tensor::ones([num_splats], &context.device),
                );
                let s = s.slice_assign(
                    [
                        Range {
                            start: 2*num_splats,
                            end: 3*num_splats,
                        },
                    ],
                    Tensor::ones([num_splats], &context.device) * 3.0,
                );
                splats.set_selected(s);
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

                    #[cfg(target_family = "wasm")]
                    async_std::task::spawn_local(fut);
                    #[cfg(not(target_family = "wasm"))]
                    async_std::task::spawn(fut);
                }
            }
        })
    }
}

impl ViewerPanel for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: ViewerMessage, _: &mut ViewerContext) {
        match message.clone() {
            ViewerMessage::PickFile => {
                self.last_message = None;
                self.splats = None;
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
                if !training {
                    self.live_update = true;
                }
            }
            ViewerMessage::Splats { iter: _, splats } => {
                if self.live_update {
                    self.dirty = true;
                    self.last_message = Some(message);
                    self.splats = Some(*splats);
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
                ui.heading("Note: Running in browser is experimental");

                ui.label(
                    r#"
In browser training is about 2x lower than the native app. For bigger training
runs consider using the native app."#,
                );
            });

            return;
        }

        if self.last_cam_trans
            != glam::Affine3A::from_rotation_translation(
                context.camera.rotation,
                context.camera.position,
            )
        {
            self.dirty = true;
        }

        if let Some(message) = self.last_message.clone() {
            match message {
                ViewerMessage::Error(e) => {
                    ui.label("Error: ".to_owned() + &e.to_string());
                }
                ViewerMessage::Splats { iter: _, splats: _ } => {
                    // let splats = self.splats.as_mut().unwrap();
                    let mut size = ui.available_size();
                    let focal = context.camera.focal(glam::uvec2(1, 1));
                    let aspect_ratio = focal.y / focal.x;
                    if size.x / size.y > aspect_ratio {
                        size.x = size.y * aspect_ratio;
                    } else {
                        size.y = size.x / aspect_ratio;
                    }

                    let cur_time = Instant::now();

                    if let Some(last_draw) = self.last_draw {
                        let delta_time = cur_time - last_draw;
                        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);
                        let rect = context.controls.handle_user_input(
                            ui,
                            size,
                            &mut context.camera,
                            delta_time,
                        );
                        self.draw_splats(ui, context, rect, context.dataset.train.background);

                        self.show_splat_options(ui, context);
                    }
                    self.last_draw = Some(cur_time);
                }
                _ => {}
            }
        }
    }
}
