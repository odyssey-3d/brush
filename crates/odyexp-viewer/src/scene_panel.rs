use std::{sync::Arc, time::Duration};

use tracing::trace_span;

use brush_render::{
    camera::{focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use brush_ui::burn_texture::BurnTexture;

use burn_wgpu::Wgpu;
use core::f32;

use eframe::egui_wgpu::Renderer;
use egui::epaint::mutex::RwLock as EguiRwLock;

use egui::{Color32, Rect};

use web_time::Instant;

use crate::app_context::{ViewerContext, ViewerMessage};

use crate::draw::Grid;

type Backend = Wgpu;

pub(crate) struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    frame: f32,
    err: Option<Arc<anyhow::Error>>,

    is_loading: bool,
    is_paused: bool,

    last_size: glam::UVec2,
    dirty: bool,

    renderer: Arc<EguiRwLock<Renderer>>,

    grid: Grid,
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
            dirty: true,
            last_size: glam::UVec2::ZERO,
            is_loading: false,
            is_paused: false,
            renderer,
            grid: Grid::new(16, 0.5).with_color(Color32::from_gray(117).gamma_multiply(0.2)),
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        context: &ViewerContext,
        size: glam::UVec2,
        rect: Rect,
        splats: &Splats<Backend>,
    ) {
        if self.dirty {
            let _span = trace_span!("Render splats").entered();
            let (img, _) = splats.render(&context.camera, size, true);
            self.backbuffer.update_texture(img, self.renderer.clone());
            self.dirty = false;
            self.last_size = size;
        }

        if let Some(id) = self.backbuffer.id() {
            ui.scope(|ui| {
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
        context: &ViewerContext,
        delta_time: Duration,
    ) -> egui::InnerResponse<()> {
        ui.horizontal(|ui| {
            if self.is_loading {
                ui.horizontal(|ui| {
                    ui.label("Loading... Please wait.");
                    ui.spinner();
                });
            }

            if context.view_splats.len() > 1 {
                self.dirty = true;

                if !self.is_loading {
                    let label = if self.is_paused {
                        "⏸ paused"
                    } else {
                        "⏵ playing"
                    };

                    if ui.selectable_label(!self.is_paused, label).clicked() {
                        self.is_paused = !self.is_paused;
                    }

                    if !self.is_paused {
                        self.frame += delta_time.as_secs_f32();
                        self.dirty = true;
                    }
                }
            }
        })
    }

    pub(crate) fn on_message(&mut self, message: &ViewerMessage, _context: &mut ViewerContext) {
        self.dirty = true;

        match message {
            ViewerMessage::NewSource => {
                self.is_paused = false;
                self.is_loading = false;
                self.err = None;
            }
            ViewerMessage::DoneLoading => {
                self.is_loading = false;
            }
            ViewerMessage::StartLoading { filename: _ } => {
                self.is_loading = true;
            }
            ViewerMessage::ViewSplats {
                up_axis: _,
                splats: _,
                frame: _,
            } => {}
            ViewerMessage::Error(e) => {
                self.err = Some(e.clone());
            } // _ => {}
        }
    }

    pub(crate) fn show(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        let cur_time = Instant::now();
        let delta_time = self
            .last_draw
            .map(|last| cur_time - last)
            .unwrap_or(Duration::from_millis(10));

        self.last_draw = Some(cur_time);

        let mut size = ui.available_size();
        // Always keep some margin at the bottom
        size.y -= 50.0;

        if size.x < 8.0 || size.y < 8.0 {
            return;
        }
        let focal_y = fov_to_focal(context.camera.fov_y, size.y as u32) as f32;
        context.camera.fov_x = focal_to_fov(focal_y as f64, size.x as u32);

        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);
        let rect = context.controls.handle_user_input(ui, size, delta_time);

        self.dirty |= context.controls.dirty;
        context.update_camera();
        context.controls.dirty = false;

        self.dirty |= self.last_size != size;

        let viewport = rect;
        let view_matrix = context.camera.world_to_local();
        let projection_matrix = glam::Mat4::perspective_infinite_lh(
            context.camera.fov_y as f32,
            (viewport.width() / viewport.height()).into(),
            0.1,
        );

        let mvp = projection_matrix * view_matrix;

        self.grid.draw(ui.painter(), rect, mvp);
        if !context.view_splats.is_empty() {
            if let Some(splats) = context.current_splats() {
                self.draw_splats(ui, context, size, rect, splats);
            }
            self.show_splat_options(ui, context, delta_time);
        }
    }
}
