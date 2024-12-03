use std::sync::Arc;

use eframe::{egui, egui_wgpu::Renderer};
use egui::epaint::mutex::RwLock as EguiRwLock;
use egui::Color32;

use crate::{
    app_context::{ViewerContext, ViewerMessage},
    scene_panel::ScenePanel,
};

pub(crate) struct MainPanel {
    frame: egui::Frame,
    scene: Box<ScenePanel>,
}

impl MainPanel {
    pub(crate) fn new(
        queue: Arc<wgpu::Queue>,
        device: Arc<wgpu::Device>,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        Self {
            scene: Box::new(ScenePanel::new(queue, device, renderer)),
            frame: egui::Frame {
                inner_margin: egui::epaint::Margin::same(0.0),
                outer_margin: egui::epaint::Margin::same(0.0),
                rounding: egui::Rounding::ZERO,
                shadow: eframe::epaint::Shadow::default(),
                stroke: egui::Stroke::default(),
                fill: Color32::BLACK,
            },
        }
    }

    pub(crate) fn on_message(&mut self, message: &ViewerMessage, context: &mut ViewerContext) {
        self.scene.on_message(message, context);
    }

    pub fn show(&mut self, app_context: &ViewerContext) {
        egui::CentralPanel::default()
            .frame(self.frame)
            .show(&app_context.egui_ctx, |ui| {
                // Close when pressing escape (in a native viewer anyway).
                if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
                    app_context
                        .egui_ctx
                        .send_viewport_cmd(egui::ViewportCommand::Close);
                }
                self.scene.show(ui, app_context);
            });
    }
}
