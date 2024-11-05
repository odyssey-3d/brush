use brush_render::camera::Camera;
use egui::Margin;
use glam::{Quat, Vec3};

use super::{panel_title, PanelTypes};
use crate::{viewer::ViewerContext, ViewerPanel};

use std::format;

enum Quality {
    Low,
    Normal,
}

pub(crate) struct ViewerOptionsPanel {}

impl ViewerOptionsPanel {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

fn show_open_panel_options(
    ui: &mut egui::Ui,
    panel_type: &PanelTypes,
    context: &mut ViewerContext,
) {
    let panel_title = panel_title(panel_type).to_owned();
    let mut checked = context.open_panels.contains(&panel_title);
    let label = format!("Show {}", panel_title);
    if (ui.checkbox(&mut checked, label)).clicked() {
        if checked {
            context.open_panels.insert(panel_title);
        } else {
            context.open_panels.remove(&panel_title);
        }
    }
}

impl ViewerPanel for ViewerOptionsPanel {
    fn title(&self) -> String {
        panel_title(&PanelTypes::ViewOptions).to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        ui.add_space(5.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            ui.menu_button("⚙", |ui| {
                ui.with_layout(
                    egui::Layout::top_down_justified(egui::Align::TOP).with_main_wrap(true),
                    |ui| {
                        let mut panels_to_check =
                            vec![PanelTypes::TrainingOptions, PanelTypes::Presets];

                        #[cfg(not(target_family = "wasm"))]
                        {
                            panels_to_check.push(PanelTypes::Rerun);
                        }

                        for panel in panels_to_check {
                            show_open_panel_options(ui, &panel, context);
                        }
                    },
                );
            });
        });

        egui::Frame::default()
            .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
            .outer_margin(Margin::same(5.0))
            .inner_margin(Margin::same(5.0))
            .show(ui, |ui| {
                ui.label("Camera");
                ui.separator();
                ui.label(format!("Position: {}", context.camera.position));
                ui.label(format!("Rotation: {}", context.camera.rotation));
                ui.label(format!("Rotation: {}", context.controls.focus));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    if ui.button("Reset").clicked() {
                        context.controls.focus = Vec3::ZERO;
                        context.camera = Camera::new(
                            -Vec3::Z * 5.0,
                            Quat::IDENTITY,
                            glam::vec2(0.5, 0.5),
                            glam::vec2(0.5, 0.5),
                        );
                    }
                });
            });
    }
}
