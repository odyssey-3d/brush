use super::{panel_title, PanelTypes};
use crate::{viewer::ViewerContext, ViewerPanel};

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
        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::TOP).with_main_wrap(true),
            |ui| {
                let mut panels_to_check = vec![
                    PanelTypes::TrainingOptions,
                    PanelTypes::Presets,
                    PanelTypes::Stats,
                ];

                #[cfg(not(target_family = "wasm"))]
                {
                    panels_to_check.push(PanelTypes::Rerun);
                }

                for panel in panels_to_check {
                    show_open_panel_options(ui, &panel, context);
                }
            },
        );
    }
}

