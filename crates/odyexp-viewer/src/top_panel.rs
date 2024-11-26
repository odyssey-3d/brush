use eframe::egui;

use crate::{
    app_context::{ViewerContext, ViewerMessage},
    load::DataSource,
};

pub(crate) struct TopPanel {
    frame: egui::Frame,
    is_loading: bool,
}

impl TopPanel {
    pub(crate) fn new() -> Self {
        Self {
            frame: egui::containers::Frame {
                inner_margin: egui::epaint::Margin::same(0.0),
                outer_margin: egui::epaint::Margin::same(0.0),
                rounding: egui::Rounding::ZERO,
                shadow: eframe::epaint::Shadow::default(),
                fill: egui::Color32::BLACK,
                stroke: egui::Stroke::default(),
            },
            is_loading: false,
        }
    }

    pub(crate) fn on_message(&mut self, message: &ViewerMessage, _context: &mut ViewerContext) {
        match message {
            ViewerMessage::StartLoading { .. } => {
                self.is_loading = true;
            }
            ViewerMessage::DoneLoading => {
                self.is_loading = false;
            }
            _ => {}
        }
    }

    pub fn show(&mut self, app_context: &mut ViewerContext) {
        let panel_height = app_context.ui_layout.top_panel_height;

        let ctx = app_context.egui_ctx.clone();
        let screen_rect = ctx.input(|i: &egui::InputState| i.screen_rect());

        egui::TopBottomPanel::top("wrap_app_top_bar")
            .default_height(panel_height)
            .frame(self.frame)
            .show(&ctx, |ui| {
                ui.with_layout(
                    egui::Layout::left_to_right(egui::Align::Center).with_main_justify(true),
                    |ui| {
                        ui.scope(|ui| {
                            let button_height = 36.0;

                            ui.style_mut().text_styles.insert(
                                egui::TextStyle::Button,
                                egui::FontId::new(24.0, eframe::epaint::FontFamily::Proportional),
                            );
                            ui.style_mut().text_styles.insert(
                                egui::TextStyle::Heading,
                                egui::FontId::new(24.0, eframe::epaint::FontFamily::Proportional),
                            );

                            ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                                egui::Color32::BLACK;
                            ui.style_mut().visuals.widgets.active.weak_bg_fill =
                                egui::Color32::BLACK;
                            ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                                egui::Color32::BLACK;

                            ui.style_mut().visuals.widgets.active.bg_stroke =
                                egui::Stroke::new(1.0, egui::Color32::DARK_GRAY);
                            ui.style_mut().visuals.widgets.inactive.bg_stroke =
                                egui::Stroke::new(1.0, egui::Color32::DARK_GRAY);
                            ui.style_mut().visuals.widgets.hovered.bg_stroke =
                                egui::Stroke::new(1.0, egui::Color32::GRAY);

                            if let Some(filename) = &app_context.filename {
                                ui.heading(filename);
                            } else {
                                ui.heading("<No Scene loaded>");
                            }

                            let right_side = screen_rect.width();

                            let margin = 5.0;
                            let offset = (panel_height - button_height) / 2.0;

                            let button_pos = egui::pos2(margin, offset);

                            let button_size = egui::vec2(button_height, button_height);

                            if ui
                                .put(
                                    egui::Rect::from_min_size(button_pos, button_size),
                                    egui::ImageButton::new(egui::Image::new(egui::include_image!(
                                        "../assets/back-arrow.png"
                                    )))
                                    .frame(false)
                                    .rounding(egui::Rounding::same(margin)),
                                )
                                .clicked()
                            {
                                app_context.start_ply_load(DataSource::PickFile);
                            };

                            let button_pos =
                                egui::pos2(button_pos.x + button_size.x + margin, button_pos.y);

                            ui.put(
                                egui::Rect::from_min_size(button_pos, button_size),
                                egui::Image::new(egui::include_image!("../assets/avatar.png")),
                            );

                            // top right corner
                            let button_pos = egui::pos2(right_side - 2.0 * margin, button_pos.y);
                            let download =
                                egui::Image::new(egui::include_image!("../assets/download.png"));
                            let button_pos =
                                egui::pos2(button_pos.x - button_size.x - margin, button_pos.y);
                            if ui
                                .put(
                                    egui::Rect::from_min_size(button_pos, button_size),
                                    egui::ImageButton::new(download)
                                        .rounding(egui::Rounding::same(margin)),
                                )
                                .clicked()
                            {
                                println!("clicked");
                            };

                            if self.is_loading {
                                let button_pos =
                                    egui::pos2(button_pos.x - button_size.x - margin, margin);
                                ui.put(
                                    egui::Rect::from_min_size(button_pos, button_size),
                                    egui::Spinner::new(),
                                );
                            }
                        });
                    },
                );
            });
    }
}
