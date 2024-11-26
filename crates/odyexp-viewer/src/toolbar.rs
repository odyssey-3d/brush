use eframe::egui;
use egui::Color32;

use crate::app_context::{ViewerContext, ViewerMessage};

pub(crate) struct Toolbar {
    frame: egui::Frame,
}

impl Toolbar {
    pub(crate) fn new() -> Self {
        Self {
            frame: egui::Frame {
                inner_margin: egui::epaint::Margin::same(0.0),
                outer_margin: egui::epaint::Margin::same(6.0),
                rounding: egui::Rounding::same(12.0),
                shadow: eframe::epaint::Shadow::default(),
                stroke: egui::Stroke::new(0.5, Color32::DARK_GRAY.gamma_multiply(0.5)),
                fill: Color32::BLACK,
            }
            .multiply_with_opacity(0.5),
        }
    }

    pub(crate) fn on_message(&mut self, _message: &ViewerMessage, _context: &mut ViewerContext) {}

    pub fn show(&mut self, app_context: &mut ViewerContext) {
        let top_panel_height = app_context.ui_layout.top_panel_height;

        let ctx = &app_context.egui_ctx.clone();
        let outer_margin = self.frame.outer_margin.left_top();
        let button_rounding = self.frame.rounding.ne;

        let button_ratio = 0.03; // relative to screen height

        let screen_rect = ctx.input(|i: &egui::InputState| i.screen_rect());

        let button_size = button_ratio * screen_rect.height();
        let margin = 0.2 * button_size;
        let toolbar_width = button_size + 2.0 * margin;

        let position = egui::pos2(
            0.5 * button_size + margin,
            0.4 * screen_rect.height() - top_panel_height,
        );

        let button_size = egui::vec2(button_size, button_size);

        // let window_rect = self.draw_edit_tools(
        //     ctx,
        //     position,
        //     outer_margin,
        //     margin,
        //     button_size,
        //     button_rounding,
        //     toolbar_width,
        // );

        // let position = egui::pos2(position.x, window_rect.top());

        self.draw_camera_tools(
            ctx,
            position,
            outer_margin,
            margin,
            button_size,
            button_rounding,
            toolbar_width,
        );
    }

    fn draw_camera_tools(
        &mut self,
        ctx: &egui::Context,
        position: egui::Pos2,
        outer_margin: egui::Vec2,
        margin: f32,
        button_size: egui::Vec2,
        button_rounding: f32,
        toolbar_width: f32,
    ) {
        self.tool_group(&ctx, position, |ui| {
            let button_pos = egui::pos2(
                position.x + outer_margin.x + margin,
                position.y + outer_margin.y + margin,
            );
            if self
                .tool_button(
                    ui,
                    egui::Image::new(egui::include_image!("../assets/camera.png")),
                    button_pos,
                    button_size,
                    button_rounding,
                    true,
                )
                .clicked()
            {
                println!("camera button clicked");
            };

            ui.allocate_space(egui::vec2(toolbar_width, outer_margin.y));
            ui.cursor()
        });
    }
    #[allow(dead_code)]
    fn draw_edit_tools(
        &mut self,
        ctx: &egui::Context,
        position: egui::Pos2,
        outer_margin: egui::Vec2,
        margin: f32,
        button_size: egui::Vec2,
        button_rounding: f32,
        toolbar_width: f32,
    ) -> egui::Rect {
        let window_rect = self.tool_group(&ctx, position, |ui| {
            let mut button_pos = egui::pos2(
                position.x + outer_margin.x + margin,
                position.y + outer_margin.y + margin,
            );
            if self
                .tool_button(
                    ui,
                    egui::Image::new(egui::include_image!("../assets/brush.png")),
                    button_pos,
                    button_size,
                    button_rounding,
                    false,
                )
                .clicked()
            {};
            button_pos.y += button_size.y + margin * 2.0;
            if self
                .tool_button(
                    ui,
                    egui::Image::new(egui::include_image!("../assets/lighting.png")),
                    button_pos,
                    button_size,
                    button_rounding,
                    false,
                )
                .clicked()
            {
                println!("download button clicked");
            };
            ui.allocate_space(egui::vec2(toolbar_width, outer_margin.y));

            ui.cursor()
        });
        window_rect
    }

    fn tool_group(
        &self,
        ctx: &egui::Context,
        window_position: egui::Pos2,
        add_contents: impl FnOnce(&mut egui::Ui) -> egui::Rect,
    ) -> egui::Rect {
        let window_rect = egui::Window::new(format!("toolbar {}", window_position))
            .frame(self.frame)
            .collapsible(false)
            .title_bar(false)
            .resizable(false)
            .fixed_pos(window_position)
            .show(ctx, add_contents)
            .unwrap()
            .inner
            .unwrap();

        window_rect
    }

    fn tool_button(
        &self,
        ui: &mut egui::Ui,
        button_image: egui::Image,
        button_pos: egui::Pos2,
        button_size: egui::Vec2,
        rounding: f32,
        is_active: bool,
    ) -> egui::Response {
        ui.scope(|ui| {
            let color = if is_active {
                egui::Color32::from_rgb(42, 102, 228)
            } else {
                egui::Color32::TRANSPARENT
            };

            ui.style_mut().visuals.widgets.inactive.weak_bg_fill = color;
            ui.style_mut().visuals.widgets.active.weak_bg_fill = color;
            ui.style_mut().visuals.widgets.hovered.weak_bg_fill = color;
            ui.style_mut().visuals.widgets.noninteractive.weak_bg_fill = color;

            ui.style_mut().visuals.widgets.active.bg_stroke = egui::Stroke::NONE;
            ui.style_mut().visuals.widgets.inactive.bg_stroke = egui::Stroke::NONE;
            ui.style_mut().visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
            if is_active {
                ui.style_mut().visuals.widgets.hovered.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::GRAY);
            } else {
                ui.style_mut().visuals.widgets.hovered.bg_stroke = egui::Stroke::NONE;
            }
            ui.put(
                egui::Rect::from_min_size(button_pos, button_size),
                egui::ImageButton::new(button_image.bg_fill(Color32::TRANSPARENT))
                    .rounding(egui::Rounding::same(rounding))
                    .sense(if is_active {
                        egui::Sense::click()
                    } else {
                        egui::Sense::hover()
                    }),
            )
        })
        .inner
    }
}
