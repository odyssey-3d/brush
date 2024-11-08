use brush_render::camera::Camera;
use egui::{Direction, Margin, Rect};
use glam::{Mat3, Quat, Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraRotateMode {
    Orbit,
    PanTilt,
}

pub struct CameraController {
    pub focus: Vec3,
    pub distance: f32,
    pub heading: f32,
    pub pitch: f32,

    pub rotate_mode: CameraRotateMode,

    pub movement_speed: f32,
    pub rotation_speed: f32,
    pub zoom_speed: f32,

    dolly_momentum: Vec3,
    rotate_momentum: Vec2,

    button_size: f32,
    fine_tuning_scalar: f32,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            focus: Vec3::ZERO,
            distance: 10.0,
            heading: 0.0,
            pitch: 0.0,
            rotate_mode: CameraRotateMode::PanTilt,
            movement_speed: 0.2,
            rotation_speed: 0.005,
            zoom_speed: 0.002,

            dolly_momentum: Vec3::ZERO,
            rotate_momentum: Vec2::ZERO,

            button_size: 20.0,
            fine_tuning_scalar: 0.2,
        }
    }

    pub fn rotate_dolly_and_zoom(
        &mut self,
        camera: &mut Camera,
        movement: Vec3,
        rotate: Vec2,
        scroll: f32,
        delta_time: f32,
    ) {
        self.zoom(camera, scroll);
        self.dolly(camera, movement, delta_time);
        match self.rotate_mode {
            CameraRotateMode::Orbit => {
                self.orbit(camera, rotate, delta_time);
            }
            CameraRotateMode::PanTilt => {
                self.pan_and_tilt(camera, rotate, delta_time);
            }
        }
    }

    pub fn zoom(&mut self, camera: &mut Camera, scroll: f32) {
        let mut radius = self.distance;
        radius -= scroll * radius * self.zoom_speed;

        let min = 0.25;
        let max = 100.0;

        if radius < min {
            radius = radius * 0.5 + min * 0.5;
        }

        if radius > max {
            radius = radius * 0.5 + max * 0.5;
        }
        self.distance = radius;
        let rot_matrix = Mat3::from_quat(camera.rotation);
        camera.position = self.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -self.distance));
    }

    pub fn dolly(&mut self, camera: &mut Camera, movement: Vec3, delta_time: f32) {
        self.dolly_momentum += movement * self.movement_speed;
        let damping = 0.0005f32.powf(delta_time);
        self.dolly_momentum *= damping;

        let pan_velocity = self.dolly_momentum * delta_time;
        let scaled_pan = pan_velocity;

        let right = camera.rotation * Vec3::X * -scaled_pan.x;
        let up = camera.rotation * Vec3::Y * -scaled_pan.y;
        let forward = camera.rotation * Vec3::Z * -scaled_pan.z;

        let translation = (right + up + forward) * self.distance;
        self.focus += translation;

        let rot_matrix = Mat3::from_quat(camera.rotation);
        camera.position = self.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -self.distance));
    }

    pub fn pan_and_tilt(&mut self, camera: &mut Camera, rotate: Vec2, delta_time: f32) {
        self.rotate_momentum += rotate * self.rotation_speed;
        let damping = 0.0005f32.powf(delta_time);
        self.rotate_momentum *= damping;

        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0;
        let delta_y = rotate_velocity.y * std::f32::consts::PI;
        let yaw = Quat::from_rotation_y(delta_x);
        let pitch = Quat::from_rotation_x(-delta_y);
        camera.rotation = yaw * camera.rotation * pitch;
        let rot_matrix = Mat3::from_quat(camera.rotation);
        self.focus = camera.position - rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -self.distance));
    }

    pub fn orbit(&mut self, camera: &mut Camera, rotate: Vec2, delta_time: f32) {
        self.rotate_momentum += rotate * self.rotation_speed;
        let damping = 0.0005f32.powf(delta_time);
        self.rotate_momentum *= damping;

        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0;
        let delta_y = rotate_velocity.y * std::f32::consts::PI;
        let yaw = Quat::from_rotation_y(delta_x);
        let pitch = Quat::from_rotation_x(-delta_y);
        camera.rotation = yaw * camera.rotation * pitch;
        let rot_matrix = Mat3::from_quat(camera.rotation);
        camera.position = self.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -self.distance));
    }

    pub fn handle_rotate(&mut self, camera: &mut Camera, rotate: Vec2, delta_time: f32) {
        match self.rotate_mode {
            CameraRotateMode::Orbit => {
                self.orbit(camera, rotate, delta_time);
            }
            CameraRotateMode::PanTilt => {
                let rotate = Vec2::new(rotate.x, -rotate.y);
                self.pan_and_tilt(camera, rotate, delta_time);
            }
        }
    }

    pub fn is_animating(&self) -> bool {
        self.dolly_momentum.length_squared() > 1e-2 || self.rotate_momentum.length_squared() > 1e-2
    }

    fn check_for_dolly(
        &mut self,
        ui: &mut egui::Ui,
        camera: &mut Camera,
        delta_time: std::time::Duration,
    ) {
        let mut dolly_x = 0.0;
        let mut dolly_y = 0.0;
        let mut dolly_z = 0.0;

        if ui.input(|r| r.key_down(egui::Key::E)) {
            dolly_y += 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::Q)) {
            dolly_y -= 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::A)) {
            dolly_x += 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::D)) {
            dolly_x -= 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::W)) {
            dolly_z -= 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::S)) {
            dolly_z += 1.0;
        }

        if ui.input(|r| r.modifiers.shift_only()) {
            dolly_x *= self.fine_tuning_scalar;
            dolly_y *= self.fine_tuning_scalar;
            dolly_z *= self.fine_tuning_scalar;
        }

        self.dolly(
            camera,
            Vec3::new(dolly_x, dolly_y, dolly_z),
            delta_time.as_secs_f32(),
        );
    }

    fn check_for_pan_tilt(
        &mut self,
        ui: &mut egui::Ui,
        camera: &mut Camera,
        delta_time: std::time::Duration,
    ) {
        let mut rotate_x = 0.0;
        let mut rotate_y = 0.0;
        if ui.input(|r| r.key_down(egui::Key::ArrowRight)) {
            rotate_x += 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::ArrowLeft)) {
            rotate_x -= 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::ArrowUp)) {
            rotate_y += 1.0;
        }
        if ui.input(|r| r.key_down(egui::Key::ArrowDown)) {
            rotate_y -= 1.0;
        }

        if ui.input(|r| r.modifiers.shift_only()) {
            rotate_x *= self.fine_tuning_scalar;
            rotate_y *= self.fine_tuning_scalar;
        }

        self.handle_rotate(
            camera,
            Vec2::new(rotate_x, rotate_y) * 20.0,
            delta_time.as_secs_f32(),
        );
    }

    pub fn handle_user_input(
        &mut self,
        ui: &mut egui::Ui,
        size: glam::UVec2,
        camera: &mut Camera,
        delta_time: std::time::Duration,
    ) -> Rect {
        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2::new(size.x as f32, size.y as f32),
            egui::Sense::drag(),
        );

        let mouse_delta = glam::vec2(response.drag_delta().x, response.drag_delta().y);
        let scrolled = ui.input(|r| r.smooth_scroll_delta).y;

        let (movement, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
            (Vec2::ZERO, mouse_delta)
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            (mouse_delta, Vec2::ZERO)
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };

        let movement = Vec3::new(movement.x, movement.y, 0.0);

        self.rotate_dolly_and_zoom(camera, movement, rotate, scrolled, delta_time.as_secs_f32());
        self.check_for_dolly(ui, camera, delta_time);
        self.check_for_pan_tilt(ui, camera, delta_time);

        rect
    }

    pub fn show_ui_controls(&mut self, ui: &mut egui::Ui, camera: &mut Camera) {
        egui::Frame::default()
            .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
            .outer_margin(Margin::same(5.0))
            .inner_margin(Margin::same(5.0))
            .show(ui, |ui| {
                ui.label("Camera");
                ui.separator();

                self.draw_control_buttons(ui, camera);

                ui.separator();
                ui.label(format!("Focus: {}", self.focus));
                ui.label(format!("Position: {}", camera.position));
                ui.label(format!("Rotation: {}", camera.rotation));
            });
    }

    fn draw_control_buttons(&mut self, ui: &mut egui::Ui, camera: &mut Camera) {
        ui.style_mut().interaction.tooltip_delay = 0.01;
        ui.style_mut().interaction.tooltip_grace_time = 0.01;
        if ui.button("Reset").clicked() {
            self.focus = Vec3::ZERO;
            camera.position = -Vec3::Z * 5.0;
            camera.rotation = Quat::IDENTITY;
            camera.center_uv = glam::vec2(0.5, 0.5);
        }
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.rotate_mode, CameraRotateMode::Orbit, "Orbit");
            ui.radio_value(&mut self.rotate_mode, CameraRotateMode::PanTilt, "Pan/Tilt");
        });

        ui.with_layout(
            egui::Layout::centered_and_justified(Direction::LeftToRight),
            |ui| {
                ui.vertical(|ui| {
                    ui.add_sized(
                        [self.button_size * 3.5, self.button_size],
                        egui::Label::new("Movement"),
                    );
                    egui::Grid::new("movement")
                        .spacing([0.0, 0.0])
                        .min_col_width(25.0)
                        .min_row_height(25.0)
                        .show(ui, |ui| {
                            self.draw_empty_cell(ui);
                            self.draw_button(ui, camera, "⬆", "W", &|controller, camera| {
                                controller.dolly(camera, -Vec3::Z, 0.5)
                            });
                            self.draw_empty_cell(ui);
                            ui.end_row();

                            self.draw_button(ui, camera, "⬅", "A", &|controller, camera| {
                                controller.dolly(camera, -Vec3::X, 0.5)
                            });
                            self.draw_button(ui, camera, "⬇", "S", &|controller, camera| {
                                controller.dolly(camera, Vec3::Z, 0.5)
                            });
                            self.draw_button(ui, camera, "➡", "D", &|controller, camera| {
                                controller.dolly(camera, Vec3::X, 0.5)
                            });
                        });
                    ui.add_sized(
                        [self.button_size * 3.5, self.button_size],
                        egui::Label::new("In/Out L/R"),
                    );
                });

                ui.vertical(|ui| {
                    ui.add_sized(
                        [self.button_size * 3.5, self.button_size],
                        egui::Label::new("Vertical"),
                    );
                    egui::Grid::new("vertical")
                        .spacing([0.0, 1.0])
                        .min_col_width(25.0)
                        .min_row_height(25.0)
                        .show(ui, |ui| {
                            self.draw_empty_cell(ui);
                            self.draw_button(ui, camera, "⏫", "E", &|controller, camera| {
                                controller.dolly(camera, Vec3::Y, 0.5)
                            });
                            self.draw_empty_cell(ui);
                            ui.end_row();
                            self.draw_empty_cell(ui);
                            self.draw_button(ui, camera, "⏬", "Q", &|controller, camera| {
                                controller.dolly(camera, -Vec3::Y, 0.5)
                            });
                            self.draw_empty_cell(ui);
                        });
                    ui.add_sized(
                        [self.button_size * 3.5, self.button_size],
                        egui::Label::new("Up/Down"),
                    );
                });

                ui.vertical(|ui| {
                    ui.add_sized(
                        [self.button_size * 3.5, self.button_size],
                        egui::Label::new("Rotation"),
                    );
                    egui::Grid::new("rotation")
                        .spacing([0.0, 1.0])
                        .min_col_width(25.0)
                        .min_row_height(25.0)
                        .show(ui, |ui| {
                            self.draw_empty_cell(ui);
                            self.draw_button(ui, camera, "⤴", "⬆", &|controller, camera| {
                                controller.handle_rotate(camera, glam::vec2(0.0, 100.0), 0.5);
                            });
                            self.draw_empty_cell(ui);
                            ui.end_row();
                            self.draw_button(ui, camera, "⮪", "⬅", &|controller, camera| {
                                controller.handle_rotate(camera, glam::vec2(-100.0, 0.0), 0.5);
                            });
                            self.draw_button(ui, camera, "⤵", "⬇", &|controller, camera| {
                                controller.handle_rotate(camera, glam::vec2(0.0, -100.0), 0.5);
                            });
                            self.draw_button(ui, camera, "⮫", "➡", &|controller, camera| {
                                controller.handle_rotate(camera, glam::vec2(100.0, 0.0), 0.5);
                            });
                        });
                    ui.add_sized(
                        [self.button_size * 3.5, self.button_size],
                        egui::Label::new("Pan/Tilt"),
                    );
                });
            },
        );
    }

    fn draw_button(
        &mut self,
        ui: &mut egui::Ui,
        camera: &mut Camera,
        text: &str,
        tooltip: &str,
        func: &dyn Fn(&mut CameraController, &mut Camera),
    ) {
        if ui
            .add_sized(
                [self.button_size, self.button_size],
                egui::Button::new(text),
            )
            .on_hover_text_at_pointer(tooltip)
            .clicked()
        {
            func(self, camera);
        }
    }
    fn draw_empty_cell(&self, ui: &mut egui::Ui) {
        ui.add_sized([self.button_size, self.button_size], egui::Label::new(" "));
    }
}
