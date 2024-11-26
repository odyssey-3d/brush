use core::f32;
use egui::Rect;
use glam::{Affine3A, Mat3A, Vec2, Vec3A};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraRotateMode {
    Orbit,
    PanTilt,
}

pub struct CameraController {
    pub transform: Affine3A,

    pub focus: Vec3A,
    pub dirty: bool,

    pub distance: f32,

    pub rotate_mode: CameraRotateMode,

    pub movement_speed: f32,
    pub rotation_speed: f32,
    pub zoom_speed: f32,

    dolly_momentum: Vec3A,
    rotate_momentum: Vec2,

    fine_tuning_scalar: f32,
}

impl CameraController {
    pub fn new(transform: Affine3A) -> Self {
        Self {
            transform,
            focus: Vec3A::ZERO,
            dirty: false,

            distance: 10.0,
            rotate_mode: CameraRotateMode::PanTilt,
            movement_speed: 0.2,
            rotation_speed: 0.005,
            zoom_speed: 0.002,

            dolly_momentum: Vec3A::ZERO,
            rotate_momentum: Vec2::ZERO,

            fine_tuning_scalar: 0.2,
        }
    }

    pub fn rotate_dolly_and_zoom(
        &mut self,
        movement: Vec3A,
        rotate: Vec2,
        scroll: f32,
        delta_time: f32,
    ) {
        self.zoom(scroll);
        self.dolly(movement, delta_time);
        match self.rotate_mode {
            CameraRotateMode::Orbit => {
                self.orbit(rotate, delta_time);
            }
            CameraRotateMode::PanTilt => {
                self.pan_and_tilt(rotate, delta_time);
            }
        }
    }

    pub fn zoom(&mut self, scroll: f32) {
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

        let rotation = self.transform.matrix3;
        self.transform.translation = self.focus + rotation * Vec3A::new(0.0, 0.0, -self.distance);
        self.transform.matrix3 = rotation;
    }

    pub fn dolly(&mut self, movement: Vec3A, delta_time: f32) {
        let rotation = self.transform.matrix3;

        self.dolly_momentum += movement * self.movement_speed;
        let damping = 0.0005f32.powf(delta_time);
        self.dolly_momentum *= damping;

        let pan_velocity = self.dolly_momentum * delta_time;
        let scaled_pan = pan_velocity;

        let right = rotation * Vec3A::X * -scaled_pan.x;
        let up = rotation * Vec3A::Y * -scaled_pan.y;
        let forward = rotation * Vec3A::Z * -scaled_pan.z;

        let translation = (right + up + forward) * self.distance;
        self.focus += translation;

        let rotation = self.transform.matrix3;
        self.transform.translation = self.focus + rotation * Vec3A::new(0.0, 0.0, -self.distance);
    }

    pub fn pan_and_tilt(&mut self, rotate: Vec2, delta_time: f32) {
        self.rotate_momentum += rotate * self.rotation_speed;
        let damping = 0.0005f32.powf(delta_time);
        self.rotate_momentum *= damping;

        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0;
        let delta_y = rotate_velocity.y * std::f32::consts::PI;
        let yaw = Mat3A::from_rotation_y(delta_x);
        let pitch = Mat3A::from_rotation_x(-delta_y);

        self.transform.matrix3 = yaw * self.transform.matrix3 * pitch;
        self.focus = self.transform.translation
            - self.transform.matrix3 * Vec3A::new(0.0, 0.0, -self.distance);
    }

    pub fn orbit(&mut self, rotate: Vec2, delta_time: f32) {
        self.rotate_momentum += rotate * self.rotation_speed;
        let damping = 0.0005f32.powf(delta_time);
        self.rotate_momentum *= damping;

        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0;
        let delta_y = rotate_velocity.y * std::f32::consts::PI;
        let yaw = Mat3A::from_rotation_y(delta_x);
        let pitch = Mat3A::from_rotation_x(-delta_y);
        self.transform.matrix3 = yaw * self.transform.matrix3 * pitch;
        self.transform.translation =
            self.focus + self.transform.matrix3 * Vec3A::new(0.0, 0.0, -self.distance);
    }

    pub fn handle_rotate(&mut self, rotate: Vec2, delta_time: f32) {
        match self.rotate_mode {
            CameraRotateMode::Orbit => {
                self.orbit(rotate, delta_time);
            }
            CameraRotateMode::PanTilt => {
                let rotate = Vec2::new(rotate.x, -rotate.y);
                self.pan_and_tilt(rotate, delta_time);
            }
        }
    }

    fn check_for_dolly(&mut self, ui: &mut egui::Ui, delta_time: std::time::Duration) {
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
            Vec3A::new(dolly_x, dolly_y, dolly_z),
            delta_time.as_secs_f32(),
        );
    }

    fn check_for_pan_tilt(&mut self, ui: &mut egui::Ui, delta_time: std::time::Duration) {
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

        self.rotate_mode = if ui.input(|r| r.modifiers.command_only()) {
            CameraRotateMode::Orbit
        } else {
            CameraRotateMode::PanTilt
        };

        if rotate_x.abs() > 0.0 || rotate_y.abs() > 0.0 {
            self.handle_rotate(
                Vec2::new(rotate_x, rotate_y) * 20.0,
                delta_time.as_secs_f32(),
            );
        }
    }

    pub fn handle_user_input(
        &mut self,
        ui: &mut egui::Ui,
        size: glam::UVec2,
        delta_time: std::time::Duration,
    ) -> Rect {
        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2::new(size.x as f32, size.y as f32),
            egui::Sense::drag(),
        );

        let mouse_delta = glam::vec2(response.drag_delta().x, response.drag_delta().y);
        let scrolled = ui.input(|r| {
            r.smooth_scroll_delta.y
                + r.multi_touch()
                    .map(|t| (t.zoom_delta - 1.0) * self.distance * 5.0)
                    .unwrap_or(0.0)
        });

        let (movement, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
            (Vec2::ZERO, mouse_delta)
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            (mouse_delta, Vec2::ZERO)
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };

        let movement = Vec3A::new(movement.x, movement.y, 0.0);

        self.check_for_dolly(ui, delta_time);
        self.check_for_pan_tilt(ui, delta_time);
        self.rotate_dolly_and_zoom(movement, rotate, scrolled, delta_time.as_secs_f32());

        self.dirty = scrolled.abs() > 0.0
            || movement.length_squared() > 0.0
            || rotate.length_squared() > 0.0
            || self.dolly_momentum.length_squared() > 0.001
            || self.rotate_momentum.length_squared() > 0.001
            || self.dirty;

        rect
    }
}
