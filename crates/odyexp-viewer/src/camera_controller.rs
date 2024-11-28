use core::f32;
use std::ops::Range;

use egui::Rect;
use glam::{Affine3A, EulerRot, Quat, Vec2, Vec3A};

pub(crate) struct CameraSettings {
    pub focal: f64,
    pub radius: f32,

    pub yaw_range: Range<f32>,
    pub pitch_range: Range<f32>,
    pub radius_range: Range<f32>,
}

pub(crate) fn parse_camera_settings(
    search_params: std::collections::HashMap<String, String>,
) -> CameraSettings {
    let focal = search_params
        .get("focal")
        .and_then(|f| f.parse().ok())
        .unwrap_or(0.5);
    let radius = search_params
        .get("radius")
        .and_then(|f| f.parse().ok())
        .unwrap_or(4.0);
    let min_radius = search_params
        .get("min_radius")
        .and_then(|f| f.parse().ok())
        .unwrap_or(1.0);
    let max_radius = search_params
        .get("max_radius")
        .and_then(|f| f.parse().ok())
        .unwrap_or(100.0);

    let min_yaw = search_params
        .get("min_yaw")
        .and_then(|f| f.parse::<f32>().ok())
        .map(|d| d.to_radians())
        .unwrap_or(f32::MIN);
    let max_yaw = search_params
        .get("max_yaw")
        .and_then(|f| f.parse::<f32>().ok())
        .map(|d| d.to_radians())
        .unwrap_or(f32::MAX);

    let min_pitch = search_params
        .get("min_pitch")
        .and_then(|f| f.parse::<f32>().ok())
        .map(|d| d.to_radians())
        .unwrap_or(f32::MIN);
    let max_pitch = search_params
        .get("max_pitch")
        .and_then(|f| f.parse::<f32>().ok())
        .map(|d| d.to_radians())
        .unwrap_or(f32::MAX);

    let cam_settings = CameraSettings {
        focal,
        radius,
        radius_range: min_radius..max_radius,
        yaw_range: min_yaw..max_yaw,
        pitch_range: min_pitch..max_pitch,
    };
    cam_settings
}

#[derive(Debug, PartialEq)]
pub(crate) enum CameraRotateMode {
    Orbit,
    PanTilt,
}

pub(crate) struct CameraController {
    pub position: Vec3A,
    pub yaw: f32,
    pub pitch: f32,

    pub focus: Vec3A,
    pub dirty: bool,

    pub radius: f32,

    pub rotate_mode: CameraRotateMode,

    pub movement_speed: f32,
    pub rotation_speed: f32,
    pub zoom_speed: f32,

    dolly_momentum: Vec3A,
    rotate_momentum: Vec2,

    radius_range: Range<f32>,
    yaw_range: Range<f32>,
    pitch_range: Range<f32>,

    fine_tuning_scalar: f32,

    base_focus: Vec3A,
    base_position: Vec3A,
    base_yaw: f32,
    base_pitch: f32,
    base_distance: f32,
}

impl CameraController {
    pub fn new(
        radius: f32,
        radius_range: Range<f32>,
        yaw_range: Range<f32>,
        pitch_range: Range<f32>,
    ) -> Self {
        let position = -Vec3A::Z * radius;
        let base_position = position;
        Self {
            yaw: 0.0,
            pitch: 0.0,
            radius,

            position,
            focus: Vec3A::ZERO,

            radius_range,
            yaw_range,
            pitch_range,

            dirty: false,

            rotate_mode: CameraRotateMode::PanTilt,
            movement_speed: 0.2,
            rotation_speed: 0.005,
            zoom_speed: 0.002,

            dolly_momentum: Vec3A::ZERO,
            rotate_momentum: Vec2::ZERO,

            fine_tuning_scalar: 0.2,

            base_position,
            base_yaw: 0.0,
            base_pitch: 0.0,
            base_focus: Vec3A::ZERO,
            base_distance: radius,
        }
    }

    fn clamp_smooth(val: f32, range: Range<f32>) -> f32 {
        let mut val = val;
        if val < range.start {
            val = val * 0.5 + range.start * 0.5;
        }

        if val > range.end {
            val = val * 0.5 + range.end * 0.5;
        }
        val
    }

    #[allow(dead_code)]
    fn clamp_rotation(quat: Quat, pitch_range: Range<f32>, yaw_range: Range<f32>) -> Quat {
        let (pitch, yaw, _) = quat.to_euler(EulerRot::YXZ);
        let clamped_pitch = Self::clamp_smooth(pitch, pitch_range);
        let clamped_yaw = Self::clamp_smooth(yaw, yaw_range);
        Quat::from_euler(EulerRot::YXZ, clamped_yaw, clamped_pitch, 0.0)
    }

    pub fn reset(&mut self) {
        self.position = self.base_position;
        self.yaw = self.base_yaw;
        self.pitch = self.base_pitch;
        self.radius = self.base_distance;
        self.focus = self.base_focus;
        self.dolly_momentum = Vec3A::ZERO;
        self.rotate_momentum = Vec2::ZERO;
        self.dirty = true;
    }

    pub fn camera_has_moved(&self) -> bool {
        self.position != self.base_position
            || self.yaw != self.base_yaw
            || self.pitch != self.base_pitch
            || self.radius != self.base_distance
            || self.focus != self.base_focus
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
        self.handle_rotate(rotate, delta_time);
    }

    fn get_rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    fn update_position(&mut self) {
        let rotation = self.get_rotation();
        self.position = self.focus + rotation * Vec3A::new(0.0, 0.0, -self.radius);
    }

    fn update_focus(&mut self) {
        let rotation = self.get_rotation();
        self.focus = self.position - rotation * Vec3A::new(0.0, 0.0, -self.radius);
    }

    fn zoom(&mut self, scroll: f32) {
        let mut radius = self.radius;
        radius -= scroll * radius * self.zoom_speed;
        radius = Self::clamp_smooth(radius, self.radius_range.clone());
        self.radius = radius;
    }

    pub fn dolly(&mut self, movement: Vec3A, delta_time: f32) {
        let rotation = self.get_rotation();

        let damping = 0.0005f32.powf(delta_time);
        self.dolly_momentum += movement * self.movement_speed;
        self.dolly_momentum *= damping;

        let pan_velocity = self.dolly_momentum * delta_time;
        let scaled_pan = pan_velocity;

        let right = rotation * Vec3A::X * -scaled_pan.x;
        let up = rotation * Vec3A::Y * -scaled_pan.y;
        let forward = rotation * Vec3A::Z * -scaled_pan.z;

        let translation = (right + up + forward) * self.radius;
        self.focus += translation;
        self.update_position();
    }

    pub fn handle_rotate(&mut self, rotate: Vec2, delta_time: f32) {
        let damping = 0.0005f32.powf(delta_time);
        self.rotate_momentum += rotate * self.rotation_speed;
        self.rotate_momentum *= damping;

        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0;
        let delta_y = rotate_velocity.y * std::f32::consts::PI;

        let mut yaw = self.yaw;
        let mut pitch = self.pitch;
        yaw = Self::clamp_smooth(yaw + delta_x, self.yaw_range.clone());
        pitch = Self::clamp_smooth(pitch - delta_y, self.pitch_range.clone());
        self.yaw = yaw;
        self.pitch = pitch;

        if self.rotate_mode == CameraRotateMode::Orbit {
            self.update_position();
        } else {
            self.update_focus();
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

        if dolly_x.abs() > 0.0 || dolly_y.abs() > 0.0 || dolly_z.abs() > 0.0 {
            self.dolly(
                Vec3A::new(dolly_x, dolly_y, dolly_z),
                delta_time.as_secs_f32(),
            );
        }
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
                    .map(|t| (t.zoom_delta - 1.0) * self.radius * 5.0)
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

        self.rotate_mode = if ui.input(|r| r.modifiers.command_only()) {
            CameraRotateMode::Orbit
        } else {
            CameraRotateMode::PanTilt
        };

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

    pub(crate) fn transform(&self) -> Affine3A {
        Affine3A::from_rotation_translation(
            Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch),
            self.position.into(),
        )
    }
}
