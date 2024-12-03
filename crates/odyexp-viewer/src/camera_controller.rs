use core::f32;
use std::ops::Range;

use egui::Rect;
use gamepads::{Gamepad, Gamepads};
use glam::{Affine3A, Quat, Vec2, Vec3A};

pub(crate) struct CameraSettings {
    pub focal: f64,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,

    pub yaw_range: Range<f32>,
    pub pitch_range: Range<f32>,
    pub radius_range: Range<f32>,
}

#[derive(Debug, PartialEq)]
enum ControlMode {
    Normal,
    SlowDown,
    SpeedUp,
}

pub(crate) struct ControlSensitivity {
    pub movement: f32,
    pub rotation: f32,
    pub zoom: f32,

    pub max_movement: f32,

    pub slow_down_scalar: f32,
    pub speed_up_scalar: f32,

    pub gamepad_dolly_sensitivity: f32,
    pub gamepad_rotate_sensitivity: f32,

    pub key_dolly_sensitivity: f32,
    pub key_rotate_sensitivity: f32,
}

#[derive(Debug, PartialEq)]
pub(crate) enum CameraRotateMode {
    Orbit,
    PanTilt,
}

pub(crate) struct CameraController {
    pub position: Vec3A,
    pub rotation: Quat,

    pub focus: Vec3A,
    pub dirty: bool,

    pub radius: f32,

    pub rotate_mode: CameraRotateMode,

    pub control_sensitivity: ControlSensitivity,

    dolly_momentum: Vec3A,
    rotate_momentum: Vec2,

    control_mode: ControlMode,

    radius_range: Range<f32>,
    yaw_range: Range<f32>,
    pitch_range: Range<f32>,

    base_focus: Vec3A,
    base_position: Vec3A,
    base_rotation: Quat,
    base_distance: f32,
}

impl CameraController {
    pub fn new(
        radius: f32,
        pitch: f32,
        yaw: f32,
        radius_range: Range<f32>,
        yaw_range: Range<f32>,
        pitch_range: Range<f32>,
    ) -> Self {
        let rotation = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);
        let position = rotation * Vec3A::new(0.0, 0.0, -radius);
        Self {
            radius,

            position,
            rotation,
            focus: Vec3A::ZERO,

            radius_range,
            yaw_range,
            pitch_range,

            dirty: false,

            rotate_mode: CameraRotateMode::PanTilt,
            control_sensitivity: ControlSensitivity::new(2.0, 0.001, 0.002, 0.5, 0.2, 5.0),

            control_mode: ControlMode::Normal,

            dolly_momentum: Vec3A::ZERO,
            rotate_momentum: Vec2::ZERO,

            base_position: position,
            base_rotation: rotation,
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

    pub fn reset(&mut self) {
        self.position = self.base_position;
        self.rotation = self.base_rotation;
        self.radius = self.base_distance;
        self.focus = self.base_focus;
        self.dolly_momentum = Vec3A::ZERO;
        self.rotate_momentum = Vec2::ZERO;
        self.dirty = true;
    }

    pub fn camera_has_moved(&self) -> bool {
        self.position != self.base_position
            || self.rotation != self.base_rotation
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
        self.handle_movement(movement, delta_time);
        self.handle_rotate(rotate, delta_time);
    }

    fn update_position(&mut self) {
        self.position = self.focus + self.rotation * Vec3A::new(0.0, 0.0, -self.radius);
    }

    fn update_focus(&mut self) {
        self.focus = self.position - self.rotation * Vec3A::new(0.0, 0.0, -self.radius);
    }

    fn zoom(&mut self, scroll: f32) {
        let mut radius = self.radius;
        radius -= scroll * radius * self.control_sensitivity.zoom;
        radius = Self::clamp_smooth(radius, self.radius_range.clone());
        self.radius = radius;
    }

    fn handle_movement(&mut self, movement: Vec3A, delta_time: f32) {
        let damping = 0.0005f32.powf(delta_time);
        self.dolly_momentum += movement * self.control_sensitivity.movement;
        self.dolly_momentum *= damping;

        let max_movement = if self.control_mode == ControlMode::SlowDown {
            self.control_sensitivity.slow_down_scalar * self.control_sensitivity.max_movement
        } else if self.control_mode == ControlMode::SpeedUp {
            self.control_sensitivity.speed_up_scalar * self.control_sensitivity.max_movement
        } else {
            self.control_sensitivity.max_movement
        };
        self.dolly_momentum = self.dolly_momentum.clamp_length_max(max_movement);

        let pan_velocity = self.dolly_momentum * delta_time;
        let scaled_pan = pan_velocity;

        let right = self.rotation * Vec3A::X * -scaled_pan.x;
        let up = self.rotation * Vec3A::Y * -scaled_pan.y;
        let forward = self.rotation * Vec3A::Z * -scaled_pan.z;

        let translation = (right + up + forward) * self.radius;
        self.focus += translation;
        self.update_position();
        if self.dolly_momentum.length_squared() < 1e-6 {
            self.dolly_momentum = Vec3A::ZERO;
        }
    }

    fn handle_rotate(&mut self, rotate: Vec2, delta_time: f32) {
        let damping = 0.0005f32.powf(delta_time);
        self.rotate_momentum += rotate * self.control_sensitivity.rotation;
        self.rotate_momentum *= damping;

        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0;
        let delta_y = rotate_velocity.y * std::f32::consts::PI;

        let (yaw, pitch, roll) = self.rotation.to_euler(glam::EulerRot::YXZ);
        let yaw = Self::clamp_smooth(yaw + delta_x, self.yaw_range.clone());
        let pitch = Self::clamp_smooth(pitch - delta_y, self.pitch_range.clone());

        self.rotation =
            Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch) * Quat::from_rotation_z(roll);

        if self.rotate_mode == CameraRotateMode::Orbit {
            self.update_position();
        } else {
            self.update_focus();
        }

        if self.rotate_momentum.length_squared() < 1e-6 {
            self.rotate_momentum = Vec2::ZERO;
        }
    }

    fn check_for_dolly_keys(&mut self, ui: &mut egui::Ui) -> Vec3A {
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

        Vec3A::new(dolly_x, dolly_y, dolly_z * 2.0) * self.control_sensitivity.key_dolly_sensitivity
    }

    fn check_for_dolly_gamepad(&mut self, gamepad: &Gamepad) -> Vec3A {
        let mut dolly_x = 0.0;
        let mut dolly_y = 0.0;
        let mut dolly_z = 0.0;

        let left_stick = gamepad.left_stick();

        dolly_x -= left_stick.0;
        dolly_z -= left_stick.1;

        if gamepad.is_currently_pressed(gamepads::Button::DPadUp) {
            dolly_y += 1.0;
        }
        if gamepad.is_currently_pressed(gamepads::Button::DPadDown) {
            dolly_y -= 1.0;
        }
        if gamepad.is_currently_pressed(gamepads::Button::DPadLeft) {
            dolly_x += 1.0;
        }
        if gamepad.is_currently_pressed(gamepads::Button::DPadRight) {
            dolly_x -= 1.0;
        }

        Vec3A::new(dolly_x, dolly_y, dolly_z * 2.0)
            * self.control_sensitivity.gamepad_dolly_sensitivity
    }

    fn check_for_rotate_keys(&mut self, ui: &mut egui::Ui) -> Vec2 {
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

        Vec2::new(rotate_x, rotate_y) * self.control_sensitivity.key_rotate_sensitivity
    }

    fn check_for_rotate_gamepad(&mut self, gamepad: &Gamepad) -> Vec2 {
        let right_stick = gamepad.right_stick();
        Vec2::new(right_stick.0, -right_stick.1)
            * self.control_sensitivity.gamepad_rotate_sensitivity
    }

    pub fn handle_user_input(
        &mut self,
        ui: &mut egui::Ui,
        size: glam::UVec2,
        delta_time: std::time::Duration,
        gamepads: &Gamepads,
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

        let mut orbit = false;
        if ui.input(|r| r.modifiers.command_only()) {
            orbit = true;
        } else {
            for gamepad in gamepads.all() {
                if gamepad.is_currently_pressed(gamepads::Button::FrontLeftUpper) {
                    orbit = true;
                    break;
                }
            }
        }

        self.rotate_mode = if orbit {
            CameraRotateMode::Orbit
        } else {
            CameraRotateMode::PanTilt
        };

        let (movement, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
            (Vec2::ZERO, mouse_delta)
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            (mouse_delta, Vec2::ZERO)
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };

        let mut movement = Vec3A::new(movement.x, movement.y, 0.0);
        movement += self.check_for_dolly_keys(ui);

        let mut rotate = rotate;
        rotate += self.check_for_rotate_keys(ui);

        for gamepad in gamepads.all() {
            movement += self.check_for_dolly_gamepad(&gamepad);
            rotate += self.check_for_rotate_gamepad(&gamepad);
        }

        self.control_mode = ControlMode::Normal;
        if ui.input(|r| r.modifiers.shift_only()) {
            self.control_mode = ControlMode::SlowDown;
        } else if ui.input(|r| r.modifiers.alt) {
            self.control_mode = ControlMode::SpeedUp;
        } else {
            for gamepad in gamepads.all() {
                if gamepad.is_currently_pressed(gamepads::Button::FrontLeftLower) {
                    self.control_mode = ControlMode::SlowDown;
                    break;
                } else if gamepad.is_currently_pressed(gamepads::Button::FrontRightLower) {
                    self.control_mode = ControlMode::SpeedUp;
                    break;
                }
            }
        }

        self.rotate_dolly_and_zoom(movement, rotate, scrolled, delta_time.as_secs_f32());

        self.dirty = scrolled.abs() > 0.0
            || movement.length_squared() > 0.0
            || rotate.length_squared() > 0.0
            || self.dolly_momentum.length_squared() > 1e-6
            || self.rotate_momentum.length_squared() > 1e-6
            || self.dirty;

        rect
    }

    pub(crate) fn transform(&self) -> Affine3A {
        Affine3A::from_rotation_translation(self.rotation, self.position.into())
    }
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
        .unwrap_or(2.0);

    let yaw = search_params
        .get("yaw_deg")
        .and_then(|f| f.parse::<f32>().ok())
        .map(|d| d.to_radians())
        .unwrap_or(0.0);
    let yaw = yaw / 180.0 * std::f32::consts::PI;

    let pitch = search_params
        .get("pitch_deg")
        .and_then(|f| f.parse::<f32>().ok())
        .map(|d| d.to_radians())
        .unwrap_or(-10.0);
    let pitch = pitch / 180.0 * std::f32::consts::PI;

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
        yaw,
        pitch,
        radius_range: min_radius..max_radius,
        yaw_range: min_yaw..max_yaw,
        pitch_range: min_pitch..max_pitch,
    };
    cam_settings
}

impl ControlSensitivity {
    pub fn new(
        movement: f32,
        rotation: f32,
        zoom: f32,
        max_movement: f32,
        slow_down_scalar: f32,
        speed_up_scalar: f32,
    ) -> Self {
        Self {
            movement,
            rotation,
            zoom,
            max_movement,
            slow_down_scalar,
            speed_up_scalar,
            gamepad_dolly_sensitivity: 0.1,
            gamepad_rotate_sensitivity: 5.0,
            key_dolly_sensitivity: 0.1,
            key_rotate_sensitivity: 5.0,
        }
    }
}
