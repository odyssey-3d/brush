use brush_render::camera::Camera;
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
        }
    }

    pub fn rotate_dolly_and_zoom(
        &mut self,
        camera: &mut Camera,
        dolly: Vec3,
        rotate: Vec2,
        scroll: f32,
        delta_time: f32,
    ) {
        self.zoom(camera, scroll);
        self.dolly(camera, dolly, delta_time);
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

    pub fn dolly(&mut self, camera: &mut Camera, dolly: Vec3, delta_time: f32) {
        self.dolly_momentum += dolly * self.movement_speed;
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

    pub fn pan_and_tilt(
        &mut self,
        camera: &mut Camera,
        rotate: Vec2,
        delta_time: f32,
    ) {
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

    pub fn handle_rotate(
        &mut self,
        camera: &mut Camera,
        rotate: Vec2,
        delta_time: f32,
    ) {
        match self.rotate_mode {
            CameraRotateMode::Orbit => {
                self.orbit(camera, rotate, delta_time);
            }
            CameraRotateMode::PanTilt => {
                self.pan_and_tilt(camera, rotate, delta_time);
            }
        }
    }

    pub fn is_animating(&self) -> bool {
        self.dolly_momentum.length_squared() > 1e-2 || self.rotate_momentum.length_squared() > 1e-2
    }
}
