use egui::{Pos2, Rect};
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

#[allow(dead_code)]
/// Calculates 2d screen coordinates from 3d world coordinates
/// mvp : view projection matrix
pub(crate) fn world_to_screen(viewport: Rect, mvp: Mat4, pos: Vec3) -> Option<Pos2> {
    let mut pos = mvp * Vec4::from((pos, 1.0));

    if pos.w < 1e-10 {
        return None;
    }

    pos /= pos.w;

    let center = viewport.center();

    Some(Pos2::new(
        (center.x + pos.x * viewport.width() / 2.0) as f32,
        (center.y + pos.y * viewport.height() / 2.0) as f32,
    ))
}

#[allow(dead_code)]
/// Calculates 3d world coordinates from 2d screen coordinates
/// mat : inverse of projection matrix
pub(crate) fn screen_to_world(viewport: Rect, mat: Mat4, pos: Pos2, z: f32) -> Vec3 {
    let x = (((pos.x - viewport.min.x) / viewport.width()) * 2.0 - 1.0) as f32;
    let y = (((pos.y - viewport.min.y) / viewport.height()) * 2.0 - 1.0) as f32;

    let mut world_pos = mat * Vec4::new(x, -y, z, 1.0);

    // w is zero when far plane is set to infinity
    if world_pos.w.abs() < 1e-7 {
        world_pos.w = 1e-7;
    }

    world_pos /= world_pos.w;

    world_pos.xyz()
}
