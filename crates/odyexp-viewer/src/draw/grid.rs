use egui::{Color32, Painter, Rect, Stroke};
use glam::{Mat4, Vec3};

use super::world_to_screen;

pub(crate) struct Grid {
    size: u32,
    cell_size: f32,
    color: Color32,
    highlight_color: Color32,
}

impl Grid {
    pub fn new(size: u32, cell_size: f32) -> Self {
        Self {
            size,
            cell_size,
            color: Color32::DARK_GRAY,
            highlight_color: Color32::from_gray(117),
        }
    }
    pub fn with_color(self, color: Color32) -> Self {
        Self {
            size: self.size,
            cell_size: self.cell_size,
            color,
            highlight_color: self.highlight_color,
        }
    }

    pub fn draw(&self, painter: &Painter, viewport: Rect, mvp: Mat4) {
        let total_x = self.size as f32 * self.cell_size;
        let offset_x = total_x / 2.0;

        let light_stroke = Stroke::new(1.0, self.color);
        for i in 0..=self.size {
            let x = i as f32 * self.cell_size - offset_x;

            let start = world_to_screen(viewport, mvp, Vec3::new(x, 0.0, -offset_x));
            let end = world_to_screen(viewport, mvp, Vec3::new(x, 0.0, total_x - offset_x));
            if start.is_some() && end.is_some() {
                painter.line_segment([start.unwrap(), end.unwrap()], light_stroke);
            }

            let start = world_to_screen(viewport, mvp, Vec3::new(-offset_x, 0.0, x));
            let end = world_to_screen(viewport, mvp, Vec3::new(total_x - offset_x, 0.0, x));
            if start.is_some() && end.is_some() {
                painter.line_segment([start.unwrap(), end.unwrap()], light_stroke);
            }
        }

        //draw crosshair
        let heavy_stroke = Stroke::new(2.0, self.highlight_color);
        let seg_length = 0.2 * self.cell_size;

        let centre = Vec3::ZERO;
        let world_centre = world_to_screen(viewport, mvp, centre);
        if let Some(world_centre) = world_centre {
            if let Some(world_x) = world_to_screen(viewport, mvp, Vec3::new(seg_length, 0.0, 0.0)) {
                painter.line_segment([world_centre, world_x], heavy_stroke);
            }
            if let Some(world_neg_x) =
                world_to_screen(viewport, mvp, Vec3::new(-seg_length, 0.0, 0.0))
            {
                painter.line_segment([world_centre, world_neg_x], heavy_stroke);
            }
            if let Some(world_pos_z) =
                world_to_screen(viewport, mvp, Vec3::new(0.0, 0.0, seg_length))
            {
                painter.line_segment([world_centre, world_pos_z], heavy_stroke);
            }
            if let Some(world_neg_z) =
                world_to_screen(viewport, mvp, Vec3::new(0.0, 0.0, -seg_length))
            {
                painter.line_segment([world_centre, world_neg_z], heavy_stroke);
            }
        }

        let signs = vec![[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]];

        for sign in signs {
            let corner = Vec3::new(self.cell_size * sign[0], 0.0, self.cell_size * sign[1]);
            let world_corner = world_to_screen(viewport, mvp, corner);
            if let Some(world_corner) = world_corner {
                if let Some(corner_seg_1) = world_to_screen(
                    viewport,
                    mvp,
                    corner + Vec3::new(0.0, 0.0, -seg_length * sign[1]),
                ) {
                    painter.line_segment([world_corner, corner_seg_1], heavy_stroke);
                }
                if let Some(corner_seg_2) = world_to_screen(
                    viewport,
                    mvp,
                    corner + Vec3::new(-seg_length * sign[0], 0.0, 0.0),
                ) {
                    painter.line_segment([world_corner, corner_seg_2], heavy_stroke);
                }
            }
        }
    }
}
