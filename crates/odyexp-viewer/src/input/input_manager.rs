use egui::{Rect, Ui};
use gamepads::Gamepads;
use glam::UVec2;
use std::time::Duration;

use crate::app_context::AppContext;

pub(crate) struct InputManager {
    gamepads: Gamepads,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            gamepads: Gamepads::new(),
        }
    }

    pub fn update(
        &mut self,
        context: &mut AppContext,
        ui: &mut Ui,
        size: UVec2,
        delta_time: Duration,
    ) -> Rect {
        self.gamepads.poll();
        let rect = context
            .controls
            .handle_user_input(ui, size, delta_time, &self.gamepads);


        rect
    }
}
