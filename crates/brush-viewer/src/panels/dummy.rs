use crate::{viewer::ViewerContext, ViewerPanel};

enum Quality {
    Low,
    Normal,
}

pub(crate) struct DummyPanel {}

impl DummyPanel {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl ViewerPanel for DummyPanel {
    fn title(&self) -> String {
        "".to_owned()
    }

    fn ui(&mut self, _ui: &mut egui::Ui, _context: &mut ViewerContext) {}
}
