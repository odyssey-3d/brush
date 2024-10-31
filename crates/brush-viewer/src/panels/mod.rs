mod datasets;
mod training_options;

mod presets;
mod scene;
mod stats;

mod dummy;
mod viewer_options;

pub(crate) use datasets::*;
pub(crate) use dummy::*;
pub(crate) use training_options::*;
pub(crate) use presets::*;
pub(crate) use scene::*;
pub(crate) use stats::*;
pub(crate) use viewer_options::*;

#[cfg(not(target_family = "wasm"))]
mod rerun;

#[cfg(not(target_family = "wasm"))]
pub(crate) use rerun::*;

#[cfg(feature = "tracing")]
mod tracing_debug;

#[cfg(feature = "tracing")]
pub(crate) use tracing_debug::*;

#[derive(Eq, Hash, PartialEq, Copy, Clone, Debug)]
pub enum PanelTypes {
    ViewOptions = 0,
    TrainingOptions = 1,
    Presets = 2,
    Stats = 3,
    Rerun = 4,
    Datasets = 5,
    Dummy = 6,
}

pub fn panel_title(panel: &PanelTypes) -> &'static str {
    match panel {
        PanelTypes::ViewOptions => "View Options",
        PanelTypes::TrainingOptions => "Training Options",
        PanelTypes::Presets => "Presets",
        PanelTypes::Stats => "Stats",
        PanelTypes::Rerun => "Rerun",
        PanelTypes::Datasets => "Datasets",
        PanelTypes::Dummy => "",
    }
}

pub fn build_panel(panel_type: &PanelTypes, device: burn_wgpu::WgpuDevice) -> crate::PaneType {
    match panel_type {
        PanelTypes::ViewOptions => Box::new(ViewerOptionsPanel::new()),
        PanelTypes::TrainingOptions => Box::new(TrainingOptionsPanel::new()),
        PanelTypes::Presets => Box::new(PresetsPanel::new()),
        PanelTypes::Stats => Box::new(StatsPanel::new(device)),
        PanelTypes::Rerun => Box::new(RerunPanel::new(device)),
        PanelTypes::Datasets => Box::new(DatasetPanel::new()),
        PanelTypes::Dummy => Box::new(DummyPanel::new()),
    }
}
