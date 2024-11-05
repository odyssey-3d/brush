use super::{panel_title, PanelTypes};
use crate::{
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};
use burn_jit::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use std::{collections::VecDeque, sync::Arc};
use web_time::Instant;
use wgpu::AdapterInfo;

pub(crate) struct StatsPanel {
    device: WgpuDevice,

    last_train_step: (Instant, u32),
    train_iter_per_s: f32,
    train_iter_history: VecDeque<f32>,
    last_eval_psnr: Option<f32>,

    training_started: bool,
    is_loading: bool,
    num_splats: usize,

    adapter_info: AdapterInfo,
}

impl StatsPanel {
    pub(crate) fn new(device: WgpuDevice, adapter: Arc<wgpu::Adapter>) -> Self {
        let adapter_info = adapter.get_info();
        Self {
            device,
            last_train_step: (Instant::now(), 0),
            train_iter_per_s: 0.0,
            train_iter_history: VecDeque::with_capacity(5),
            last_eval_psnr: None,
            training_started: false,
            is_loading: false,
            num_splats: 0,
            adapter_info,
        }
    }
}

fn bytes_format(bytes: u64) -> String {
    let unit = 1000;

    if bytes < unit {
        format!("{} B", bytes)
    } else {
        let size = bytes as f64;
        let exp = match size.log(1000.0).floor() as usize {
            0 => 1,
            e => e,
        };
        let unit_prefix = "KMGTPEZY".as_bytes();
        format!(
            "{:.2} {}B",
            (size / unit.pow(exp as u32) as f64),
            unit_prefix[exp - 1] as char,
        )
    }
}

impl ViewerPanel for StatsPanel {
    fn title(&self) -> String {
        panel_title(&PanelTypes::Stats).to_owned()
    }

    fn on_message(&mut self, message: crate::viewer::ViewerMessage, _: &mut ViewerContext) {
        match message {
            ViewerMessage::StartLoading {
                training,
                filename: _,
            } => {
                self.last_train_step = (Instant::now(), 0);
                self.train_iter_per_s = 0.0;
                self.train_iter_history.clear();
                self.num_splats = 0;
                self.last_eval_psnr = None;
                self.training_started = training;
                self.is_loading = true;
            }
            ViewerMessage::TrainStep {
                stats: _,
                iter,
                timestamp,
            } => {
                let current_iter_per_s = (iter - self.last_train_step.1) as f32
                    / (timestamp - self.last_train_step.0).as_secs_f32();

                self.train_iter_history.push_back(current_iter_per_s);
                if self.train_iter_history.len() > 25 {
                    self.train_iter_history.pop_front();
                }

                self.train_iter_per_s = self.train_iter_history.iter().sum::<f32>()
                    / self.train_iter_history.len() as f32;

                self.last_train_step = (timestamp, iter);
            }
            ViewerMessage::Splats { iter: _, splats } => {
                self.num_splats = splats.num_splats();
            }
            ViewerMessage::EvalResult { iter: _, eval } => {
                let avg_psnr =
                    eval.samples.iter().map(|s| s.psnr).sum::<f32>() / (eval.samples.len() as f32);
                self.last_eval_psnr = Some(avg_psnr);
            }
            ViewerMessage::DoneLoading { .. } => {
                self.is_loading = false;
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut ViewerContext) {
        egui::Grid::new("stats_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Splats");

                let loading = match self.is_loading {
                    true => Some(("Loading...").to_string()),
                    false => None,
                };
                ui.label(format!(
                    "{} {}",
                    self.num_splats,
                    loading.unwrap_or_default()
                ));
                ui.end_row();

                ui.separator();
                ui.end_row();

                if self.training_started {
                    ui.label("Train step");
                    ui.label(format!("{}", self.last_train_step.1));
                    ui.end_row();

                    ui.label("Steps/s");
                    ui.label(format!("{:.1}", self.train_iter_per_s));
                    ui.end_row();

                    ui.label("Last eval PSNR");
                    ui.label(if let Some(psnr) = self.last_eval_psnr {
                        format!("{:.}", psnr)
                    } else {
                        "--".to_owned()
                    });
                    ui.end_row();
                    ui.separator();
                    ui.end_row();
                }

                let client = WgpuRuntime::client(&self.device);
                let memory = client.memory_usage();

                ui.label("GPU memory");
                ui.end_row();

                ui.label("Bytes in use");
                ui.label(bytes_format(memory.bytes_in_use));
                ui.end_row();

                ui.label("Bytes reserved");
                ui.label(bytes_format(memory.bytes_reserved));
                ui.end_row();

                ui.label("Active allocations");
                ui.label(format!("{}", memory.number_allocs));
                ui.end_row();
            });

        // On WASM, adapter info is mostly private, not worth showing.
        if !cfg!(target_family = "wasm") {
            egui::Grid::new("gpu_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("GPU");
                    ui.end_row();

                    ui.label("Name");
                    ui.label(&self.adapter_info.name);
                    ui.end_row();

                    ui.label("Type");
                    ui.label(format!("{:?}", self.adapter_info.device_type));
                    ui.end_row();

                    ui.label("Driver");
                    ui.label(format!(
                        "{}, {}",
                        self.adapter_info.driver, self.adapter_info.driver_info
                    ));
                    ui.end_row();
                });
        }
    }
}
