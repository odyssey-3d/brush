use std::{
    collections::{BTreeSet, HashMap},
    pin::Pin,
    sync::Arc,
};

use async_fn_stream::try_fn_stream;
use async_std::{
    channel::{Receiver, Sender, TrySendError},
    stream::{Stream, StreamExt},
};
use brush_dataset::{self, splat_import, Dataset, LoadDatasetArgs, LoadInitArgs};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::PrimaryBackend;
use brush_train::train::TrainStepStats;
use brush_train::{eval::EvalStats, train::TrainConfig};
use burn::backend::Autodiff;
use burn_wgpu::WgpuDevice;
use eframe::egui;

use egui_tiles::{Container, Tile, TileId, Tiles};
use glam::{Quat, Vec3};
use web_time::Instant;

use crate::{
    camera_controller::CameraController,
    panels::{build_panel, panel_title, PanelTypes, ScenePanel, StatsPanel},
    train_loop::{self, TrainMessage},
    PaneType, ViewerTree,
};

struct TrainStats {
    loss: f32,
    train_image_index: usize,
}

#[derive(Clone)]
pub(crate) enum ViewerMessage {
    PickFile,
    StartLoading {
        training: bool,
        filename: String,
    },
    /// Some process errored out, and want to display this error
    /// to the user.
    Error(Arc<anyhow::Error>),
    /// Loaded a splat from a ply file.
    ///
    /// Nb: This includes all the intermediately loaded splats.
    Splats {
        iter: u32,
        splats: Box<Splats<PrimaryBackend>>,
    },
    /// Loaded a bunch of viewpoints to train on.
    Dataset {
        data: Dataset,
    },
    /// Splat, or dataset and initial splat, are done loading.
    DoneLoading {
        training: bool,
    },
    /// Some number of training steps are done.
    TrainStep {
        stats: Box<TrainStepStats<Autodiff<PrimaryBackend>>>,
        iter: u32,
        timestamp: Instant,
    },
    /// Eval was run sucesfully with these results.
    EvalResult {
        iter: u32,
        eval: EvalStats<PrimaryBackend>,
    },
    //show training panel
    ShowTrainingPanel {
        show: bool,
    },
}

pub struct Viewer {
    tree: egui_tiles::Tree<PaneType>,
    tree_ctx: ViewerTree,
    side_panel: TileId,
    panels: HashMap<PanelTypes, TileId>,
}

// TODO: Bit too much random shared state here.
pub(crate) struct ViewerContext {
    pub dataset: Dataset,
    pub camera: Camera,
    pub controls: CameraController,

    pub open_panels: BTreeSet<String>,
    pub filename: Option<String>,

    pub device: WgpuDevice,
    ctx: egui::Context,

    sender: Option<Sender<TrainMessage>>,
    receiver: Option<Receiver<ViewerMessage>>,
}

fn process_loading_loop(
    device: WgpuDevice,
) -> Pin<Box<impl Stream<Item = anyhow::Result<ViewerMessage>>>> {
    let stream = try_fn_stream(|emitter| async move {
        let _ = emitter.emit(ViewerMessage::PickFile).await;
        log::warn!("Start picking file");
        let picked = rrfd::pick_file().await?;
        let name = picked.file_name();

        log::warn!("Got file {name}");

        if name.contains(".ply") {
            let data = picked.read().await;

            let _ = emitter
                .emit(ViewerMessage::StartLoading {
                    training: false,
                    filename: name,
                })
                .await;
            let splat_stream =
                splat_import::load_splat_from_ply::<PrimaryBackend>(data, device.clone());
            let mut splat_stream = std::pin::pin!(splat_stream);
            while let Some(splats) = splat_stream.next().await {
                emitter
                    .emit(ViewerMessage::Splats {
                        iter: 0, // For viewing just use "training step 0", bit weird.
                        splats: Box::new(splats?),
                    })
                    .await;
            }
            emitter
                .emit(ViewerMessage::DoneLoading { training: true })
                .await;
        } else {
            anyhow::bail!("Only .ply files are supported for viewing.")
        }

        Ok(())
    });

    Box::pin(stream)
}

fn process_loop(
    device: WgpuDevice,
    train_receiver: Receiver<TrainMessage>,
    load_data_args: LoadDatasetArgs,
    load_init_args: LoadInitArgs,
    train_config: TrainConfig,
) -> Pin<Box<impl Stream<Item = anyhow::Result<ViewerMessage>>>> {
    let stream = try_fn_stream(|emitter| async move {
        let _ = emitter.emit(ViewerMessage::PickFile).await;
        log::info!("Start picking file");
        let picked = rrfd::pick_file().await?;
        let name = picked.file_name();

        log::info!("Picked file {name}");

        if name.contains(".ply") {
            let data = picked.read().await;

            let _ = emitter
                .emit(ViewerMessage::StartLoading {
                    training: false,
                    filename: name,
                })
                .await;
            let splat_stream =
                splat_import::load_splat_from_ply::<PrimaryBackend>(data, device.clone());
            let mut splat_stream = std::pin::pin!(splat_stream);
            while let Some(splats) = splat_stream.next().await {
                emitter
                    .emit(ViewerMessage::Splats {
                        iter: 0, // For viewing just use "training step 0", bit weird.
                        splats: Box::new(splats?),
                    })
                    .await;
            }
            emitter
                .emit(ViewerMessage::DoneLoading { training: true })
                .await;
        } else if name.contains(".zip") {
            let data = picked.read().await;
            let _ = emitter
                .emit(ViewerMessage::StartLoading {
                    training: true,
                    filename: name,
                })
                .await;

            let stream = train_loop::train_loop(
                data,
                device,
                train_receiver,
                load_data_args,
                load_init_args,
                train_config,
            );
            let mut stream = std::pin::pin!(stream);
            while let Some(message) = stream.next().await {
                emitter.emit(message?).await;
            }
        } else {
            anyhow::bail!("Only .ply and .zip files are supported.")
        }

        Ok(())
    });

    Box::pin(stream)
}

impl ViewerContext {
    fn new(device: WgpuDevice, ctx: egui::Context) -> Self {
        Self {
            camera: Camera::new(
                -Vec3::Z * 5.0,
                Quat::IDENTITY,
                0.866,
                0.5,
                glam::vec2(0.5, 0.5),
            ),
            controls: CameraController::new(),
            open_panels: BTreeSet::from([
                panel_title(&PanelTypes::ViewOptions).to_owned(),
                panel_title(&PanelTypes::Stats).to_owned(),
            ]),
            device,
            ctx,
            dataset: Dataset::empty(),
            receiver: None,
            sender: None,
            filename: None,
        }
    }

    pub fn focus_view(&mut self, cam: &Camera) {
        self.camera = cam.clone();
        self.controls.focus = self.camera.position
            + self.camera.rotation
                * glam::Vec3::Z
                * self.dataset.train.bounds(0.0, 0.0).extent.length()
                * 0.5;
    }

    pub(crate) fn start_ply_load(&mut self) {
        let device = self.device.clone();
        log::info!("Start data load");

        // Create a small channel. We don't want 10 updated splats to be stuck in the queue eating up memory!
        // Bigger channels could mean the train loop spends less time waiting for the UI though.
        let (sender, receiver) = async_std::channel::bounded(1);

        self.receiver = Some(receiver);
        self.sender = None;

        self.dataset = Dataset::empty();
        let ctx = self.ctx.clone();

        let fut = async move {
            // Map errors to a viewer message containing thee error.
            let mut stream = process_loading_loop(device).map(|m| match m {
                Ok(m) => m,
                Err(e) => ViewerMessage::Error(Arc::new(e)),
            });

            // Loop until there are no more messages, processing is done.
            while let Some(m) = stream.next().await {
                ctx.request_repaint();

                // If channel is closed, bail.
                if sender.send(m).await.is_err() {
                    break;
                }
            }
        };

        #[cfg(target_family = "wasm")]
        {
            let fut = crate::async_lib::with_timeout_yield(fut, web_time::Duration::from_millis(5));
            async_std::task::spawn_local(fut);
        }

        #[cfg(not(target_family = "wasm"))]
        {
            async_std::task::spawn(fut);
        }
    }

    pub(crate) fn start_data_load(
        &mut self,
        load_data_args: LoadDatasetArgs,
        load_init_args: LoadInitArgs,
        train_config: TrainConfig,
    ) {
        let device = self.device.clone();
        log::info!("Start data load");

        // create a channel for the train loop.
        let (train_sender, train_receiver) = async_std::channel::unbounded();

        // Create a small channel. We don't want 10 updated splats to be stuck in the queue eating up memory!
        // Bigger channels could mean the train loop spends less time waiting for the UI though.
        let (sender, receiver) = async_std::channel::bounded(1);

        self.receiver = Some(receiver);
        self.sender = Some(train_sender);

        self.dataset = Dataset::empty();
        let ctx = self.ctx.clone();

        let fut = async move {
            // Map errors to a viewer message containing thee error.
            let mut stream = process_loop(
                device,
                train_receiver,
                load_data_args,
                load_init_args,
                train_config,
            )
            .map(|m| match m {
                Ok(m) => m,
                Err(e) => ViewerMessage::Error(Arc::new(e)),
            });

            // Loop until there are no more messages, processing is done.
            while let Some(m) = stream.next().await {
                ctx.request_repaint();

                // If channel is closed, bail.
                if sender.send(m).await.is_err() {
                    break;
                }
            }
        };

        #[cfg(target_family = "wasm")]
        {
            let fut = crate::async_lib::with_timeout_yield(fut, web_time::Duration::from_millis(5));
            async_std::task::spawn_local(fut);
        }

        #[cfg(not(target_family = "wasm"))]
        {
            async_std::task::spawn(fut);
        }
    }

    pub fn send_train_message(&self, message: TrainMessage) {
        if let Some(sender) = self.sender.as_ref() {
            match sender.try_send(message) {
                Ok(_) => {}
                Err(TrySendError::Closed(_)) => {}
                Err(TrySendError::Full(_)) => {
                    unreachable!("Should use an unbounded channel for train messages.")
                }
            }
        }
    }
}

impl Viewer {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc.wgpu_render_state.as_ref().unwrap();

        // For now just assume we're running on the default
        let device = WgpuDevice::DefaultDevice;

        cfg_if::cfg_if! {
            if #[cfg(target_family = "wasm")] {
                use tracing_subscriber::layer::SubscriberExt;

                let subscriber = tracing_subscriber::registry().with(tracing_wasm::WASMLayer::new(Default::default()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            } else if #[cfg(feature = "tracy")] {
                use tracing_subscriber::layer::SubscriberExt;
                let subscriber = tracing_subscriber::registry()
                    .with(tracing_tracy::TracyLayer::default())
                    .with(sync_span::SyncLayer::new(device.clone()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            }
        }

        let mut tiles: Tiles<PaneType> = egui_tiles::Tiles::default();

        let context = ViewerContext::new(device.clone(), cc.egui_ctx.clone());

        let scene_pane = ScenePanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
        );

        #[allow(unused_mut)]
        let dummy = tiles.insert_pane(build_panel(&PanelTypes::Dummy, device.clone()));
        let view = tiles.insert_pane(build_panel(&PanelTypes::ViewOptions, device.clone()));
        let sides = vec![
            view,
            tiles.insert_pane(Box::new(StatsPanel::new(device, state.adapter.clone()))),
            dummy,
        ];

        let side_panel = tiles.insert_vertical_tile(sides);

        let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        let mut lin = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![side_panel, scene_pane_id],
        );
        lin.shares.set_share(side_panel, 0.2);

        let root_container = tiles.insert_container(lin);
        let mut tree = egui_tiles::Tree::new("viewer_tree", root_container, tiles);
        tree.set_visible(dummy, false);

        let tree_ctx = ViewerTree { context };

        Viewer {
            tree,
            tree_ctx,
            side_panel,
            panels: HashMap::new(),
        }
    }
}

fn check_for_panel_status(
    panel_type: &PanelTypes,
    context: &mut ViewerContext,
    panels: &mut HashMap<PanelTypes, TileId>,
    tree: &mut egui_tiles::Tree<PaneType>,
    side_panel: TileId,
) -> bool {
    let panel = panels.get(panel_type);
    if context.open_panels.contains(panel_title(panel_type)) {
        if panel.is_none() {
            let pane_id = tree
                .tiles
                .insert_pane(build_panel(panel_type, context.device.clone()));
            panels.insert(*panel_type, pane_id);
            if let Some(Tile::Container(Container::Linear(lin))) = tree.tiles.get_mut(side_panel) {
                lin.add_child(pane_id);
            }
            return true;
        }
    } else {
        if panel.is_some() {
            tree.remove_recursively(*panel.unwrap());
            panels.remove(panel_type);
            return true;
        }
    }
    false
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if let Some(rec) = self.tree_ctx.context.receiver.clone() {
            while let Ok(message) = rec.try_recv() {
                match message.clone() {
                    ViewerMessage::StartLoading {
                        training: _,
                        filename,
                    } => {
                        self.tree_ctx.context.filename = Some(filename);
                    }
                    ViewerMessage::Dataset { data: _ } => {
                        // Show the dataset panel if we've loaded one.
                        if self.panels.get(&PanelTypes::Datasets).is_none() {
                            let panel = build_panel(
                                &PanelTypes::Datasets,
                                self.tree_ctx.context.device.clone(),
                            );
                            self.tree_ctx.context.open_panels.insert(panel.title());
                            let pane_id = self.tree.tiles.insert_pane(panel);
                            if let Some(Tile::Container(Container::Linear(lin))) =
                                self.tree.tiles.get_mut(self.tree.root().unwrap())
                            {
                                lin.add_child(pane_id);
                            }
                            self.panels.insert(PanelTypes::Datasets, pane_id);
                        }
                    }
                    _ => {}
                }

                for (_, pane) in self.tree.tiles.iter_mut() {
                    match pane {
                        Tile::Pane(pane) => {
                            pane.on_message(message.clone(), &mut self.tree_ctx.context);
                        }
                        Tile::Container(_) => {}
                    }
                }

                ctx.request_repaint();
            }
        }

        let panels_to_check = vec![
            PanelTypes::TrainingOptions,
            PanelTypes::Presets,
            PanelTypes::Rerun,
        ];

        for panel in panels_to_check {
            if check_for_panel_status(
                &panel,
                &mut self.tree_ctx.context,
                &mut self.panels,
                &mut self.tree,
                self.side_panel,
            ) {
                ctx.request_repaint();
            }
        }
        egui::TopBottomPanel::top("wrap_app_top_bar")
            .frame(egui::Frame::none().inner_margin(4.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.scope(|ui| {
                        ui.style_mut().text_styles.insert(
                            egui::TextStyle::Button,
                            egui::FontId::new(24.0, eframe::epaint::FontFamily::Proportional),
                        );
                        ui.style_mut().text_styles.insert(
                            egui::TextStyle::Heading,
                            egui::FontId::new(24.0, eframe::epaint::FontFamily::Proportional),
                        );
                        if self.tree_ctx.context.filename.is_none() {
                            ui.heading("<No Scene loaded>");
                            if ui.button("...").clicked() {
                                self.tree_ctx.context.start_ply_load();
                            }
                        } else {
                            ui.heading(self.tree_ctx.context.filename.as_ref().unwrap());
                            if ui.button("...").clicked() {
                                self.tree_ctx.context.start_ply_load();
                            }
                        }
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // // Close when pressing escape (in a native viewer anyway).
            // if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
            //     ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            // }

            self.tree.ui(&mut self.tree_ctx, ui);
        });
    }
}
