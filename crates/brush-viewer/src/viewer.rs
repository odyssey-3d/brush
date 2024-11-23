use std::{
    collections::{BTreeSet, HashMap},
    pin::Pin,
    sync::Arc,
};

use async_fn_stream::try_fn_stream;

use brush_dataset::{self, splat_import, Dataset, LoadDatasetArgs, LoadInitArgs};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_train::train::TrainStepStats;
use brush_train::{eval::EvalStats, train::TrainConfig};
use burn::backend::Autodiff;
use burn_wgpu::{Wgpu, WgpuDevice};
use eframe::egui;
use egui_tiles::{Container, Tile, TileId, Tiles};
use glam::{Affine3A, Quat, Vec3, Vec3A};
use tokio_with_wasm::alias as tokio;

use ::tokio::io::AsyncReadExt;
use ::tokio::sync::mpsc::error::TrySendError;
use ::tokio::sync::mpsc::{Receiver, Sender};
use ::tokio::{io::AsyncRead, io::BufReader, sync::mpsc::channel};
use tokio::task;

use tokio_stream::{Stream, StreamExt};
use web_time::Instant;

type Backend = Wgpu;

use crate::{
    camera_controller::CameraController,
    panels::{build_panel, panel_title, PanelTypes, ScenePanel, StatsPanel, TracingPanel},
    train_loop::{self, TrainMessage},
    PaneType, ViewerTree,
};

struct TrainStats {
    loss: f32,
    train_image_index: usize,
}

#[derive(Clone)]
pub(crate) enum ViewerMessage {
    NewSource,
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
    /// Nb: Animated splats will have the 'frame' number set.
    ViewSplats {
        up_axis: Vec3,
        splats: Box<Splats<Backend>>,
        frame: usize,
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
        splats: Box<Splats<Backend>>,
        stats: Box<TrainStepStats<Autodiff<Backend>>>,
        iter: u32,
        timestamp: Instant,
    },
    /// Eval was run sucesfully with these results.
    EvalResult {
        iter: u32,
        eval: EvalStats<Backend>,
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

    pub model_transform: glam::Affine3A,

    device: WgpuDevice,
    ctx: egui::Context,

    sender: Option<Sender<TrainMessage>>,
    receiver: Option<Receiver<ViewerMessage>>,
}

fn process_loading_loop(
    source: DataSource,
    device: WgpuDevice,
) -> Pin<Box<impl Stream<Item = anyhow::Result<ViewerMessage>>>> {
    let stream = try_fn_stream(|emitter| async move {
        let _ = emitter.emit(ViewerMessage::NewSource).await;

        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let (data, filename) = source.read().await?;
        let mut data = BufReader::new(data);
        let mut peek = [0; 128];
        data.read_exact(&mut peek).await?;
        let data = std::io::Cursor::new(peek).chain(data);

        log::info!("{:?}", String::from_utf8(peek.to_vec()));
        if peek.starts_with("ply".as_bytes()) {
            log::info!("Attempting to load data as .ply data");

            let _ = emitter
                .emit(ViewerMessage::StartLoading {
                    training: false,
                    filename,
                })
                .await;

            let subsample = None; // Subsampling a trained ply doesn't really make sense.
            let splat_stream = splat_import::load_splat_from_ply(data, subsample, device.clone());

            let mut splat_stream = std::pin::pin!(splat_stream);

            while let Some(message) = splat_stream.next().await {
                let message = message?;
                emitter
                    .emit(ViewerMessage::ViewSplats {
                        up_axis: message.meta.up_axis,
                        splats: Box::new(message.splats),
                        frame: message.meta.current_frame,
                    })
                    .await;
            }

            emitter
                .emit(ViewerMessage::DoneLoading { training: false})
                .await;
        } else if peek.starts_with("<!DOCTYPE html>".as_bytes()) {
            anyhow::bail!("Failed to download data (are you trying to download from Google Drive? You might have to use the proxy.")
        } else {
            anyhow::bail!("only zip and ply files are supported.");
        }

        Ok(())
    });

    Box::pin(stream)
}

fn process_loop(
    source: DataSource,
    device: WgpuDevice,
    train_receiver: Receiver<TrainMessage>,
    load_data_args: LoadDatasetArgs,
    load_init_args: LoadInitArgs,
    train_config: TrainConfig,
) -> Pin<Box<impl Stream<Item = anyhow::Result<ViewerMessage>>>> {
    let stream = try_fn_stream(|emitter| async move {
        let _ = emitter.emit(ViewerMessage::NewSource).await;

        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let (data, filename) = source.read().await?;
        let mut data = BufReader::new(data);
        let mut peek = [0; 128];
        data.read_exact(&mut peek).await?;
        let data = std::io::Cursor::new(peek).chain(data);

        log::info!("{:?}", String::from_utf8(peek.to_vec()));

        if peek.starts_with("ply".as_bytes()) {
            log::info!("Attempting to load data as .ply data");

            let _ = emitter
                .emit(ViewerMessage::StartLoading {
                    training: true,
                    filename,
                })
                .await;

            let subsample = None; // Subsampling a trained ply doesn't really make sense.
            let splat_stream = splat_import::load_splat_from_ply(data, subsample, device.clone());

            let mut splat_stream = std::pin::pin!(splat_stream);

            while let Some(message) = splat_stream.next().await {
                let message = message?;
                emitter
                    .emit(ViewerMessage::ViewSplats {
                        up_axis: message.meta.up_axis,
                        splats: Box::new(message.splats),
                        frame: message.meta.current_frame,
                    })
                    .await;
            }

            emitter
                .emit(ViewerMessage::DoneLoading { training: true })
                .await;
        } else if peek.starts_with("PK".as_bytes()) {
            log::info!("Attempting to load data as .zip data");

            let _ = emitter
                .emit(ViewerMessage::StartLoading {
                    training: true,
                    filename: filename,
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
        } else if peek.starts_with("<!DOCTYPE html>".as_bytes()) {
            anyhow::bail!("Failed to download data (are you trying to download from Google Drive? You might have to use the proxy.")
        } else {
            anyhow::bail!("only zip and ply files are supported.");
        }

        Ok(())
    });

    Box::pin(stream)
}

#[derive(Debug)]
pub enum DataSource {
    PickFile,
    Url(String),
}
#[cfg(target_family = "wasm")]
type DataRead = Pin<Box<dyn AsyncRead>>;

#[cfg(not(target_family = "wasm"))]
type DataRead = Pin<Box<dyn AsyncRead + Send>>;

impl DataSource {
    async fn read(&self) -> anyhow::Result<(DataRead, String)> {
        match self {
            DataSource::PickFile => {
                let picked = rrfd::pick_file().await?;
                match picked {
                    rrfd::FileHandle::Rfd(file_handle) => {
                        let filename = file_handle.file_name();
                        let data = file_handle.read().await;
                        Ok((Box::pin(std::io::Cursor::new(data)), filename))
                    }
                }
            }
            DataSource::Url(url) => {
                let mut url = url.to_owned();
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    url = format!("https://{}", url);
                }
                let response = reqwest::get(url.clone()).await?.bytes_stream();
                let mapped = response
                    .map(|e| e.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
                Ok((Box::pin(tokio_util::io::StreamReader::new(mapped)), url))
            }
        }
    }
}

impl ViewerContext {
    fn new(device: WgpuDevice, ctx: egui::Context, up_axis: Vec3) -> Self {
        let rotation = Quat::from_rotation_arc(Vec3::Y, up_axis);

        let model_transform =
            glam::Affine3A::from_rotation_translation(rotation, Vec3::ZERO).inverse();
        let controls = CameraController::new(glam::Affine3A::from_translation(-Vec3::Z * 10.0));

        // Camera position will be controller by controls.
        let camera = Camera::new(Vec3::ZERO, Quat::IDENTITY, 0.35, 0.35, glam::vec2(0.5, 0.5));

        Self {
            camera,
            controls,
            model_transform,
            device,
            ctx,
            dataset: Dataset::empty(),
            receiver: None,
            sender: None,
            open_panels: BTreeSet::from([
                panel_title(&PanelTypes::ViewOptions).to_owned(),
                panel_title(&PanelTypes::Stats).to_owned(),
            ]),
            filename: None,
        }
    }

    pub fn set_up_axis(&mut self, up_axis: Vec3) {
        let rotation = Quat::from_rotation_arc(Vec3::Y, up_axis);
        let model_transform =
            glam::Affine3A::from_rotation_translation(rotation, Vec3::ZERO).inverse();
        self.model_transform = model_transform;
    }

    pub fn focus_view(&mut self, cam: &Camera) {
        // set the controls transform.
        let cam_transform = Affine3A::from_rotation_translation(cam.rotation, cam.position);
        self.controls.transform = self.model_transform.inverse() * cam_transform;
        // Copy the camera, mostly to copy over the intrinsics and such.
        self.controls.focus = self.controls.transform.translation
            + self.controls.transform.matrix3
                * Vec3A::Z
                * self.dataset.train.bounds(0.0, 0.0).extent.length()
                * 0.5;
        self.controls.dirty = true;
        self.camera = cam.clone();
    }

    pub(crate) fn start_ply_load(&mut self, source: DataSource) {
        let device = self.device.clone();
        log::info!("Start data load");

        // Create a small channel. We don't want 10 updated splats to be stuck in the queue eating up memory!
        // Bigger channels could mean the train loop spends less time waiting for the UI though.
        let (sender, receiver) = channel(1);

        self.receiver = Some(receiver);
        self.sender = None;

        self.dataset = Dataset::empty();
        let ctx = self.ctx.clone();

        let fut = async move {
            // Map errors to a viewer message containing thee error.
            let mut stream = process_loading_loop(source, device).map(|m| match m {
                Ok(m) => m,
                Err(e) => {
                    println!("Err: {:?}", e);
                    ViewerMessage::Error(Arc::new(e))
                }
            });

            // Loop until there are no more messages, processing is done.
            while let Some(m) = stream.next().await {
                ctx.request_repaint();

                // Give back to the runtime for a second.
                // This only really matters in the browser.
                tokio::task::yield_now().await;

                // If channel is closed, bail.
                if sender.send(m).await.is_err() {
                    break;
                }
            }
        };

        task::spawn(fut);
    }

    pub(crate) fn start_data_load(
        &mut self,
        source: DataSource,
        load_data_args: LoadDatasetArgs,
        load_init_args: LoadInitArgs,
        train_config: TrainConfig,
    ) {
        let device = self.device.clone();
        log::info!("Start data load {source:?}");

        // create a channel for the train loop.
        let (train_sender, train_receiver) = channel(32);

        // Create a small channel. We don't want 10 updated splats to be stuck in the queue eating up memory!
        // Bigger channels could mean the train loop spends less time waiting for the UI though.
        let (sender, receiver) = channel(1);

        self.receiver = Some(receiver);
        self.sender = Some(train_sender);

        self.dataset = Dataset::empty();
        let ctx = self.ctx.clone();

        let fut = async move {
            // Map errors to a viewer message containing thee error.
            let mut stream = process_loop(
                source,
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

                // Give back to the runtime for a second.
                // This only really matters in the browser.
                tokio::task::yield_now().await;

                // If channel is closed, bail.
                if sender.send(m).await.is_err() {
                    break;
                }
            }
        };

        task::spawn(fut);
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
        // For now just assume we're running on the default
        let state = cc.wgpu_render_state.as_ref().unwrap();
        let device = brush_ui::create_wgpu_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

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

        let mut start_url = None;
        if cfg!(target_family = "wasm") {
            if let Some(window) = web_sys::window() {
                if let Ok(search) = window.location().search() {
                    if let Ok(search_params) = web_sys::UrlSearchParams::new_with_str(&search) {
                        let url = search_params.get("url");
                        start_url = url;
                    }
                }
            }
        }

        let mut tiles: Tiles<PaneType> = egui_tiles::Tiles::default();
        let up_axis = Vec3::Y;
        let context = ViewerContext::new(device.clone(), cc.egui_ctx.clone(), up_axis);

        let scene_pane = ScenePanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
        );

        #[allow(unused_mut)]
        let dummy = tiles.insert_pane(build_panel(&PanelTypes::Dummy, device.clone()));
        let view = tiles.insert_pane(build_panel(&PanelTypes::ViewOptions, device.clone()));
        let mut sides = vec![
            view,
            tiles.insert_pane(Box::new(StatsPanel::new(
                device.clone(),
                state.adapter.clone(),
            ))),
            dummy,
        ];

        if cfg!(feature = "tracing") {
            sides.push(tiles.insert_pane(Box::new(TracingPanel::default())));
        }

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

        let mut tree_ctx = ViewerTree { context };

        if let Some(start_url) = start_url {
            tree_ctx.context.start_data_load(
                DataSource::Url(start_url.to_owned()),
                LoadDatasetArgs::default(),
                LoadInitArgs::default(),
                TrainConfig::default(),
            );
        }

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
        if let Some(rec) = self.tree_ctx.context.receiver.as_mut() {
            let mut messages = vec![];

            while let Ok(message) = rec.try_recv() {
                messages.push(message);
            }

            for message in messages {
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
                            pane.on_message(&message, &mut self.tree_ctx.context);
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
                                self.tree_ctx.context.start_ply_load(DataSource::PickFile);
                            }
                        } else {
                            ui.heading(self.tree_ctx.context.filename.as_ref().unwrap());
                            if ui.button("...").clicked() {
                                self.tree_ctx.context.start_ply_load(DataSource::PickFile);
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
