use std::collections::HashMap;
use std::sync::Arc;

use brush_render::{camera::Camera, gaussian_splats::Splats};
use burn_wgpu::{Wgpu, WgpuDevice};

use glam::{Affine3A, Quat, Vec3};

use tokio_with_wasm::alias as tokio;

use ::tokio::sync::mpsc::{channel, Receiver, UnboundedReceiver, UnboundedSender};

use tokio::task;
use tokio_stream::StreamExt;

use crate::camera_controller::CameraController;
use crate::camera_controller::CameraSettings;
use crate::load::{process_loading_loop, DataSource};

use brush_dataset::splat_export;

type Backend = Wgpu;

pub enum UiControlMessage {
    LoadData(String),
    SaveSplats,
    ResetCamera,
}

#[derive(Clone, Debug)]
pub(crate) enum ViewerMessage {
    NewSource,
    StartLoading {
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
    /// Splats are done loading.
    DoneLoading,
}

#[derive(Clone, Debug)]
pub(crate) struct UILayout {
    pub top_panel_height: f32,
}

impl UILayout {
    pub fn default() -> Self {
        Self {
            top_panel_height: 65.0,
        }
    }
}

// TODO: Bit too much random shared state here.
pub(crate) struct ViewerContext {
    pub model_transform: Affine3A,

    pub device: WgpuDevice,
    pub egui_ctx: egui::Context,

    pub camera: Camera,
    pub controls: CameraController,

    pub process_messages_receiver: Option<Receiver<ViewerMessage>>,

    pub ui_control_receiver: UnboundedReceiver<UiControlMessage>,
    pub ui_control_sender: UnboundedSender<UiControlMessage>,

    pub filename: Option<String>,
    pub view_splats: Vec<Splats<Wgpu>>,
    pub frame: f32,

    pub ui_layout: UILayout,
}

impl ViewerContext {
    pub(crate) fn new(
        device: WgpuDevice,
        ctx: egui::Context,
        cam_settings: CameraSettings,
        controller: UnboundedReceiver<UiControlMessage>,
    ) -> Self {
        let model_transform = Affine3A::IDENTITY;

        let controls = CameraController::new(
            cam_settings.radius,
            cam_settings.pitch,
            cam_settings.yaw,
            cam_settings.radius_range,
            cam_settings.yaw_range,
            cam_settings.pitch_range,
        );

        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            cam_settings.focal,
            cam_settings.focal,
            glam::vec2(0.5, 0.5),
        );

        // TODO: Generalize this "inner control" logic.
        let (inner_send, inner_recv) = ::tokio::sync::mpsc::unbounded_channel();
        let sender = inner_send.clone();
        let ctx_spawn = ctx.clone();
        let mut controller = controller;
        task::spawn(async move {
            // Loop until there are no more messages, processing is done.
            while let Some(m) = controller.recv().await {
                ctx_spawn.request_repaint();

                // Give back to the runtime for a second.
                // This only really matters in the browser.
                tokio::task::yield_now().await;

                // If channel is closed, bail.
                if sender.send(m).is_err() {
                    break;
                }
            }
        });

        Self {
            model_transform,
            camera,
            controls,
            device,
            egui_ctx: ctx,
            process_messages_receiver: None,
            filename: None,
            view_splats: vec![],
            frame: 0.0,
            ui_layout: UILayout::default(),
            ui_control_receiver: inner_recv,
            ui_control_sender: inner_send,
        }
    }

    pub(crate) fn reset_camera(&mut self) {
        self.controls.reset();
        self.update_camera();
    }

    pub(crate) fn update_camera(&mut self) {
        let total_transform = self.model_transform * self.controls.transform();
        self.camera.position = total_transform.translation.into();
        self.camera.rotation = Quat::from_mat3a(&total_transform.matrix3);
    }

    pub(crate) fn set_up_axis(&mut self, up_axis: Vec3) {
        let rotation = Quat::from_rotation_arc(Vec3::Y, up_axis);
        let model_transform = Affine3A::from_rotation_translation(rotation, Vec3::ZERO).inverse();
        self.model_transform = model_transform;
    }

    pub(crate) fn load_splats_from_ply(&mut self, source: DataSource) {
        let device = self.device.clone();
        log::info!("Start data load");

        // Create a small channel. We don't want 10 updated splats to be stuck in the queue eating up memory!
        // Bigger channels could mean the train loop spends less time waiting for the UI though.
        let (sender, receiver) = channel(1);

        self.process_messages_receiver = Some(receiver);

        let ctx = self.egui_ctx.clone();

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

    pub(crate) fn save_splats_to_ply(&mut self, splats: Splats<Wgpu>) {
        let fut = async move {
            let file = rrfd::save_file("export.ply").await;

            // Not sure where/how to show this error if any.
            match file {
                Err(e) => {
                    log::error!("Failed to save file: {e}");
                }
                Ok(file) => {
                    let data = splat_export::splat_to_ply(splats).await;

                    let data = match data {
                        Ok(data) => data,
                        Err(e) => {
                            log::error!("Failed to serialize file: {e}");
                            return;
                        }
                    };

                    if let Err(e) = file.write(&data).await {
                        log::error!("Failed to write file: {e}");
                    }
                }
            }
        };

        tokio::task::spawn(fut);
    }

    pub(crate) fn process_control_messages(&mut self) {
        while let Ok(m) = self.ui_control_receiver.try_recv() {
            match m {
                UiControlMessage::LoadData(url) => {
                    self.load_splats_from_ply(DataSource::Url(url.to_owned()));
                }
                UiControlMessage::SaveSplats => {
                    self.save_splats_to_ply(self.current_splats().clone());
                }
                UiControlMessage::ResetCamera => {
                    self.reset_camera();
                }
            }
        }
    }

    pub(crate) fn current_splats(&self) -> &Splats<Wgpu> {
        const FPS: usize = 24;
        let frame: usize = ((self.frame * FPS as f32).floor() as usize) % self.view_splats.len();
        self.view_splats.get(frame).unwrap()
    }
}

pub(crate) fn parse_search(search: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    let search = search.trim_start_matches('?');
    for pair in search.split('&') {
        // Split each pair on '=' to separate key and value
        if let Some((key, value)) = pair.split_once('=') {
            // URL decode the key and value and insert into HashMap
            params.insert(
                urlencoding::decode(key).unwrap_or_default().into_owned(),
                urlencoding::decode(value).unwrap_or_default().into_owned(),
            );
        }
    }
    params
}
