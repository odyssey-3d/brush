use ::tokio::sync::mpsc::UnboundedReceiver;
use eframe::egui;

use crate::{
    app_context::{parse_search, CameraSettings, UiControlMessage, ViewerContext, ViewerMessage},
    load::DataSource,
    main_panel::MainPanel,
    toolbar::Toolbar,
    top_panel::TopPanel,
};

pub struct Viewer {
    app_context: ViewerContext,
    main_panel: MainPanel,
    top_panel: TopPanel,
    toolbar: Toolbar,
}

impl Viewer {
    pub fn new(
        cc: &eframe::CreationContext,
        start_uri: Option<String>,
        _controller: UnboundedReceiver<UiControlMessage>,
    ) -> Self {
        let state = cc.wgpu_render_state.as_ref().unwrap();
        let device = brush_ui::create_wgpu_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        #[cfg(target_family = "wasm")]
        let start_uri = start_uri.or(web_sys::window().and_then(|w| w.location().search().ok()));
        let search_params = parse_search(&start_uri.unwrap_or("".to_owned()));

        let focal = search_params
            .get("focal")
            .and_then(|f| f.parse().ok())
            .unwrap_or(0.5);
        let radius = search_params
            .get("radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(4.0);
        let min_radius = search_params
            .get("min_radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(1.0);
        let max_radius = search_params
            .get("max_radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(10.0);

        let min_yaw = search_params
            .get("min_yaw")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MIN);
        let max_yaw = search_params
            .get("max_yaw")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MAX);

        let min_pitch = search_params
            .get("min_pitch")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MIN);
        let max_pitch = search_params
            .get("max_pitch")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MAX);

        let cam_settings = CameraSettings {
            focal,
            radius,
            radius_range: min_radius..max_radius,
            yaw_range: min_yaw..max_yaw,
            pitch_range: min_pitch..max_pitch,
        };
        let mut context = ViewerContext::new(device.clone(), cc.egui_ctx.clone(), cam_settings);

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

        if let Some(start_url) = start_url {
            context.start_ply_load(DataSource::Url(start_url.to_owned()));
        }

        let main_panel = MainPanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
        );

        let top_panel = TopPanel::new();
        let toolbar = Toolbar::new();

        Viewer {
            app_context: context,
            main_panel,
            top_panel,
            toolbar,
        }
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if let Some(rec) = self.app_context.receiver.as_mut() {
            let mut messages = vec![];

            while let Ok(message) = rec.try_recv() {
                messages.push(message);
            }

            for message in messages {
                match message.clone() {
                    ViewerMessage::ViewSplats {
                        up_axis,
                        splats,
                        frame,
                    } => {
                        self.app_context.set_up_axis(up_axis);
                        self.app_context.view_splats.truncate(frame);
                        log::info!("Received splat at {frame}");
                        self.app_context.view_splats.push(*splats.clone());
                        self.app_context.frame = frame as f32 - 0.5;
                    }
                    ViewerMessage::StartLoading { filename } => {
                        self.app_context.filename = Some(filename);
                        self.app_context.view_splats = vec![];
                    }
                    _ => {}
                }

                self.top_panel.on_message(&message, &mut self.app_context);
                self.main_panel.on_message(&message, &mut self.app_context);
                self.toolbar.on_message(&message, &mut self.app_context);

                ctx.request_repaint();
            }
        }

        egui_extras::install_image_loaders(ctx);
        self.top_panel.show(&mut self.app_context);
        self.main_panel.show(&mut self.app_context);
        self.toolbar.show(&mut self.app_context);
    }
}
