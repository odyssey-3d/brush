use ::tokio::sync::mpsc::UnboundedReceiver;
use eframe::egui;

use crate::{
    app_context::{parse_search, UiControlMessage, AppContext, AppMessage},
    camera_controller::parse_camera_settings,
    main_panel::MainPanel,
    toolbar::Toolbar,
    top_panel::TopPanel,
};

pub struct Viewer {
    app_context: AppContext,
    main_panel: MainPanel,
    top_panel: TopPanel,
    toolbar: Toolbar,
}

impl Viewer {
    pub fn new(
        cc: &eframe::CreationContext,
        start_uri: Option<String>,
        controller: UnboundedReceiver<UiControlMessage>,
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

        let cam_settings = parse_camera_settings(search_params);
        let context = AppContext::new(
            device.clone(),
            cc.egui_ctx.clone(),
            cam_settings,
            controller,
        );

        let mut start_url = None;
        if cfg!(target_family = "wasm") {
            if let Some(window) = web_sys::window() {
                if let Ok(search) = window.location().search() {
                    if let Ok(search_params) = web_sys::UrlSearchParams::new_with_str(&search) {
                        // this is from brush's url string format
                        if let Some(url) = search_params.get("url") {
                            start_url = Some(url.to_owned());
                        }
                        //this is from our url string format
                        if let Some(ply_url) = search_params.get("ply") {
                            start_url = Some(ply_url.to_owned());
                        }
                    }
                }
            }
        }

        if let Some(start_url) = start_url {
            let _ = context
                .ui_control_sender
                .send(UiControlMessage::LoadData(start_url.to_owned()));
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

        self.app_context.process_control_messages();

        if let Some(rec) = self.app_context.process_messages_receiver.as_mut() {
            let mut messages = vec![];

            while let Ok(message) = rec.try_recv() {
                messages.push(message);
            }

            for message in messages {
                match message.clone() {
                    AppMessage::ViewSplats {
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
                    AppMessage::StartLoading { filename } => {
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
