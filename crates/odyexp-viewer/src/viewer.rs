use eframe::egui;

use crate::{
    app_context::{ViewerContext, ViewerMessage},
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
    pub fn new(cc: &eframe::CreationContext) -> Self {
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

        let up_axis = glam::Vec3::Y;
        let mut context = ViewerContext::new(device.clone(), cc.egui_ctx.clone(), up_axis);

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
