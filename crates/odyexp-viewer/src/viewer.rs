use tokio_with_wasm::alias as tokio;

use eframe::egui;

use crate::app_context::{ViewerContext, ViewerMessage};
use crate::main_panel::MainPanel;
use crate::top_panel::TopPanel;

pub struct Viewer {
    app_context: ViewerContext,
    main_panel: MainPanel,
    top_panel: TopPanel,
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

        // let mut start_url = None;
        // if cfg!(target_family = "wasm") {
        //     if let Some(window) = web_sys::window() {
        //         if let Ok(search) = window.location().search() {
        //             if let Ok(search_params) = web_sys::UrlSearchParams::new_with_str(&search) {
        //                 let url = search_params.get("url");
        //                 start_url = url;
        //             }
        //         }
        //     }
        // }

        // let mut tiles: Tiles<PaneType> = egui_tiles::Tiles::default();
        let up_axis = glam::Vec3::Y;
        let context = ViewerContext::new(device.clone(), cc.egui_ctx.clone(), up_axis);

        let main_panel = MainPanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
        );

        let top_panel = TopPanel::new();

        // let loading_subs = vec![
        //     tiles.insert_pane(Box::new(LoadDataPanel::new())),
        //     tiles.insert_pane(Box::new(PresetsPanel::new())),
        // ];
        // let loading_pane = tiles.insert_tab_tile(loading_subs);

        // #[allow(unused_mut)]
        // let mut sides = vec![
        //     loading_pane,
        //     tiles.insert_pane(Box::new(StatsPanel::new(
        //         device.clone(),
        //         state.adapter.clone(),
        //     ))),
        // ];

        // #[cfg(not(target_family = "wasm"))]
        // {
        //     sides.push(tiles.insert_pane(Box::new(crate::panels::RerunPanel::new(device.clone()))));
        // }

        // if cfg!(feature = "tracing") {
        //     sides.push(tiles.insert_pane(Box::new(TracingPanel::default())));
        // }

        // let side_panel = tiles.insert_vertical_tile(sides);
        // let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        // let mut lin = egui_tiles::Linear::new(
        //     egui_tiles::LinearDir::Horizontal,
        //     vec![side_panel, scene_pane_id],
        // );
        // lin.shares.set_share(side_panel, 0.4);

        // let root_container = tiles.insert_container(lin);
        // let tree = egui_tiles::Tree::new("viewer_tree", root_container, tiles);

        // let mut tree_ctx = ViewerTree { context };

        // if let Some(start_url) = start_url {
        //     tree_ctx.context.start_data_load(
        //         DataSource::Url(start_url.to_owned()),
        //         LoadDatasetArgs::default(),
        //         LoadInitArgs::default(),
        //         TrainConfig::default(),
        //     );
        // }

        Viewer {
            app_context: context,
            main_panel,
            top_panel,
            // tree,
            // tree_ctx,
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

                ctx.request_repaint();
            }
        }

        egui_extras::install_image_loaders(ctx);
        self.top_panel.show(&mut self.app_context);
        self.main_panel.show(&self.app_context);
    }
}
