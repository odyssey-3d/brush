#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use tokio_with_wasm::alias as tokio;

#[cfg(target_family = "wasm")]
use wasm_bindgen::JsCast;

fn main() {
    let wgpu_options = brush_ui::create_egui_options();

    // Unused.
    let (_, rec) = ::tokio::sync::mpsc::unbounded_channel();

    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            use tracing_subscriber::layer::SubscriberExt;

            let subscriber = tracing_subscriber::registry().with(tracing_wasm::WASMLayer::new(Default::default()));
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        } else if #[cfg(feature = "tracy")] {
            use tracing_subscriber::layer::SubscriberExt;
            let subscriber = tracing_subscriber::registry()
                .with(tracing_tracy::TracyLayer::default());
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async {
            env_logger::init();

            // NB: Load carrying icon. egui at head fails when no icon is included
            // as the built-in one is git-lfs which cargo doesn't clone properly.
            let icon =
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .unwrap();

            let native_options = eframe::NativeOptions {
                // Build app display.
                viewport: egui::ViewportBuilder::default()
                    .with_inner_size(egui::Vec2::new(1450.0, 1200.0))
                    .with_active(true)
                    .with_icon(std::sync::Arc::new(icon)),
                wgpu_options,
                ..Default::default()
            };

            eframe::run_native(
                "Odyssey Explorer",
                native_options,
                Box::new(move |cc| Ok(Box::new(odyexp_viewer::viewer::Viewer::new(cc, None, rec)))),
            )
            .unwrap();
        });
    }

    #[cfg(target_family = "wasm")]
    {
        if cfg!(debug_assertions) {
            eframe::WebLogger::init(log::LevelFilter::Debug).ok();
        }

        let document = web_sys::window().unwrap().document().unwrap();

        if let Some(canvas) = document
            .get_element_by_id("main_canvas")
            .and_then(|x| x.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        {
            // On wasm, run as a local task.
            tokio::spawn(async {
                let web_options = eframe::WebOptions {
                    wgpu_options,
                    ..Default::default()
                };

                eframe::WebRunner::new()
                    .start(
                        canvas,
                        web_options,
                        Box::new(|cc| {
                            Ok(Box::new(odyexp_viewer::viewer::Viewer::new(cc, None, rec)))
                        }),
                    )
                    .await
                    .expect("failed to start eframe");
            });
        }
    }
}

#[cfg(target_family = "wasm")]
mod embedded {
    use ::tokio::sync::mpsc::UnboundedSender;
    use tokio_with_wasm::alias as tokio;

    use odyexp_viewer::app_context::UiControlMessage;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct EmbeddedViewer {
        ui_control: UnboundedSender<UiControlMessage>,
    }

    #[wasm_bindgen]
    impl EmbeddedViewer {
        #[wasm_bindgen(constructor)]
        pub fn new(canvas_name: &str, url: &str) -> EmbeddedViewer {
            let wgpu_options = brush_ui::create_egui_options();
            let document = web_sys::window().unwrap().document().unwrap();
            let canvas = document
                .get_element_by_id(canvas_name)
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();

            // Unused.
            let (send, rec) = ::tokio::sync::mpsc::unbounded_channel();

            let url = url.to_owned();
            // On wasm, run as a local task.
            tokio::spawn(async {
                eframe::WebRunner::new()
                    .start(
                        canvas,
                        eframe::WebOptions {
                            wgpu_options,
                            ..Default::default()
                        },
                        Box::new(|cc| {
                            Ok(Box::new(odyexp_viewer::viewer::Viewer::new(
                                cc,
                                Some(url),
                                rec,
                            )))
                        }),
                    )
                    .await
                    .expect("failed to start eframe");
            });

            EmbeddedViewer { ui_control: send }
        }

        #[wasm_bindgen]
        pub fn load_url(&self, url: &str) {
            // If channel is dropped, don't do anything.
            let _ = self
                .ui_control
                .send(UiControlMessage::LoadData(url.to_owned()));
        }

        #[wasm_bindgen]
        pub fn save_splats(&self) {
            let _ = self.ui_control.send(UiControlMessage::SaveSplats);
        }

        #[wasm_bindgen]
        pub fn reset_camera(&self) {
            let _ = self.ui_control.send(UiControlMessage::ResetCamera);
        }
    }
}

#[cfg(target_family = "wasm")]
pub use embedded::*;
