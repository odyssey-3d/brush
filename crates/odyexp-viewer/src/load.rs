use std::pin::Pin;

use async_fn_stream::try_fn_stream;

use odyexp_io::splat_import;

use ::tokio::io::AsyncReadExt;
use ::tokio::{io::AsyncRead, io::BufReader};

use tokio_stream::{Stream, StreamExt};

use burn_wgpu::WgpuDevice;

use crate::app_context::AppMessage;

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

                //strip url so you only have the .ply
                let stripped_url = url.split(".ply").next().unwrap().to_string();
                let mut stripped_url = stripped_url.rsplit("/");
                let mut filename = stripped_url.next().unwrap().to_string();
                if let Some(second) = stripped_url.next() {
                    filename = format!("{}/{}.ply", second, filename);
                }

                Ok((
                    Box::pin(tokio_util::io::StreamReader::new(mapped)),
                    filename.to_string(),
                ))
            }
        }
    }
}

pub(crate) fn process_loading_loop(
    source: DataSource,
    device: WgpuDevice,
) -> Pin<Box<impl Stream<Item = anyhow::Result<AppMessage>>>> {
    let stream = try_fn_stream(|emitter| async move {
        let _ = emitter.emit(AppMessage::NewSource).await;

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

            let _ = emitter.emit(AppMessage::StartLoading { filename }).await;

            let subsample = None; // Subsampling a trained ply doesn't really make sense.
            let splat_stream = splat_import::load_splat_from_ply(data, subsample, device.clone());

            let mut splat_stream = std::pin::pin!(splat_stream);

            while let Some(message) = splat_stream.next().await {
                let message = message?;
                emitter
                    .emit(AppMessage::ViewSplats {
                        up_axis: message.meta.up_axis,
                        splats: Box::new(message.splats),
                        frame: message.meta.current_frame,
                    })
                    .await;
            }

            emitter.emit(AppMessage::DoneLoading).await;
        } else if peek.starts_with("<!DOCTYPE html>".as_bytes()) {
            anyhow::bail!("Failed to download data (are you trying to download from Google Drive? You might have to use the proxy.")
        } else {
            anyhow::bail!("only zip and ply files are supported.");
        }

        Ok(())
    });

    Box::pin(stream)
}
