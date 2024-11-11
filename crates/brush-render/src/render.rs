use super::{shaders, Backend, RenderAux};

use std::mem::{offset_of, size_of};

use crate::{
    camera::Camera,
    dim_check::DimCheck,
    kernels::{
        GatherGrads, GetTileBinEdges, MapGaussiansToIntersect, ProjectBackwards, ProjectSplats,
        ProjectVisible, Rasterize, RasterizeBackwards,
    },
    PrimaryBackend,
};

use brush_kernel::{
    calc_cube_count, create_dispatch_buffer, create_tensor, create_uniform_buffer, CubeCount,
};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::ops::IntTensorOps;
use burn::tensor::ops::{FloatTensor, FloatTensorOps};
use burn::tensor::{Tensor, TensorPrimitive};
use burn::{backend::autodiff::NodeID, tensor::BasicAutodiffOps};
use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        Autodiff,
    },
    tensor::Float,
};
use burn_wgpu::{JitTensor, WgpuRuntime};
use glam::uvec2;

pub const SH_C0: f32 = shaders::gather_grads::SH_C0;

pub const fn sh_coeffs_for_degree(degree: u32) -> u32 {
    (degree + 1).pow(2)
}

pub fn sh_degree_from_coeffs(coeffs_per_channel: u32) -> u32 {
    match coeffs_per_channel {
        1 => 0,
        4 => 1,
        9 => 2,
        16 => 3,
        25 => 4,
        _ => panic!("Invalid nr. of sh bases {coeffs_per_channel}"),
    }
}

fn render_forward(
    camera: &Camera,
    img_size: glam::UVec2,
    means: JitTensor<WgpuRuntime, f32>,
    log_scales: JitTensor<WgpuRuntime, f32>,
    quats: JitTensor<WgpuRuntime, f32>,
    sh_coeffs: JitTensor<WgpuRuntime, f32>,
    raw_opacities: JitTensor<WgpuRuntime, f32>,
    raster_u32: bool,
) -> (JitTensor<WgpuRuntime, f32>, RenderAux<PrimaryBackend>) {
    let device = &means.device.clone();
    let client = means.client.clone();

    // Check whether any work needs to be flushed.
    tracing::trace_span!("pre setup", sync_burn = true).in_scope(|| {});

    let _span = tracing::trace_span!("render_forward", sync_burn = true).entered();

    // Check whether dimesions are valid.
    DimCheck::new()
        .check_dims(&means, &["D".into(), 3.into()])
        .check_dims(&log_scales, &["D".into(), 3.into()])
        .check_dims(&quats, &["D".into(), 4.into()])
        .check_dims(&sh_coeffs, &["D".into(), "C".into(), 3.into()])
        .check_dims(&raw_opacities, &["D".into()]);

    // Divide screen into tiles.
    let tile_bounds = uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    );

    // A note on some confusing naming that'll be used throughout this function:
    // Gaussians are stored in various states of buffers, eg. at the start they're all in one big bufffer,
    // then we sparsely store some results, then sort gaussian based on depths, etc.
    // Overall this means there's lots of indices flying all over the place, and it's hard to keep track
    // what is indexing what. So, for some sanity, try to match a few "gaussian ids" (gid) variable names.
    // - Global Gaussin ID - global_gid
    // - Compacted Gaussian ID - compact_gid
    // - Per tile intersection depth sorted ID - tiled_gid
    // - Sorted by tile per tile intersection depth sorted ID - sorted_tiled_gid
    // Then, various buffers map between these, which are named x_from_y_gid, eg.
    //  global_from_compact_gid.

    // Tile rendering setup.
    let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape.dims[1] as u32);
    let total_splats = means.shape.dims[0] as u32;
    let uniforms_buffer = create_uniform_buffer(
        shaders::helpers::RenderUniforms {
            viewmat: camera.world_to_local().to_cols_array_2d(),
            focal: camera.focal(img_size).into(),
            pixel_center: camera.center(img_size).into(),
            img_size: img_size.into(),
            tile_bounds: tile_bounds.into(),
            num_visible: 0,
            sh_degree,
            total_splats,
            padding: 0,
        },
        device,
        &client,
    );

    let device = &means.device.clone();

    let num_points = means.shape.dims[0];
    let client = &means.client.clone();

    let (global_from_compact_gid, num_visible) = {
        let global_from_presort_gid = PrimaryBackend::int_zeros([num_points].into(), device);
        let depths = create_tensor::<f32, 1, _>([num_points], device, client);

        tracing::trace_span!("ProjectSplats", sync_burn = true).in_scope(||
            // SAFETY: wgsl FFI, kernel checked to have no OOB.
            unsafe {
            client.execute_unchecked(
                ProjectSplats::task(),
                calc_cube_count([num_points as u32], ProjectSplats::WORKGROUP_SIZE),
                vec![
                    uniforms_buffer.clone().handle.binding(),
                    means.clone().handle.binding(),
                    log_scales.clone().handle.binding(),
                    quats.clone().handle.binding(),
                    global_from_presort_gid.clone().handle.binding(),
                    depths.clone().handle.binding(),
                ],
            );
        });

        // Get just the number of visible splats from the uniforms buffer.
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        let num_visible = PrimaryBackend::int_slice(
            uniforms_buffer.clone(),
            &[num_vis_field_offset..num_vis_field_offset + 1],
        );

        let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort", sync_burn = true)
            .in_scope(|| {
                // Interpret the depth as a u32. This is fine for a radix sort, as long as the depth > 0.0,
                // which we know to be the case given how we cull splats.
                radix_argsort(depths, global_from_presort_gid, num_visible.clone(), 32)
            });

        (global_from_compact_gid, num_visible)
    };

    let projected_size = size_of::<shaders::helpers::ProjectedSplat>() / size_of::<f32>();
    let projected_splats = create_tensor::<f32, 2, _>([num_points, projected_size], device, client);

    // Number of tiles hit per splat. Has to be zerod as we later sum over this.
    let num_tiles_hit = PrimaryBackend::int_zeros([num_points].into(), device);
    let num_vis_wg = create_dispatch_buffer(num_visible.clone(), [shaders::helpers::MAIN_WG, 1, 1]);

    tracing::trace_span!("ProjectVisibile", sync_burn = true).in_scope(|| unsafe {
        client.execute_unchecked(
            ProjectVisible::task(),
            CubeCount::Dynamic(num_vis_wg.clone().handle.binding()),
            vec![
                uniforms_buffer.clone().handle.binding(),
                means.handle.binding(),
                log_scales.handle.binding(),
                quats.handle.binding(),
                sh_coeffs.handle.binding(),
                raw_opacities.handle.binding(),
                global_from_compact_gid.handle.clone().binding(),
                projected_splats.handle.clone().binding(),
                num_tiles_hit.handle.clone().binding(),
            ],
        );
    });

    let cum_tiles_hit = tracing::trace_span!("PrefixSum", sync_burn = true).in_scope(|| {
        // TODO: Only need to do this up to num_visible gaussians really.
        prefix_sum(num_tiles_hit)
    });

    let num_intersections =
        PrimaryBackend::int_slice(cum_tiles_hit.clone(), &[num_points - 1..num_points]);

    let num_tiles = tile_bounds[0] * tile_bounds[1];

    // Each intersection maps to a gaussian.
    let (tile_bins, compact_gid_from_isect) = {
        // On wasm, we cannot do a sync readback at all.
        // Instead, can just estimate a max number of intersects. All the kernels only handle the actual
        // cound of intersects, and spin up empty threads for the rest atm. In the future, could use indirect
        // dispatch to avoid this.
        // Estimating the max number of intersects can be a bad hack though... The worst case sceneario is so massive
        // that it's easy to run out of memory... How do we actually properly deal with this :/
        let max_intersects = num_points
            .saturating_mul(num_tiles as usize)
            .min(128 * 65535);

        let tile_id_from_isect = create_tensor::<i32, 1, _>([max_intersects], device, client);
        let compact_gid_from_isect = create_tensor::<i32, 1, _>([max_intersects], device, client);

        tracing::trace_span!("MapGaussiansToIntersect", sync_burn = true).in_scope(|| unsafe {
            client.execute_unchecked(
                MapGaussiansToIntersect::task(),
                CubeCount::Dynamic(num_vis_wg.handle.binding()),
                vec![
                    uniforms_buffer.clone().handle.binding(),
                    projected_splats.handle.clone().binding(),
                    cum_tiles_hit.handle.clone().binding(),
                    tile_id_from_isect.handle.clone().binding(),
                    compact_gid_from_isect.handle.clone().binding(),
                ],
            );
        });

        // We're sorting by tile ID, but we know beforehand what the maximum value
        // can be. We don't need to sort all the leading 0 bits!
        let bits = u32::BITS - num_tiles.leading_zeros();

        let (tile_id_from_isect, compact_gid_from_isect) =
            tracing::trace_span!("Tile sort", sync_burn = true).in_scope(|| {
                radix_argsort(
                    tile_id_from_isect,
                    compact_gid_from_isect,
                    num_intersections.clone(),
                    bits,
                )
            });

        let _span = tracing::trace_span!("GetTileBinEdges", sync_burn = true).entered();

        let tile_bins = PrimaryBackend::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            device,
        );
        unsafe {
            client.execute_unchecked(
                GetTileBinEdges::task(),
                CubeCount::Dynamic(
                    create_dispatch_buffer(
                        num_intersections.clone(),
                        GetTileBinEdges::WORKGROUP_SIZE,
                    )
                    .handle
                    .binding(),
                ),
                vec![
                    tile_id_from_isect.handle.clone().binding(),
                    num_intersections.handle.clone().binding(),
                    tile_bins.handle.clone().binding(),
                ],
            );
        }

        (tile_bins, compact_gid_from_isect)
    };

    let _span = tracing::trace_span!("Rasterize", sync_burn = true).entered();

    let out_dim = if raster_u32 {
        // Channels are packed into 4 bytes aka one float.
        1
    } else {
        4
    };

    let out_img = create_tensor(
        [img_size.y as usize, img_size.x as usize, out_dim],
        device,
        client,
    );

    // Only record the final visible splat per tile if we're not rendering a u32 buffer.
    // If we're renering a u32 buffer, we can't autodiff anyway, and final index is only needed for
    // the backward pass.
    let mut handles = vec![
        uniforms_buffer.clone().handle.binding(),
        compact_gid_from_isect.handle.clone().binding(),
        tile_bins.handle.clone().binding(),
        projected_splats.handle.clone().binding(),
        out_img.handle.clone().binding(),
    ];

    // Record the final visible splat per tile.
    let final_index =
        create_tensor::<i32, 2, _>([img_size.y as usize, img_size.x as usize], device, client);

    if !raster_u32 {
        handles.push(final_index.handle.clone().binding());
    }

    unsafe {
        client.execute_unchecked(
            Rasterize::task(raster_u32),
            calc_cube_count([img_size.x, img_size.y], Rasterize::WORKGROUP_SIZE),
            handles,
        );
    }

    (
        out_img,
        RenderAux {
            uniforms_buffer,
            num_visible,
            num_intersections,
            tile_bins,
            cum_tiles_hit,
            projected_splats,
            final_index,
            compact_gid_from_isect,
            global_from_compact_gid,
        },
    )
}

impl Backend for PrimaryBackend {
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: Tensor<Self, 2>,
        _xy_dummy: Tensor<Self, 2>,
        log_scales: Tensor<Self, 2>,
        quats: Tensor<Self, 2>,
        sh_coeffs: Tensor<Self, 3>,
        raw_opacity: Tensor<Self, 1>,
        render_u32_buffer: bool,
    ) -> (Tensor<Self, 3>, RenderAux<Self>) {
        let (out_img, aux) = render_forward(
            camera,
            img_size,
            means.into_primitive().tensor(),
            log_scales.into_primitive().tensor(),
            quats.into_primitive().tensor(),
            sh_coeffs.into_primitive().tensor(),
            raw_opacity.into_primitive().tensor(),
            render_u32_buffer,
        );

        (Tensor::from_primitive(TensorPrimitive::Float(out_img)), aux)
    }
}

#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    means: NodeID,
    log_scales: NodeID,
    quats: NodeID,
    raw_opac: NodeID,
    sh_degree: u32,
    out_img: B::FloatTensorPrimitive,
    aux: RenderAux<B>,
}

#[derive(Debug)]
struct RenderBackwards;

impl<C: CheckpointStrategy> Backend for Autodiff<PrimaryBackend, C> {
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: Tensor<Self, 2>,
        xy_dummy: Tensor<Self, 2>,
        log_scales: Tensor<Self, 2>,
        quats: Tensor<Self, 2>,
        sh_coeffs: Tensor<Self, 3>,
        raw_opacity: Tensor<Self, 1>,
        render_u32_buffer: bool,
    ) -> (Tensor<Self, 3>, RenderAux<Self>) {
        // Get backend tensors & dequantize if needed. Could try and support quantized inputs
        // in the future.
        let means = means.into_primitive().tensor();
        let xy_dummy = xy_dummy.into_primitive().tensor();
        let log_scales = log_scales.into_primitive().tensor();
        let quats = quats.into_primitive().tensor();
        let sh_coeffs = sh_coeffs.into_primitive().tensor();
        let raw_opacity = raw_opacity.into_primitive().tensor();

        // Render complete forward pass.
        let (out_img, aux) = render_forward(
            camera,
            img_size,
            means.clone().into_primitive(),
            log_scales.clone().into_primitive(),
            quats.clone().into_primitive(),
            sh_coeffs.clone().into_primitive(),
            raw_opacity.clone().into_primitive(),
            render_u32_buffer,
        );

        // Not sure why going into the autodiff float tensor type is so verbose.
        let diff_proj = <Float as BasicAutodiffOps<Self>>::from_inner(TensorPrimitive::Float(
            aux.projected_splats.clone(),
        ))
        .tensor();

        let auxc = aux.clone();
        let wrapped_aux = RenderAux::<Self> {
            projected_splats: diff_proj,
            uniforms_buffer: aux.uniforms_buffer,
            num_intersections: aux.num_intersections,
            num_visible: aux.num_visible,
            final_index: aux.final_index,
            cum_tiles_hit: aux.cum_tiles_hit,
            tile_bins: aux.tile_bins,
            compact_gid_from_isect: aux.compact_gid_from_isect,
            global_from_compact_gid: aux.global_from_compact_gid,
        };

        // Prepare backward pass, and check if we even need to do it. Store nodes that need gradients.
        let prep_nodes = RenderBackwards
            .prepare::<C>([
                means.clone().node,
                xy_dummy.clone().node,
                log_scales.clone().node,
                quats.clone().node,
                sh_coeffs.clone().node,
                raw_opacity.clone().node,
            ])
            .compute_bound()
            .stateful();

        let sh_degree = sh_degree_from_coeffs(sh_coeffs.primitive.shape.dims[1] as u32);

        match prep_nodes {
            OpsKind::Tracked(mut prep) => {
                // Save state needed for backward pass.
                let state = GaussianBackwardState {
                    means: prep.checkpoint(&means),
                    log_scales: prep.checkpoint(&log_scales),
                    quats: prep.checkpoint(&quats),
                    raw_opac: prep.checkpoint(&raw_opacity),
                    sh_degree,
                    aux: auxc,
                    out_img: out_img.clone(),
                };

                let finish = prep.finish(state, out_img);

                (
                    Tensor::from_primitive(TensorPrimitive::Float(finish)),
                    wrapped_aux,
                )
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just use the original operation without
                // keeping any state.
                (
                    Tensor::from_primitive(TensorPrimitive::Float(prep.finish(out_img))),
                    wrapped_aux,
                )
            }
        }
    }
}

impl Backward<PrimaryBackend, 6> for RenderBackwards {
    type State = GaussianBackwardState<PrimaryBackend>;

    fn backward(
        self,
        ops: Ops<Self::State, 6>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_gaussians backwards").entered();

        let state = ops.state;
        let aux = state.aux;

        let img_dimgs = state.out_img.shape.dims;
        let img_size = glam::uvec2(img_dimgs[1] as u32, img_dimgs[0] as u32);

        let v_output = grads.consume::<PrimaryBackend>(&ops.node);
        let client = &v_output.client;
        let device = &v_output.device;

        let means = checkpointer.retrieve_node_output::<FloatTensor<PrimaryBackend>>(state.means);
        let quats = checkpointer.retrieve_node_output::<FloatTensor<PrimaryBackend>>(state.quats);
        let log_scales =
            checkpointer.retrieve_node_output::<FloatTensor<PrimaryBackend>>(state.log_scales);
        let raw_opac =
            checkpointer.retrieve_node_output::<FloatTensor<PrimaryBackend>>(state.raw_opac);

        let num_points = means.shape.dims[0];
        let num_visible = aux.num_visible;

        let (v_xys, v_xys_global, v_conics, v_coeffs, v_opacities) = {
            let tile_bounds = uvec2(
                img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
                img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
            );

            let invocations = tile_bounds.x * tile_bounds.y;

            // These gradients are atomically added to so important to zero them.
            let v_xys_local = PrimaryBackend::float_zeros([num_points, 2].into(), device);
            let v_conics = PrimaryBackend::float_zeros([num_points, 3].into(), device);
            let v_colors = PrimaryBackend::float_zeros([num_points, 4].into(), device);

            // TODO: Properly register hardware atomic floats as a cube feature when
            // https://github.com/gfx-rs/wgpu/pull/6234 lands.
            //
            // On mac, this is needed as our wgpu version doesn't support CAS on metal yet...
            let hard_floats = cfg!(target_os = "macos");

            tracing::trace_span!("RasterizeBackwards", sync_burn = true).in_scope(|| unsafe {
                client.execute_unchecked(
                    RasterizeBackwards::task(hard_floats),
                    CubeCount::Static(invocations, 1, 1),
                    vec![
                        aux.uniforms_buffer.clone().handle.binding(),
                        aux.compact_gid_from_isect.handle.binding(),
                        aux.tile_bins.handle.binding(),
                        aux.projected_splats.handle.binding(),
                        aux.final_index.handle.binding(),
                        state.out_img.handle.binding(),
                        v_output.handle.binding(),
                        v_xys_local.clone().handle.binding(),
                        v_conics.clone().handle.binding(),
                        v_colors.clone().handle.binding(),
                    ],
                );
            });

            let v_coeffs_shape = [
                num_points,
                sh_coeffs_for_degree(state.sh_degree) as usize,
                3,
            ];
            let v_coeffs = PrimaryBackend::float_zeros(v_coeffs_shape.into(), device);
            let v_opacities = PrimaryBackend::float_zeros([num_points].into(), device);

            let _span = tracing::trace_span!("GatherGrads", sync_burn = true).entered();

            let num_vis_wg =
                create_dispatch_buffer(num_visible.clone(), GatherGrads::WORKGROUP_SIZE);

            let v_xys_global = PrimaryBackend::float_zeros([num_points, 2].into(), device);
            unsafe {
                client.execute_unchecked(
                    GatherGrads::task(),
                    CubeCount::Dynamic(num_vis_wg.handle.binding()),
                    vec![
                        aux.uniforms_buffer.clone().handle.binding(),
                        aux.global_from_compact_gid.clone().handle.binding(),
                        raw_opac.clone().handle.binding(),
                        means.clone().handle.binding(),
                        v_colors.clone().handle.binding(),
                        v_xys_local.clone().handle.binding(),
                        v_coeffs.handle.clone().binding(),
                        v_opacities.handle.clone().binding(),
                        v_xys_global.handle.clone().binding(),
                    ],
                );
            }

            (v_xys_local, v_xys_global, v_conics, v_coeffs, v_opacities)
        };

        // Create tensors to hold gradients.

        // Nb: these are packed vec3 values, special care is taken in the kernel to respect alignment.
        // Nb: These have to be zerod out - as we only write to visible splats.
        let v_means = PrimaryBackend::float_zeros([num_points, 3].into(), device);
        let v_scales = PrimaryBackend::float_zeros([num_points, 3].into(), device);
        let v_quats = PrimaryBackend::float_zeros([num_points, 4].into(), device);

        tracing::trace_span!("ProjectBackwards", sync_burn = true).in_scope(|| unsafe {
            client.execute_unchecked(
                ProjectBackwards::task(),
                calc_cube_count([num_points as u32], ProjectBackwards::WORKGROUP_SIZE),
                vec![
                    aux.uniforms_buffer.handle.binding(),
                    means.handle.binding(),
                    log_scales.handle.binding(),
                    quats.handle.binding(),
                    aux.global_from_compact_gid.handle.binding(),
                    v_xys.handle.clone().binding(),
                    v_conics.handle.binding(),
                    v_means.handle.clone().binding(),
                    v_scales.handle.clone().binding(),
                    v_quats.handle.clone().binding(),
                ],
            );
        });

        // Register gradients for parent nodes (This code is already skipped entirely
        // if no parent nodes require gradients).
        let [mean_parent, xys_parent, log_scales_parent, quats_parent, coeffs_parent, raw_opacity_parent] =
            ops.parents;

        if let Some(node) = mean_parent {
            grads.register::<PrimaryBackend>(node.id, v_means);
        }

        // Register the gradients for the dummy xy input.
        if let Some(node) = xys_parent {
            grads.register::<PrimaryBackend>(node.id, v_xys_global);
        }

        if let Some(node) = log_scales_parent {
            grads.register::<PrimaryBackend>(node.id, v_scales);
        }

        if let Some(node) = quats_parent {
            grads.register::<PrimaryBackend>(node.id, v_quats);
        }

        if let Some(node) = coeffs_parent {
            grads.register::<PrimaryBackend>(node.id, v_coeffs);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<PrimaryBackend>(node.id, v_opacities);
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use std::fs::File;
    use std::io::Read;

    use crate::{
        camera::{focal_to_fov, fov_to_focal},
        gaussian_splats::Splats,
        safetensor_utils::safetensor_to_burn,
    };

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use async_std::task;
    use brush_rerun::{BurnToImage, BurnToRerun};
    use burn::tensor::{Float, Int};
    use burn_wgpu::WgpuDevice;

    type DiffBack = Autodiff<PrimaryBackend>;
    use anyhow::{Context, Result};
    use safetensors::SafeTensors;

    const USE_RERUN: bool = false;

    #[test]
    fn renders_at_all() {
        // Check if rendering doesn't hard crash or anything.
        // These are some zero-sized gaussians, so we know
        // what the result should look like.
        let cam = Camera::new(
            glam::vec3(0.0, 0.0, 0.0),
            glam::Quat::IDENTITY,
            0.5,
            0.5,
            glam::vec2(0.5, 0.5),
        );
        let img_size = glam::uvec2(32, 32);
        let device = WgpuDevice::DefaultDevice;
        let num_points = 8;
        let means = Tensor::<DiffBack, 2, _>::zeros([num_points, 3], &device);
        let xy_dummy = Tensor::<DiffBack, 2, _>::zeros([num_points, 2], &device);
        let log_scales = Tensor::ones([num_points, 3], &device) * 2.0;
        let quats = Tensor::<_, 1, _>::from_floats(glam::Quat::IDENTITY.to_array(), &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, num_points);
        let sh_coeffs = Tensor::ones([num_points, 1, 3], &device);
        let raw_opacity = Tensor::zeros([num_points], &device);
        let (output, _) = DiffBack::render_splats(
            &cam,
            img_size,
            means,
            xy_dummy,
            log_scales,
            quats,
            sh_coeffs,
            raw_opacity,
            false,
        );

        let rgb = output.clone().slice([0..32, 0..32, 0..3]);
        let alpha = output.clone().slice([0..32, 0..32, 3..4]);
        let rgb_mean = rgb.clone().mean().to_data().as_slice::<f32>().unwrap()[0];
        let alpha_mean = alpha.clone().mean().to_data().as_slice::<f32>().unwrap()[0];
        assert_approx_eq!(rgb_mean, 0.0, 1e-5);
        assert_approx_eq!(alpha_mean, 0.0);
    }

    #[test]
    fn test_reference() -> Result<()> {
        let device = WgpuDevice::DefaultDevice;

        let crab_img = image::open("./test_cases/crab.png")?;
        // Convert the image to RGB format
        // Get the raw buffer
        let raw_buffer = crab_img.to_rgb8().into_raw();
        let crab_tens: Tensor<DiffBack, 3> = Tensor::<_, 1>::from_floats(
            raw_buffer
                .iter()
                .map(|&b| b as f32 / 255.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([crab_img.height() as usize, crab_img.width() as usize, 3]);

        let rec = if USE_RERUN {
            rerun::RecordingStreamBuilder::new("render test")
                .connect()
                .ok()
        } else {
            None
        };

        for (i, path) in ["tiny_case", "basic_case", "mix_case"].iter().enumerate() {
            let mut buffer = Vec::new();
            let _ =
                File::open(format!("./test_cases/{path}.safetensors"))?.read_to_end(&mut buffer)?;

            let tensors = SafeTensors::deserialize(&buffer)?;
            let splats = Splats::<DiffBack>::from_safetensors(&tensors, &device)?;

            let xys_ref = safetensor_to_burn::<DiffBack, 2>(tensors.tensor("xys")?, &device);
            let conics_ref = safetensor_to_burn::<DiffBack, 2>(tensors.tensor("conics")?, &device);
            let img_ref = safetensor_to_burn::<DiffBack, 3>(tensors.tensor("out_img")?, &device);
            let [h, w, _] = img_ref.dims();

            let fov = std::f64::consts::PI * 0.5;

            let focal = fov_to_focal(fov, w as u32);
            let fov_x = focal_to_fov(focal, w as u32);
            let fov_y = focal_to_fov(focal, h as u32);

            let cam = Camera::new(
                glam::vec3(0.0, 0.0, -8.0),
                glam::Quat::IDENTITY,
                fov_x,
                fov_y,
                glam::vec2(0.5, 0.5),
            );

            let (out, aux) = splats.render(&cam, glam::uvec2(w as u32, h as u32), false);

            let out_rgb = out.clone().slice([0..h, 0..w, 0..3]);
            if let Some(rec) = rec.as_ref() {
                task::block_on::<_, anyhow::Result<()>>(async {
                    rec.set_time_sequence("test case", i as i64);
                    rec.log("img/render", &out_rgb.clone().into_rerun_image().await)?;
                    rec.log("img/ref", &img_ref.clone().into_rerun_image().await)?;
                    rec.log(
                        "img/dif",
                        &(img_ref.clone() - out_rgb.clone()).into_rerun().await,
                    )?;
                    rec.log(
                        "images/tile depth",
                        &aux.read_tile_depth().into_rerun().await,
                    )?;

                    Ok(())
                })?;
            }

            let num_visible = task::block_on(aux.read_num_visible()) as usize;

            let projected_splats =
                Tensor::from_primitive(TensorPrimitive::Float(aux.projected_splats.clone()));

            let xys: Tensor<DiffBack, 2, Float> =
                projected_splats.clone().slice([0..num_visible, 0..2]);

            let conics: Tensor<DiffBack, 2, Float> =
                projected_splats.clone().slice([0..num_visible, 2..5]);

            let perm =
                Tensor::<DiffBack, 1, Int>::from_primitive(aux.global_from_compact_gid.clone());

            let xys_ref = xys_ref.select(0, perm.clone()).slice([0..num_visible]);
            let conics_ref = conics_ref.select(0, perm.clone()).slice([0..num_visible]);

            let grads = (out_rgb.clone() - crab_tens.clone())
                .powf_scalar(2.0)
                .mean()
                .backward();

            let v_opacities_ref =
                safetensor_to_burn::<DiffBack, 1>(tensors.tensor("v_opacities")?, &device).inner();
            let v_opacities = splats.raw_opacity.grad(&grads).context("opacities grad")?;

            let v_coeffs_ref =
                safetensor_to_burn::<DiffBack, 3>(tensors.tensor("v_coeffs")?, &device).inner();
            let v_coeffs = splats.sh_coeffs.grad(&grads).context("coeffs grad")?;

            let v_quats = splats.rotation.grad(&grads).context("quats grad")?;
            let v_quats_ref =
                safetensor_to_burn::<DiffBack, 2>(tensors.tensor("v_quats")?, &device).inner();

            let v_scales = splats.log_scales.grad(&grads).context("scales grad")?;
            let v_scales_ref =
                safetensor_to_burn::<DiffBack, 2>(tensors.tensor("v_scales")?, &device).inner();

            let v_means_ref =
                safetensor_to_burn::<DiffBack, 2>(tensors.tensor("v_means")?, &device).inner();
            let v_means = splats.means.grad(&grads).context("means grad")?;

            let v_xys_ref =
                safetensor_to_burn::<DiffBack, 2>(tensors.tensor("v_xy")?, &device).inner();
            let v_xys = splats.xys_dummy.grad(&grads).context("no xys grad")?;

            assert!(xys.all_close(xys_ref, Some(1e-4), Some(1e-10)));

            // TODO: Annoying that these aren't as close.
            assert!(conics.all_close(conics_ref, Some(1e-4), Some(5e-7)));

            // Slightly less precise than other values. This might be because
            // gSplat uses halfs for the image blending.
            assert!(out_rgb.all_close(img_ref, Some(1e-4), Some(1e-9)));
            assert!(v_xys.all_close(v_xys_ref, Some(1e-4), Some(1e-9)));
            assert!(v_opacities.all_close(v_opacities_ref, Some(1e-4), Some(1e-10)));
            assert!(v_coeffs.all_close(v_coeffs_ref, Some(1e-4), Some(1e-9)));
            assert!(v_scales.all_close(v_scales_ref, Some(1e-4), Some(1e-9)));
            assert!(v_means.all_close(v_means_ref, Some(1e-4), Some(1e-9)));

            // TODO: Fix this test.
            assert!(v_quats.all_close(v_quats_ref, Some(1e-1), Some(1e-1)));
        }
        Ok(())
    }

    // #[test]
    // fn test_mean_grads() {
    //     let cam = Camera::new(glam::vec3(0.0, 0.0, -5.0), glam::Quat::IDENTITY, 0.5, 0.5);
    //     let num_points = 1;

    //     let img_size = glam::uvec2(16, 16);
    //     let device = WgpuDevice::BestAvailable;

    //     let means = Tensor::<DiffBack, 2, _>::zeros([num_points, 3], &device).require_grad();
    //     let log_scales = Tensor::ones([num_points, 3], &device).require_grad();
    //     let quats = Tensor::from_data(glam::Quat::IDENTITY.to_array(), &device)
    //         .unsqueeze_dim(0)
    //         .repeat(0, num_points)
    //         .require_grad();
    //     let sh_coeffs = Tensor::zeros([num_points, 4], &device).require_grad();
    //     let raw_opacity = Tensor::zeros([num_points], &device).require_grad();

    //     let mut dloss_dmeans_stat = Tensor::zeros([num_points], &device);

    //     // Calculate a stochasic gradient estimation by perturbing random dimensions.
    //     let num_iters = 50;

    //     for _ in 0..num_iters {
    //         let eps = 1e-4;

    //         let flip_vec = Tensor::<DiffBack, 1>::random(
    //             [num_points],
    //             burn::tensor::Distribution::Uniform(-1.0, 1.0),
    //             &device,
    //         );
    //         let seps = flip_vec * eps;

    //         let l1 = render(
    //             &cam,
    //             img_size,
    //             means.clone(),
    //             log_scales.clone(),
    //             quats.clone(),
    //             sh_coeffs.clone(),
    //             raw_opacity.clone() - seps.clone(),
    //             glam::Vec3::ZERO,
    //         )
    //         .0
    //         .mean();

    //         let l2 = render(
    //             &cam,
    //             img_size,
    //             means.clone(),
    //             log_scales.clone(),
    //             quats.clone(),
    //             sh_coeffs.clone(),
    //             raw_opacity.clone() + seps.clone(),
    //             glam::Vec3::ZERO,
    //         )
    //         .0
    //         .mean();

    //         let df = l2 - l1;
    //         dloss_dmeans_stat = dloss_dmeans_stat + df * (seps * 2.0).recip();
    //     }

    //     dloss_dmeans_stat = dloss_dmeans_stat / (num_iters as f32);
    //     let dloss_dmeans_stat = dloss_dmeans_stat.into_data().value;

    //     let loss = render(
    //         &cam,
    //         img_size,
    //         means.clone(),
    //         log_scales.clone(),
    //         quats.clone(),
    //         sh_coeffs.clone(),
    //         raw_opacity.clone(),
    //         glam::Vec3::ZERO,
    //     )
    //     .0
    //     .mean();
    //     // calculate numerical gradients.
    //     // Compare to reference value.

    //     // Check if rendering doesn't hard crash or anything.
    //     // These are some zero-sized gaussians, so we know
    //     // what the result should look like.
    //     let grads = loss.backward();

    //     // Get the gradient of the rendered image.
    //     let dloss_dmeans = (Tensor::<BurnBack, 1>::from_primitive(
    //         grads.get(&raw_opacity.clone().into_primitive()).unwrap(),
    //     ))
    //     .into_data()
    //     .value;

    //     println!("Stat grads {dloss_dmeans_stat:.5?}");
    //     println!("Calc grads {dloss_dmeans:.5?}");

    //     // TODO: These results don't make sense at all currently, which is either
    //     // mildly bad news or very bad news :)
    // }
}
