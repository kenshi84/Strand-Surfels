// SPDX-License-Identifier: MIT
//
// Host entry points of the strand surfel CUDA extension (see ext.cpp for the
// Python bindings and strand_surfels/rendering for the autograd wrappers).

#pragma once

#include <cstdint>
#include <tuple>

#include <ATen/core/Tensor.h>

namespace strand_surfels {

// projection.cu ------------------------------------------------------------

// returns radii [N] int32, means2d [N, 2], depths [N], ray_transforms [N, 9],
// dirs [N, 3]
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_surfels_fwd(
    const at::Tensor &means,     // [N, 3]
    const at::Tensor &quats,     // [N, 4]
    const at::Tensor &scales,    // [N, 2]
    const at::Tensor &opacities, // [N]
    const at::Tensor &viewmat,   // [4, 4]
    const at::Tensor &K,         // [3, 3]
    int64_t image_width,
    int64_t image_height,
    double scale_modifier
);

// returns v_means [N, 3], v_quats [N, 4], v_scales [N, 2]
std::tuple<at::Tensor, at::Tensor, at::Tensor> project_surfels_bwd(
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &viewmat,
    const at::Tensor &K,
    double scale_modifier,
    const at::Tensor &radii,
    const at::Tensor &ray_transforms,
    const at::Tensor &v_means2d,        // [N, 2]
    const at::Tensor &v_ray_transforms, // [N, 9]
    const at::Tensor &v_dirs            // [N, 3]
);

// intersect.cu -------------------------------------------------------------

// returns tile_ranges [n_tiles, 2] int32, flatten_ids [n_isects] int32,
// bucket_offsets [n_tiles] int32 (inclusive prefix sum), n_buckets
std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t> intersect_tiles(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    int64_t image_width,
    int64_t image_height
);

// rasterize_fwd.cu ---------------------------------------------------------

// returns render_colors [3, H, W], render_aux [7, H, W],
// final_state [3, H, W], contributors [2, H, W] int32, max_contrib [n_tiles]
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rasterize_fwd(
    const at::Tensor &means2d,
    const at::Tensor &ray_transforms,
    const at::Tensor &colors,
    const at::Tensor &dirs,
    const at::Tensor &opacities,
    const at::Tensor &background,
    int64_t image_width,
    int64_t image_height,
    const at::Tensor &tile_ranges,
    const at::Tensor &flatten_ids,
    const at::Tensor &bucket_offsets,
    bool save_samples,
    const at::Tensor &bucket_to_tile, // [n_buckets] int32
    const at::Tensor &sampled_state,  // [n_buckets, 3, 256]
    const at::Tensor &sampled_color,  // [n_buckets, 3, 256]
    const at::Tensor &sampled_aux     // [n_buckets, 4, 256]
);

// rasterize_bwd.cu ---------------------------------------------------------

// returns v_means2d [N, 2], v_ray_transforms [N, 9], v_colors [N, 3],
// v_dirs [N, 3], v_opacities [N], v_pse [N]
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rasterize_bwd(
    const at::Tensor &means2d,
    const at::Tensor &ray_transforms,
    const at::Tensor &colors,
    const at::Tensor &dirs,
    const at::Tensor &opacities,
    const at::Tensor &background,
    int64_t image_width,
    int64_t image_height,
    const at::Tensor &tile_ranges,
    const at::Tensor &flatten_ids,
    const at::Tensor &bucket_offsets,
    const at::Tensor &bucket_to_tile,
    const at::Tensor &max_contrib,
    const at::Tensor &sampled_state,
    const at::Tensor &sampled_color,
    const at::Tensor &sampled_aux,
    int64_t n_buckets,
    const at::Tensor &render_colors,
    const at::Tensor &render_aux,
    const at::Tensor &final_state,
    const at::Tensor &contributors,
    const at::Tensor &v_render_colors, // [3, H, W]
    const at::Tensor &v_render_aux,    // [7, H, W]
    const at::Tensor &v_error_map      // [1, H, W]
);

// spherical_harmonics.cu ----------------------------------------------------

// colors [N, 3] of the visible surfels (radii > 0) from SH coefficients [N, K, 3]
at::Tensor sh_to_color_fwd(
    int64_t degree,
    const at::Tensor &coeffs,
    const at::Tensor &means,
    const at::Tensor &camera_center,
    const at::Tensor &radii
);

// returns v_coeffs [N, K, 3], v_means [N, 3]
std::tuple<at::Tensor, at::Tensor> sh_to_color_bwd(
    int64_t degree,
    const at::Tensor &coeffs,
    const at::Tensor &means,
    const at::Tensor &camera_center,
    const at::Tensor &radii,
    const at::Tensor &colors,
    const at::Tensor &v_colors
);

// geometry_loss.cu -----------------------------------------------------------

// returns the loss map [3, H, W] and, if need_grad, its per-pixel derivatives
// with respect to depth [H, W] and direction [3, H, W]
std::tuple<at::Tensor, at::Tensor, at::Tensor> geometry_loss_fwd(
    const at::Tensor &depth,     // [H, W]
    const at::Tensor &dirs,      // [3, H, W]
    const at::Tensor &int_map,   // [3, H, W]
    const at::Tensor &m_inv,     // [3, 3]
    const at::Tensor &p_col3,    // [3]
    const at::Tensor &line_proj, // [3, 6]
    bool need_grad
);

// returns v_depth [H, W], v_dirs [3, H, W]
std::tuple<at::Tensor, at::Tensor> geometry_loss_bwd(
    const at::Tensor &int_map,
    const at::Tensor &jac_depth,
    const at::Tensor &jac_dir,
    const at::Tensor &v_loss // [3, H, W]
);

// 2D orientation angle [H, W] in [-pi/2, pi/2] of the projected strand directions
at::Tensor orientation_2d(
    const at::Tensor &depth,
    const at::Tensor &dirs,
    const at::Tensor &m_inv,
    const at::Tensor &p_col3,
    const at::Tensor &line_proj
);

// adam.cu -------------------------------------------------------------------

// in-place Adam step (without bias correction) on the visible surfels only
void sparse_adam_update(
    at::Tensor &param,
    const at::Tensor &grad,
    at::Tensor &exp_avg,
    at::Tensor &exp_avg_sq,
    const at::Tensor &visible, // [N] bool
    double lr,
    double beta1,
    double beta2,
    double eps
);

// gl_interop.cpp -------------------------------------------------------------

// CUDA-OpenGL interop for the Polyscope viewer (handles are integers)
namespace gl_interop {
uint64_t register_gl_buffer(uint64_t gl_buffer_id);
uint64_t register_gl_image_2d(uint64_t gl_texture_id);
void unregister_resource(uint64_t handle);
uint64_t map_resource_and_get_array(uint64_t handle);
std::tuple<uint64_t, uint64_t> map_resource_and_get_pointer(uint64_t handle);
void unmap_resource(uint64_t handle);
void memcpy_device(uint64_t dst, uint64_t src, uint64_t n_bytes);
void memcpy_2d_to_array(uint64_t dst_array, uint64_t src, uint64_t width_bytes, uint64_t height);
} // namespace gl_interop

} // namespace strand_surfels
