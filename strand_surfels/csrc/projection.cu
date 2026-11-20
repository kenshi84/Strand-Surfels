// SPDX-License-Identifier: Apache-2.0
//
// Projection of 2D Gaussian surfels into a pinhole camera.
//
// Derived from the 2DGS projection of gsplat
// (https://github.com/nerfstudio-project/gsplat, Apache-2.0,
// gsplat/cuda/csrc/Projection2DGSFused.cu and Projection2DGS.cuh).
// Changes with respect to gsplat:
//   * single camera, no batching; quaternions and 2D scales as surfel inputs
//   * opacity dependent bounding box cut-off and square screen radius
//   * the per-surfel output is the world space strand direction (first
//     tangent axis of the surfel) instead of the camera space normal
//   * screen-space bounding box gradient of the legacy 2DGS rasterizer

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "common.h"
#include "ops.h"

namespace strand_surfels {

namespace cg = cooperative_groups;

// The ray transform of a surfel is the 3x3 matrix KWH that maps homogeneous
// surfel coordinates (u, v, 1) to homogeneous pixel coordinates
// (x * z, y * z, z). Its rows Tu, Tv, Tw are stored row-major ([N, 9]).
// Tw is the camera space position, so Tw . (u, v, 1) is the camera depth.

__global__ void project_surfels_fwd_kernel(
    const uint32_t N,
    const float *__restrict__ means,     // [N, 3]
    const float *__restrict__ quats,     // [N, 4] (w, x, y, z), not normalised
    const float *__restrict__ scales,    // [N, 2]
    const float *__restrict__ opacities, // [N]
    const float *__restrict__ viewmat,   // [4, 4] world-to-camera
    const float *__restrict__ K,         // [3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float scale_modifier,
    // outputs
    int32_t *__restrict__ radii,        // [N] square screen radius in pixels, 0 if culled
    float *__restrict__ means2d,        // [N, 2] screen space center
    float *__restrict__ depths,         // [N] camera depth of the center
    float *__restrict__ ray_transforms, // [N, 9]
    float *__restrict__ dirs            // [N, 3] world space strand direction
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    radii[idx] = 0;

    mat3 R_cam;
    vec3 t_cam;
    load_viewmat(viewmat, R_cam, t_cam);
    const vec3 mean_c = R_cam * glm::make_vec3(means + 3 * idx) + t_cam;
    if (mean_c.z <= NEAR_PLANE) {
        return;
    }

    // surfel tangent axes (scaled) in camera space
    const mat3 R = quat_to_rotmat(glm::make_vec4(quats + 4 * idx));
    const vec3 axis_u = R_cam * (R[0] * (scales[2 * idx] * scale_modifier));
    const vec3 axis_v = R_cam * (R[1] * (scales[2 * idx + 1] * scale_modifier));

    // KWH: W H maps (u, v, 1) to camera space, K projects to pixels
    const float fx = K[0], cx = K[2], fy = K[4], cy = K[5];
    const vec3 Tw = vec3(axis_u.z, axis_v.z, mean_c.z);
    const vec3 Tu = fx * vec3(axis_u.x, axis_v.x, mean_c.x) + cx * Tw;
    const vec3 Tv = fy * vec3(axis_u.y, axis_v.y, mean_c.y) + cy * Tw;

    // Screen space center and extent of the surfel ellipse at the opacity
    // dependent cut-off (closed form, see hbb1/diff-surfel-rasterization#8).
    const float cutoff = opacity_cutoff(opacities[idx]);
    const vec3 t = vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
    const float d = glm::dot(t, Tw * Tw);
    if (d == 0.0f) {
        return;
    }
    const vec3 f = (1.0f / d) * t;
    const vec2 center = vec2(glm::dot(f, Tu * Tw), glm::dot(f, Tv * Tw));
    const vec2 half_extent_sq =
        center * center - vec2(glm::dot(f, Tu * Tu), glm::dot(f, Tv * Tv));
    const vec2 half_extent =
        vec2(sqrtf(fmaxf(1e-4f, half_extent_sq.x)), sqrtf(fmaxf(1e-4f, half_extent_sq.y)));
    const int radius = (int)ceilf(
        fmaxf(fmaxf(half_extent.x, half_extent.y), cutoff * FILTER_SIZE)
    );

    // cull surfels that do not touch any image tile
    uint2 rect_min, rect_max;
    surfel_tile_rect(
        center, radius, num_tiles_x(image_width), num_tiles_y(image_height),
        rect_min, rect_max
    );
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) {
        return;
    }

    radii[idx] = radius;
    means2d[2 * idx] = center.x;
    means2d[2 * idx + 1] = center.y;
    depths[idx] = mean_c.z;
    float *T = ray_transforms + 9 * idx;
    T[0] = Tu.x; T[1] = Tu.y; T[2] = Tu.z;
    T[3] = Tv.x; T[4] = Tv.y; T[5] = Tv.z;
    T[6] = Tw.x; T[7] = Tw.y; T[8] = Tw.z;
    dirs[3 * idx] = R[0].x;
    dirs[3 * idx + 1] = R[0].y;
    dirs[3 * idx + 2] = R[0].z;
}

__global__ void project_surfels_bwd_kernel(
    const uint32_t N,
    const float *__restrict__ quats,          // [N, 4]
    const float *__restrict__ scales,         // [N, 2]
    const float *__restrict__ opacities,      // [N]
    const float *__restrict__ viewmat,        // [4, 4]
    const float *__restrict__ K,              // [3, 3]
    const float scale_modifier,
    const int32_t *__restrict__ radii,        // [N]
    const float *__restrict__ ray_transforms, // [N, 9]
    // gradients of the outputs
    const float *__restrict__ v_means2d,        // [N, 2]
    const float *__restrict__ v_ray_transforms, // [N, 9]
    const float *__restrict__ v_dirs,           // [N, 3]
    // gradients of the inputs
    float *__restrict__ v_means,  // [N, 3]
    float *__restrict__ v_quats,  // [N, 4]
    float *__restrict__ v_scales  // [N, 2]
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx] <= 0) {
        return;
    }

    const float *T = ray_transforms + 9 * idx;
    const vec3 Tu = vec3(T[0], T[1], T[2]);
    const vec3 Tv = vec3(T[3], T[4], T[5]);
    const vec3 Tw = vec3(T[6], T[7], T[8]);
    const float *vT = v_ray_transforms + 9 * idx;
    vec3 v_Tu = vec3(vT[0], vT[1], vT[2]);
    vec3 v_Tv = vec3(vT[3], vT[4], vT[5]);
    vec3 v_Tw = vec3(vT[6], vT[7], vT[8]);

    // gradient through the screen space center of the bounding box (only the
    // low-pass filter part of the kernel depends on it)
    const vec2 v_center = vec2(v_means2d[2 * idx], v_means2d[2 * idx + 1]);
    if (v_center.x != 0.f || v_center.y != 0.f) {
        const float cutoff = opacity_cutoff(opacities[idx]);
        const vec3 t = vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
        const float d = glm::dot(t, Tw * Tw);
        const vec3 f = t * (1.0f / d);
        v_Tu += v_center.x * f * Tw;
        v_Tv += v_center.y * f * Tw;
        v_Tw += v_center.x * f * Tu + v_center.y * f * Tv;
        const vec3 v_f = v_center.x * Tu * Tw + v_center.y * Tv * Tw;
        const float v_d = glm::dot(v_f, f) * (-1.0f / d);
        v_Tw += v_d * 2.0f * t * Tw;
    }

    // back through K: rows of W H = [axis_u | axis_v | mean_c]
    const float fx = K[0], cx = K[2], fy = K[4], cy = K[5];
    const vec3 v_row_x = fx * v_Tu;
    const vec3 v_row_y = fy * v_Tv;
    const vec3 v_row_z = cx * v_Tu + cy * v_Tv + v_Tw;
    const vec3 v_axis_u_c = vec3(v_row_x[0], v_row_y[0], v_row_z[0]);
    const vec3 v_axis_v_c = vec3(v_row_x[1], v_row_y[1], v_row_z[1]);
    const vec3 v_mean_c = vec3(v_row_x[2], v_row_y[2], v_row_z[2]);

    // camera to world
    mat3 R_cam;
    vec3 t_cam;
    load_viewmat(viewmat, R_cam, t_cam);
    const mat3 R_cam_t = glm::transpose(R_cam);
    const vec3 v_axis_u = R_cam_t * v_axis_u_c;
    const vec3 v_axis_v = R_cam_t * v_axis_v_c;
    const vec3 v_mean = R_cam_t * v_mean_c;

    // world axes: axis_u = R[0] * s_u, axis_v = R[1] * s_v; dirs = R[0]
    const vec4 quat = glm::make_vec4(quats + 4 * idx);
    const mat3 R = quat_to_rotmat(quat);
    const float s_u = scales[2 * idx] * scale_modifier;
    const float s_v = scales[2 * idx + 1] * scale_modifier;
    const vec3 v_dir = glm::make_vec3(v_dirs + 3 * idx);
    const mat3 v_R = mat3(v_axis_u * s_u + v_dir, v_axis_v * s_v, vec3(0.f));
    const vec4 v_quat = quat_to_rotmat_vjp(quat, v_R);

    v_means[3 * idx] = v_mean.x;
    v_means[3 * idx + 1] = v_mean.y;
    v_means[3 * idx + 2] = v_mean.z;
    v_quats[4 * idx] = v_quat[0];
    v_quats[4 * idx + 1] = v_quat[1];
    v_quats[4 * idx + 2] = v_quat[2];
    v_quats[4 * idx + 3] = v_quat[3];
    v_scales[2 * idx] = glm::dot(v_axis_u, R[0]) * scale_modifier;
    v_scales[2 * idx + 1] = glm::dot(v_axis_v, R[1]) * scale_modifier;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_surfels_fwd(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &viewmat,
    const at::Tensor &K,
    const int64_t image_width,
    const int64_t image_height,
    const double scale_modifier
) {
    SS_CHECK_FLOAT(means);
    SS_CHECK_FLOAT(quats);
    SS_CHECK_FLOAT(scales);
    SS_CHECK_FLOAT(opacities);
    SS_CHECK_FLOAT(viewmat);
    SS_CHECK_FLOAT(K);
    const at::cuda::OptionalCUDAGuard device_guard(means.device());

    const uint32_t N = means.size(0);
    auto opts = means.options();
    at::Tensor radii = at::zeros({N}, opts.dtype(at::kInt));
    at::Tensor means2d = at::zeros({N, 2}, opts);
    at::Tensor depths = at::zeros({N}, opts);
    at::Tensor ray_transforms = at::zeros({N, 9}, opts);
    at::Tensor dirs = at::zeros({N, 3}, opts);
    if (N == 0) {
        return {radii, means2d, depths, ray_transforms, dirs};
    }

    project_surfels_fwd_kernel<<<(N + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        N,
        means.data_ptr<float>(),
        quats.data_ptr<float>(),
        scales.data_ptr<float>(),
        opacities.data_ptr<float>(),
        viewmat.data_ptr<float>(),
        K.data_ptr<float>(),
        (uint32_t)image_width,
        (uint32_t)image_height,
        (float)scale_modifier,
        radii.data_ptr<int32_t>(),
        means2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        ray_transforms.data_ptr<float>(),
        dirs.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {radii, means2d, depths, ray_transforms, dirs};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> project_surfels_bwd(
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &viewmat,
    const at::Tensor &K,
    const double scale_modifier,
    const at::Tensor &radii,
    const at::Tensor &ray_transforms,
    const at::Tensor &v_means2d,
    const at::Tensor &v_ray_transforms,
    const at::Tensor &v_dirs
) {
    SS_CHECK_FLOAT(quats);
    SS_CHECK_FLOAT(scales);
    SS_CHECK_FLOAT(opacities);
    SS_CHECK_FLOAT(viewmat);
    SS_CHECK_FLOAT(K);
    SS_CHECK_INPUT(radii);
    SS_CHECK_FLOAT(ray_transforms);
    SS_CHECK_FLOAT(v_means2d);
    SS_CHECK_FLOAT(v_ray_transforms);
    SS_CHECK_FLOAT(v_dirs);
    const at::cuda::OptionalCUDAGuard device_guard(quats.device());

    const uint32_t N = quats.size(0);
    auto opts = quats.options();
    at::Tensor v_means = at::zeros({N, 3}, opts);
    at::Tensor v_quats = at::zeros({N, 4}, opts);
    at::Tensor v_scales = at::zeros({N, 2}, opts);
    if (N == 0) {
        return {v_means, v_quats, v_scales};
    }

    project_surfels_bwd_kernel<<<(N + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        N,
        quats.data_ptr<float>(),
        scales.data_ptr<float>(),
        opacities.data_ptr<float>(),
        viewmat.data_ptr<float>(),
        K.data_ptr<float>(),
        (float)scale_modifier,
        radii.data_ptr<int32_t>(),
        ray_transforms.data_ptr<float>(),
        v_means2d.data_ptr<float>(),
        v_ray_transforms.data_ptr<float>(),
        v_dirs.data_ptr<float>(),
        v_means.data_ptr<float>(),
        v_quats.data_ptr<float>(),
        v_scales.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means, v_quats, v_scales};
}

} // namespace strand_surfels
