// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// Backward pass of the strand surfel rasterizer, parallelised per bucket.
//
// Instead of replaying the blending of every pixel sequentially, the surfels of
// each tile are split into buckets of BUCKET_SIZE consecutive surfels and every
// bucket is handled by one warp: lane k owns the k-th surfel of the bucket, and
// the 256 pixels of the tile stream through the warp, lane k processing pixel
// (i - k) at step i. The blending state a pixel had when it reached the bucket
// was sampled by the forward pass; lanes hand the updated state to the next
// lane with warp shuffles. This scheme is the per-Gaussian backward pass of
// Taming 3DGS (https://github.com/humansensinglab/taming-3dgs, MIT,
// Copyright (c) 2024 The Authors).
// The per-pixel gradient of the ray-surfel intersection follows the 2DGS
// backward pass of gsplat (https://github.com/nerfstudio-project/gsplat,
// Apache-2.0, gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu), extended by the
// strand direction, normalised-depth distortion, median depth and per-surfel
// error terms of this project.

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "common.h"
#include "ops.h"

namespace strand_surfels {

namespace cg = cooperative_groups;

__global__ void __launch_bounds__(BUCKET_SIZE) rasterize_bwd_kernel(
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tiles_x,
    const uint32_t n_buckets,
    const int32_t *__restrict__ tile_ranges,    // [n_tiles, 2]
    const int32_t *__restrict__ flatten_ids,    // [n_isects]
    const int32_t *__restrict__ bucket_offsets, // [n_tiles] inclusive prefix sum
    const int32_t *__restrict__ bucket_to_tile, // [n_buckets]
    const int32_t *__restrict__ max_contrib,    // [n_tiles]
    const float *__restrict__ sampled_state,    // [n_buckets, 3, TILE_PIXELS]
    const float *__restrict__ sampled_color,    // [n_buckets, COLOR_DIM, TILE_PIXELS]
    const float *__restrict__ sampled_aux,      // [n_buckets, 4, TILE_PIXELS]
    const float *__restrict__ means2d,          // [N, 2]
    const float *__restrict__ ray_transforms,   // [N, 9]
    const float *__restrict__ colors,           // [N, COLOR_DIM]
    const float *__restrict__ dirs,             // [N, 3]
    const float *__restrict__ opacities,        // [N]
    const float *__restrict__ background,       // [COLOR_DIM]
    const float *__restrict__ render_colors,    // [COLOR_DIM, H, W]
    const float *__restrict__ render_aux,       // [AUX_DIM, H, W]
    const float *__restrict__ final_state,      // [FINAL_STATE_DIM, H, W]
    const int32_t *__restrict__ contributors,   // [CONTRIB_DIM, H, W]
    // gradients of the outputs
    const float *__restrict__ v_render_colors, // [COLOR_DIM, H, W]
    const float *__restrict__ v_render_aux,    // [AUX_DIM, H, W]
    const float *__restrict__ v_error_map,     // [1, H, W]
    // gradients of the inputs (accumulated atomically)
    float *__restrict__ v_means2d,        // [N, 2]
    float *__restrict__ v_ray_transforms, // [N, 9]
    float *__restrict__ v_colors,         // [N, COLOR_DIM]
    float *__restrict__ v_dirs,           // [N, 3]
    float *__restrict__ v_opacities,      // [N]
    float *__restrict__ v_pse             // [N]
) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<BUCKET_SIZE>(block);
    const uint32_t bucket = block.group_index().x;
    if (bucket >= n_buckets) {
        return;
    }
    const uint32_t lane = warp.thread_rank();
    const uint32_t HW = image_height * image_width;

    // locate the bucket and this lane's surfel in the tile list
    const uint32_t tile_id = bucket_to_tile[bucket];
    const int32_t range_start = tile_ranges[2 * tile_id];
    const int32_t n_surfels = tile_ranges[2 * tile_id + 1] - range_start;
    const uint32_t first_bucket = tile_id == 0 ? 0 : bucket_offsets[tile_id - 1];
    const int32_t bucket_in_tile = bucket - first_bucket;
    const int32_t surfel_in_tile = bucket_in_tile * BUCKET_SIZE + lane;
    const bool valid_surfel = surfel_in_tile < n_surfels;

    // no pixel of the tile blends any surfel of this bucket
    if (bucket_in_tile * (int32_t)BUCKET_SIZE >= max_contrib[tile_id]) {
        return;
    }

    // surfel data in registers
    int32_t g = 0;
    vec2 xy(0.f);
    float opacity = 0.f;
    vec3 Tu(0.f), Tv(0.f), Tw(0.f);
    float color[COLOR_DIM] = {0.f};
    float dir[3] = {0.f};
    if (valid_surfel) {
        g = flatten_ids[range_start + surfel_in_tile];
        xy = vec2(means2d[2 * g], means2d[2 * g + 1]);
        opacity = opacities[g];
        const float *M = ray_transforms + 9 * g;
        Tu = vec3(M[0], M[1], M[2]);
        Tv = vec3(M[3], M[4], M[5]);
        Tw = vec3(M[6], M[7], M[8]);
#pragma unroll
        for (uint32_t c = 0; c < COLOR_DIM; ++c) {
            color[c] = colors[COLOR_DIM * g + c];
        }
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            dir[c] = dirs[3 * g + c];
        }
    }

    // gradient accumulators of this lane's surfel
    vec2 v_xy_acc(0.f);
    vec3 v_Tu_acc(0.f), v_Tv_acc(0.f), v_Tw_acc(0.f);
    float v_color_acc[COLOR_DIM] = {0.f};
    float v_dir_acc[3] = {0.f};
    float v_opacity_acc = 0.f;
    float v_pse_acc = 0.f;

    // per-pixel state handed from lane to lane. The "rest" quantities hold the
    // (negated) contribution of all surfels behind the current one:
    // rest = accumulated up to and including the current surfel - final value
    float T = 0.f;                 // transmittance in front of the current surfel
    float T_final = 0.f;           // transmittance after all surfels
    int32_t last_contrib = 0;      // tile position of the last blended surfel
    int32_t median_contrib = -1;   // tile position of the median depth surfel
    float m1_final = 0.f, m2_final = 0.f; // sum(w m) and sum(w m^2) of all surfels
    float rest_color[COLOR_DIM] = {0.f};
    float rest_depth = 0.f, rest_alpha = 0.f, rest_distortion = 0.f;
    float rest_dir[3] = {0.f};
    float v_pix_color[COLOR_DIM] = {0.f};
    float v_pix_depth = 0.f, v_pix_alpha = 0.f, v_pix_distortion = 0.f, v_pix_median = 0.f;
    float v_pix_dir[3] = {0.f};
    float v_pix_error = 0.f;

    const uint32_t tile_x0 = (tile_id % tiles_x) * TILE_SIZE;
    const uint32_t tile_y0 = (tile_id / tiles_x) * TILE_SIZE;

    for (uint32_t step = 0; step < TILE_PIXELS + BUCKET_SIZE - 1; ++step) {
        // hand the pixel state over to the next lane
        T = warp.shfl_up(T, 1);
        T_final = warp.shfl_up(T_final, 1);
        last_contrib = warp.shfl_up(last_contrib, 1);
        median_contrib = warp.shfl_up(median_contrib, 1);
        m1_final = warp.shfl_up(m1_final, 1);
        m2_final = warp.shfl_up(m2_final, 1);
#pragma unroll
        for (uint32_t c = 0; c < COLOR_DIM; ++c) {
            rest_color[c] = warp.shfl_up(rest_color[c], 1);
            v_pix_color[c] = warp.shfl_up(v_pix_color[c], 1);
        }
        rest_depth = warp.shfl_up(rest_depth, 1);
        rest_alpha = warp.shfl_up(rest_alpha, 1);
        rest_distortion = warp.shfl_up(rest_distortion, 1);
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            rest_dir[c] = warp.shfl_up(rest_dir[c], 1);
            v_pix_dir[c] = warp.shfl_up(v_pix_dir[c], 1);
        }
        v_pix_depth = warp.shfl_up(v_pix_depth, 1);
        v_pix_alpha = warp.shfl_up(v_pix_alpha, 1);
        v_pix_distortion = warp.shfl_up(v_pix_distortion, 1);
        v_pix_median = warp.shfl_up(v_pix_median, 1);
        v_pix_error = warp.shfl_up(v_pix_error, 1);

        // the pixel this lane works on in this step
        const int32_t idx = (int32_t)step - (int32_t)lane;
        const uint32_t px_i = tile_x0 + idx % TILE_SIZE;
        const uint32_t py_i = tile_y0 + idx / TILE_SIZE;
        const uint32_t pix_id = py_i * image_width + px_i;
        const bool valid_pixel = px_i < image_width && py_i < image_height;

        // the first lane loads the state the pixel had when reaching the bucket
        if (valid_surfel && valid_pixel && lane == 0 && idx < (int32_t)TILE_PIXELS) {
            const float *state = sampled_state + bucket * SAMPLED_STATE_DIM * TILE_PIXELS;
            T = state[idx];
            const float m1 = state[TILE_PIXELS + idx];
            const float m2 = state[2 * TILE_PIXELS + idx];
            T_final = final_state[pix_id];
            m1_final = final_state[HW + pix_id];
            m2_final = final_state[2 * HW + pix_id];
            last_contrib = contributors[pix_id];
            median_contrib = contributors[HW + pix_id];

            const float *col = sampled_color + bucket * COLOR_DIM * TILE_PIXELS;
#pragma unroll
            for (uint32_t c = 0; c < COLOR_DIM; ++c) {
                rest_color[c] = col[c * TILE_PIXELS + idx] -
                                (render_colors[c * HW + pix_id] - T_final * background[c]);
                v_pix_color[c] = v_render_colors[c * HW + pix_id];
            }
            const float *aux = sampled_aux + bucket * SAMPLED_AUX_DIM * TILE_PIXELS;
            rest_depth = aux[idx] - render_aux[AUX_DEPTH * HW + pix_id];
            rest_alpha = T_final - T;
#pragma unroll
            for (uint32_t c = 0; c < 3; ++c) {
                rest_dir[c] = aux[(1 + c) * TILE_PIXELS + idx] -
                              render_aux[(AUX_DIR + c) * HW + pix_id];
                v_pix_dir[c] = v_render_aux[(AUX_DIR + c) * HW + pix_id];
            }
            // distortion of the surfels in front of the bucket minus the total
            const float alpha_final = 1.f - T_final;
            rest_distortion = alpha_final * (m2 - m2_final) + m2_final * (T_final - T) -
                              2.f * m1_final * (m1 - m1_final);
            v_pix_depth = v_render_aux[AUX_DEPTH * HW + pix_id];
            v_pix_alpha = v_render_aux[AUX_ALPHA * HW + pix_id];
            v_pix_distortion = v_render_aux[AUX_DISTORTION * HW + pix_id];
            v_pix_median = v_render_aux[AUX_MEDIAN * HW + pix_id];
            v_pix_error = v_error_map[pix_id];
        }

        if (!valid_surfel || !valid_pixel || idx < 0 || idx >= (int32_t)TILE_PIXELS) {
            continue;
        }
        // surfels behind the last blended one did not contribute to this pixel
        if (surfel_in_tile >= last_contrib) {
            continue;
        }

        // recompute the forward quantities (same tests as the forward pass)
        const float px = (float)px_i + 0.5f;
        const float py = (float)py_i + 0.5f;
        const vec3 h_u = px * Tw - Tu;
        const vec3 h_v = py * Tw - Tv;
        const vec3 p = glm::cross(h_u, h_v);
        if (p.z == 0.0f) {
            continue;
        }
        const vec2 s = vec2(p.x / p.z, p.y / p.z);
        const float rho3d = s.x * s.x + s.y * s.y;
        const vec2 d = vec2(xy.x - px, xy.y - py);
        const float rho2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
        const float rho = fminf(rho3d, rho2d);
        const float depth = s.x * Tw.x + s.y * Tw.y + Tw.z;
        if (depth < NEAR_PLANE) {
            continue;
        }
        const float power = -0.5f * rho;
        if (power > 0.0f) {
            continue;
        }
        const float G = expf(power);
        const float alpha = fminf(MAX_ALPHA, opacity * G);
        if (alpha < MIN_ALPHA) {
            continue;
        }

        const float w = alpha * T;
        const float inv_one_minus_alpha = 1.0f / (1.0f - alpha);
        // d(pixel value)/d(alpha) = value * T - (contribution behind) / (1 - alpha)
        float v_alpha = 0.f;

        // color
        float bg_dot_v = 0.f;
#pragma unroll
        for (uint32_t c = 0; c < COLOR_DIM; ++c) {
            rest_color[c] += w * color[c];
            v_color_acc[c] += w * v_pix_color[c];
            v_alpha += (color[c] * T + inv_one_minus_alpha * rest_color[c]) * v_pix_color[c];
            bg_dot_v += background[c] * v_pix_color[c];
        }
        v_alpha += -T_final * inv_one_minus_alpha * bg_dot_v;

        // per-surfel error: blending weight times the error gradient
        v_pse_acc += w * v_pix_error;

        // depth distortion
        float v_depth = 0.f;
        const float m = ndc_depth(depth);
        const float alpha_final = 1.f - T_final;
        if (surfel_in_tile == median_contrib - 1) {
            v_depth += v_pix_median;
        }
        const float e = m2_final + m * m * alpha_final - 2.f * m * m1_final;
        rest_distortion += w * e;
        v_alpha += (e * T + inv_one_minus_alpha * rest_distortion) * v_pix_distortion;
        v_depth += 2.f * w * (m * alpha_final - m1_final) * v_pix_distortion * ndc_depth_grad(depth);

        // depth
        rest_depth += w * depth;
        v_alpha += (depth * T + inv_one_minus_alpha * rest_depth) * v_pix_depth;
        v_depth += w * v_pix_depth;

        // accumulated opacity
        rest_alpha += w;
        v_alpha += (T + inv_one_minus_alpha * rest_alpha) * v_pix_alpha;

        // strand direction
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            v_dir_acc[c] += w * v_pix_dir[c];
            rest_dir[c] += w * dir[c];
            v_alpha += (dir[c] * T + inv_one_minus_alpha * rest_dir[c]) * v_pix_dir[c];
        }

        // back to the surfel parameters. alpha = opacity * G with
        // G = exp(-rho / 2), rho = min(rho3d, rho2d); the depth depends on the
        // intersection point s in both cases.
        const float v_G = opacity * v_alpha;
        vec2 v_s = vec2(v_depth * Tw.x, v_depth * Tw.y);
        if (rho3d <= rho2d) {
            v_s += vec2(v_G * -G * s.x, v_G * -G * s.y);
        } else {
            v_xy_acc += v_G * -G * FILTER_INV_SQUARE * d;
        }
        // s = (p.x / p.z, p.y / p.z), p = h_u x h_v
        const float v_sx_pz = v_s.x / p.z;
        const float v_sy_pz = v_s.y / p.z;
        const vec3 v_p = vec3(v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y));
        const vec3 v_h_u = glm::cross(h_v, v_p);
        const vec3 v_h_v = glm::cross(v_p, h_u);
        v_Tu_acc -= v_h_u;
        v_Tv_acc -= v_h_v;
        v_Tw_acc += px * v_h_u + py * v_h_v + v_depth * vec3(s.x, s.y, 1.f);

        v_opacity_acc += G * v_alpha;

        T *= 1.0f - alpha;
    }

    if (valid_surfel) {
        gpuAtomicAdd(v_means2d + 2 * g, v_xy_acc.x);
        gpuAtomicAdd(v_means2d + 2 * g + 1, v_xy_acc.y);
        float *vT = v_ray_transforms + 9 * g;
        gpuAtomicAdd(vT + 0, v_Tu_acc.x);
        gpuAtomicAdd(vT + 1, v_Tu_acc.y);
        gpuAtomicAdd(vT + 2, v_Tu_acc.z);
        gpuAtomicAdd(vT + 3, v_Tv_acc.x);
        gpuAtomicAdd(vT + 4, v_Tv_acc.y);
        gpuAtomicAdd(vT + 5, v_Tv_acc.z);
        gpuAtomicAdd(vT + 6, v_Tw_acc.x);
        gpuAtomicAdd(vT + 7, v_Tw_acc.y);
        gpuAtomicAdd(vT + 8, v_Tw_acc.z);
#pragma unroll
        for (uint32_t c = 0; c < COLOR_DIM; ++c) {
            gpuAtomicAdd(v_colors + COLOR_DIM * g + c, v_color_acc[c]);
        }
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            gpuAtomicAdd(v_dirs + 3 * g + c, v_dir_acc[c]);
        }
        gpuAtomicAdd(v_opacities + g, v_opacity_acc);
        gpuAtomicAdd(v_pse + g, v_pse_acc);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rasterize_bwd(
    const at::Tensor &means2d,
    const at::Tensor &ray_transforms,
    const at::Tensor &colors,
    const at::Tensor &dirs,
    const at::Tensor &opacities,
    const at::Tensor &background,
    const int64_t image_width,
    const int64_t image_height,
    const at::Tensor &tile_ranges,
    const at::Tensor &flatten_ids,
    const at::Tensor &bucket_offsets,
    const at::Tensor &bucket_to_tile,
    const at::Tensor &max_contrib,
    const at::Tensor &sampled_state,
    const at::Tensor &sampled_color,
    const at::Tensor &sampled_aux,
    const int64_t n_buckets,
    const at::Tensor &render_colors,
    const at::Tensor &render_aux,
    const at::Tensor &final_state,
    const at::Tensor &contributors,
    const at::Tensor &v_render_colors,
    const at::Tensor &v_render_aux,
    const at::Tensor &v_error_map
) {
    SS_CHECK_FLOAT(means2d);
    SS_CHECK_FLOAT(ray_transforms);
    SS_CHECK_FLOAT(colors);
    SS_CHECK_FLOAT(dirs);
    SS_CHECK_FLOAT(opacities);
    SS_CHECK_FLOAT(background);
    SS_CHECK_INPUT(tile_ranges);
    SS_CHECK_INPUT(flatten_ids);
    SS_CHECK_INPUT(bucket_offsets);
    SS_CHECK_INPUT(bucket_to_tile);
    SS_CHECK_INPUT(max_contrib);
    SS_CHECK_FLOAT(sampled_state);
    SS_CHECK_FLOAT(sampled_color);
    SS_CHECK_FLOAT(sampled_aux);
    SS_CHECK_FLOAT(render_colors);
    SS_CHECK_FLOAT(render_aux);
    SS_CHECK_FLOAT(final_state);
    SS_CHECK_INPUT(contributors);
    SS_CHECK_FLOAT(v_render_colors);
    SS_CHECK_FLOAT(v_render_aux);
    SS_CHECK_FLOAT(v_error_map);
    const at::cuda::OptionalCUDAGuard device_guard(means2d.device());

    const uint32_t N = means2d.size(0);
    auto opts = means2d.options();
    at::Tensor v_means2d = at::zeros({N, 2}, opts);
    at::Tensor v_ray_transforms = at::zeros({N, 9}, opts);
    at::Tensor v_colors = at::zeros({N, COLOR_DIM}, opts);
    at::Tensor v_dirs = at::zeros({N, 3}, opts);
    at::Tensor v_opacities = at::zeros({N}, opts);
    at::Tensor v_pse = at::zeros({N}, opts);
    if (N == 0 || n_buckets == 0) {
        return {v_means2d, v_ray_transforms, v_colors, v_dirs, v_opacities, v_pse};
    }

    rasterize_bwd_kernel<<<(uint32_t)n_buckets, BUCKET_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
        (uint32_t)image_width,
        (uint32_t)image_height,
        num_tiles_x(image_width),
        (uint32_t)n_buckets,
        tile_ranges.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        bucket_offsets.data_ptr<int32_t>(),
        bucket_to_tile.data_ptr<int32_t>(),
        max_contrib.data_ptr<int32_t>(),
        sampled_state.data_ptr<float>(),
        sampled_color.data_ptr<float>(),
        sampled_aux.data_ptr<float>(),
        means2d.data_ptr<float>(),
        ray_transforms.data_ptr<float>(),
        colors.data_ptr<float>(),
        dirs.data_ptr<float>(),
        opacities.data_ptr<float>(),
        background.data_ptr<float>(),
        render_colors.data_ptr<float>(),
        render_aux.data_ptr<float>(),
        final_state.data_ptr<float>(),
        contributors.data_ptr<int32_t>(),
        v_render_colors.data_ptr<float>(),
        v_render_aux.data_ptr<float>(),
        v_error_map.data_ptr<float>(),
        v_means2d.data_ptr<float>(),
        v_ray_transforms.data_ptr<float>(),
        v_colors.data_ptr<float>(),
        v_dirs.data_ptr<float>(),
        v_opacities.data_ptr<float>(),
        v_pse.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_ray_transforms, v_colors, v_dirs, v_opacities, v_pse};
}

} // namespace strand_surfels
