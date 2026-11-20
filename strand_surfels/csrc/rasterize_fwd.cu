// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// Forward pass of the strand surfel rasterizer.
//
// The tile based rasterization loop is derived from the 2DGS forward pass of
// gsplat (https://github.com/nerfstudio-project/gsplat, Apache-2.0,
// gsplat/cuda/csrc/RasterizeToPixels2DGSFwd.cu).
// The per-bucket sampling of the blending state, which enables the bucketed
// backward pass (see rasterize_bwd.cu), follows Taming 3DGS
// (https://github.com/humansensinglab/taming-3dgs, MIT, Copyright (c) 2024
// The Authors).
// Changes with respect to gsplat: blending state sampling every BUCKET_SIZE
// surfels, rendering of the strand direction, normalised-depth distortion,
// opacity cut-off and blending limits of the legacy rasterizer.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "common.h"
#include "ops.h"

namespace strand_surfels {

namespace cg = cooperative_groups;

// One thread block per image tile, one thread per pixel. The surfels of a tile
// are processed front to back in batches that are loaded cooperatively into
// shared memory.
template <bool SAVE_SAMPLES>
__global__ void __launch_bounds__(TILE_PIXELS) rasterize_fwd_kernel(
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tiles_x,
    const int32_t *__restrict__ tile_ranges,    // [n_tiles, 2]
    const int32_t *__restrict__ flatten_ids,    // [n_isects]
    const int32_t *__restrict__ bucket_offsets, // [n_tiles] inclusive prefix sum
    const float *__restrict__ means2d,          // [N, 2]
    const float *__restrict__ ray_transforms,   // [N, 9]
    const float *__restrict__ colors,           // [N, COLOR_DIM]
    const float *__restrict__ dirs,             // [N, 3]
    const float *__restrict__ opacities,        // [N]
    const float *__restrict__ background,       // [COLOR_DIM]
    // outputs
    float *__restrict__ render_colors,   // [COLOR_DIM, H, W]
    float *__restrict__ render_aux,      // [AUX_DIM, H, W]
    float *__restrict__ final_state,     // [FINAL_STATE_DIM, H, W]
    int32_t *__restrict__ contributors,  // [CONTRIB_DIM, H, W]
    int32_t *__restrict__ max_contrib,   // [n_tiles]
    // blending state samples (SAVE_SAMPLES only)
    int32_t *__restrict__ bucket_to_tile, // [n_buckets]
    float *__restrict__ sampled_state,    // [n_buckets, SAMPLED_STATE_DIM, TILE_PIXELS]
    float *__restrict__ sampled_color,    // [n_buckets, COLOR_DIM, TILE_PIXELS]
    float *__restrict__ sampled_aux       // [n_buckets, SAMPLED_AUX_DIM, TILE_PIXELS]
) {
    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tiles_x + block.group_index().x;
    const uint32_t i = block.group_index().y * TILE_SIZE + block.thread_index().y;
    const uint32_t j = block.group_index().x * TILE_SIZE + block.thread_index().x;
    const uint32_t tr = block.thread_rank();
    const uint32_t HW = image_height * image_width;
    const uint32_t pix_id = i * image_width + j;
    // pixel center
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;

    const bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    const uint32_t range_start = tile_ranges[2 * tile_id];
    const uint32_t range_end = tile_ranges[2 * tile_id + 1];
    const uint32_t n_surfels = range_end - range_start;
    const uint32_t n_batches = (n_surfels + TILE_PIXELS - 1) / TILE_PIXELS;

    // first bucket of this tile; also record the bucket -> tile mapping
    uint32_t bucket = tile_id == 0 ? 0 : bucket_offsets[tile_id - 1];
    if (SAVE_SAMPLES) {
        const uint32_t n_tile_buckets = (n_surfels + BUCKET_SIZE - 1) / BUCKET_SIZE;
        for (uint32_t b = tr; b < n_tile_buckets; b += TILE_PIXELS) {
            bucket_to_tile[bucket + b] = tile_id;
        }
    }

    __shared__ int32_t id_batch[TILE_PIXELS];
    __shared__ vec3 xy_opacity_batch[TILE_PIXELS];
    __shared__ vec3 u_batch[TILE_PIXELS];
    __shared__ vec3 v_batch[TILE_PIXELS];
    __shared__ vec3 w_batch[TILE_PIXELS];

    // blending state of this pixel
    float T = 1.0f;                     // transmittance
    float color[COLOR_DIM] = {0.f};     // accumulated color
    float dir[3] = {0.f};               // accumulated strand direction
    float depth_sum = 0.f;              // accumulated depth
    float m1 = 0.f;                     // sum of w * m (normalised depth m)
    float m2 = 0.f;                     // sum of w * m^2
    float distortion = 0.f;
    float median_depth = 0.f;
    int32_t contributor = 0;            // position of the current surfel in the tile list (1-based)
    int32_t last_contributor = 0;       // position of the last blended surfel
    int32_t median_contributor = -1;    // position of the median depth surfel

    for (uint32_t b = 0; b < n_batches; ++b) {
        // stop when every pixel of the tile is saturated
        if (__syncthreads_count(done) == TILE_PIXELS) {
            break;
        }

        // cooperatively load the next batch of surfels
        const uint32_t batch_start = range_start + TILE_PIXELS * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xy_opacity_batch[tr] = vec3(means2d[2 * g], means2d[2 * g + 1], opacities[g]);
            const float *M = ray_transforms + 9 * g;
            u_batch[tr] = vec3(M[0], M[1], M[2]);
            v_batch[tr] = vec3(M[3], M[4], M[5]);
            w_batch[tr] = vec3(M[6], M[7], M[8]);
        }
        block.sync();

        const uint32_t batch_size = min(TILE_PIXELS, range_end - batch_start);
        for (uint32_t t = 0; !done && t < batch_size; ++t) {
            // sample the incoming blending state at the start of each bucket
            if (SAVE_SAMPLES && t % BUCKET_SIZE == 0) {
                float *state = sampled_state + bucket * SAMPLED_STATE_DIM * TILE_PIXELS;
                state[tr] = T;
                state[TILE_PIXELS + tr] = m1;
                state[2 * TILE_PIXELS + tr] = m2;
                float *col = sampled_color + bucket * COLOR_DIM * TILE_PIXELS;
#pragma unroll
                for (uint32_t c = 0; c < COLOR_DIM; ++c) {
                    col[c * TILE_PIXELS + tr] = color[c];
                }
                float *aux = sampled_aux + bucket * SAMPLED_AUX_DIM * TILE_PIXELS;
                aux[tr] = depth_sum;
#pragma unroll
                for (uint32_t c = 0; c < 3; ++c) {
                    aux[(1 + c) * TILE_PIXELS + tr] = dir[c];
                }
                ++bucket;
            }
            ++contributor;

            // ray-surfel intersection in surfel coordinates (u, v): the pixel
            // defines two planes h_u, h_v whose intersection line meets the
            // surfel plane at (s.x, s.y)
            const vec3 Tu = u_batch[t];
            const vec3 Tv = v_batch[t];
            const vec3 Tw = w_batch[t];
            const vec3 h_u = px * Tw - Tu;
            const vec3 h_v = py * Tw - Tv;
            const vec3 p = glm::cross(h_u, h_v);
            if (p.z == 0.0f) {
                continue;
            }
            const vec2 s = vec2(p.x / p.z, p.y / p.z);
            const float rho3d = s.x * s.x + s.y * s.y;
            // screen space low-pass filter
            const vec3 xy_opac = xy_opacity_batch[t];
            const vec2 d = vec2(xy_opac.x - px, xy_opac.y - py);
            const float rho2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
            const float rho = fminf(rho3d, rho2d);

            // camera depth of the intersection point
            const float depth = s.x * Tw.x + s.y * Tw.y + Tw.z;
            if (depth < NEAR_PLANE) {
                continue;
            }
            const float power = -0.5f * rho;
            if (power > 0.0f) {
                continue;
            }
            const float alpha = fminf(MAX_ALPHA, xy_opac.z * expf(power));
            if (alpha < MIN_ALPHA) {
                continue;
            }
            const float next_T = T * (1.0f - alpha);
            if (next_T < MIN_TRANSMITTANCE) {
                done = true;
                continue;
            }

            const float w = alpha * T;
            const int32_t g = id_batch[t];

            // depth distortion (2DGS, Eq. 13) on the normalised depth
            const float m = ndc_depth(depth);
            distortion += (m * m * (1.0f - T) + m2 - 2.0f * m * m1) * w;
            depth_sum += depth * w;
            m1 += m * w;
            m2 += m * m * w;
            if (T > 0.5f) {
                median_depth = depth;
                median_contributor = contributor;
            }
#pragma unroll
            for (uint32_t c = 0; c < 3; ++c) {
                dir[c] += dirs[3 * g + c] * w;
            }
#pragma unroll
            for (uint32_t c = 0; c < COLOR_DIM; ++c) {
                color[c] += colors[COLOR_DIM * g + c] * w;
            }

            T = next_T;
            last_contributor = contributor;
        }
    }

    if (inside) {
#pragma unroll
        for (uint32_t c = 0; c < COLOR_DIM; ++c) {
            render_colors[c * HW + pix_id] = color[c] + T * background[c];
        }
        render_aux[AUX_DEPTH * HW + pix_id] = depth_sum;
        render_aux[AUX_ALPHA * HW + pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            render_aux[(AUX_DIR + c) * HW + pix_id] = dir[c];
        }
        render_aux[AUX_DISTORTION * HW + pix_id] = distortion;
        render_aux[AUX_MEDIAN * HW + pix_id] = median_depth;

        final_state[pix_id] = T;
        final_state[HW + pix_id] = m1;
        final_state[2 * HW + pix_id] = m2;
        contributors[pix_id] = last_contributor;
        contributors[HW + pix_id] = median_contributor;
    }

    // largest number of surfels any pixel of the tile needed; the backward
    // pass skips buckets beyond it
    typedef cub::BlockReduce<int32_t, TILE_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS, TILE_SIZE>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int32_t tile_max = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
    if (tr == 0) {
        max_contrib[tile_id] = tile_max;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rasterize_fwd(
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
    const bool save_samples,
    const at::Tensor &bucket_to_tile,
    const at::Tensor &sampled_state,
    const at::Tensor &sampled_color,
    const at::Tensor &sampled_aux
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
    TORCH_CHECK(colors.size(-1) == COLOR_DIM, "colors must have 3 channels");
    if (save_samples) {
        SS_CHECK_INPUT(bucket_to_tile);
        SS_CHECK_FLOAT(sampled_state);
        SS_CHECK_FLOAT(sampled_color);
        SS_CHECK_FLOAT(sampled_aux);
    }
    const at::cuda::OptionalCUDAGuard device_guard(means2d.device());

    const uint32_t W = image_width, H = image_height;
    const uint32_t tiles_x = num_tiles_x(W), tiles_y = num_tiles_y(H);
    auto opts = means2d.options();
    at::Tensor render_colors = at::empty({COLOR_DIM, H, W}, opts);
    at::Tensor render_aux = at::empty({AUX_DIM, H, W}, opts);
    at::Tensor final_state = at::empty({FINAL_STATE_DIM, H, W}, opts);
    at::Tensor contributors = at::empty({CONTRIB_DIM, H, W}, opts.dtype(at::kInt));
    at::Tensor max_contrib = at::empty({tiles_x * tiles_y}, opts.dtype(at::kInt));

    const dim3 grid(tiles_x, tiles_y);
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();
#define SS_LAUNCH_FWD(SAVE)                                                    \
    rasterize_fwd_kernel<SAVE><<<grid, threads, 0, stream>>>(                  \
        W, H, tiles_x,                                                         \
        tile_ranges.data_ptr<int32_t>(),                                       \
        flatten_ids.data_ptr<int32_t>(),                                       \
        bucket_offsets.data_ptr<int32_t>(),                                    \
        means2d.data_ptr<float>(),                                             \
        ray_transforms.data_ptr<float>(),                                      \
        colors.data_ptr<float>(),                                              \
        dirs.data_ptr<float>(),                                                \
        opacities.data_ptr<float>(),                                           \
        background.data_ptr<float>(),                                          \
        render_colors.data_ptr<float>(),                                       \
        render_aux.data_ptr<float>(),                                          \
        final_state.data_ptr<float>(),                                         \
        contributors.data_ptr<int32_t>(),                                      \
        max_contrib.data_ptr<int32_t>(),                                       \
        SAVE ? bucket_to_tile.data_ptr<int32_t>() : nullptr,                   \
        SAVE ? sampled_state.data_ptr<float>() : nullptr,                      \
        SAVE ? sampled_color.data_ptr<float>() : nullptr,                      \
        SAVE ? sampled_aux.data_ptr<float>() : nullptr                         \
    )
    if (save_samples) {
        SS_LAUNCH_FWD(true);
    } else {
        SS_LAUNCH_FWD(false);
    }
#undef SS_LAUNCH_FWD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {render_colors, render_aux, final_state, contributors, max_contrib};
}

} // namespace strand_surfels
