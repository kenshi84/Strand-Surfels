// SPDX-License-Identifier: Apache-2.0
//
// Assignment of projected surfels to image tiles and depth sorting.
//
// Derived from the tile intersection of gsplat
// (https://github.com/nerfstudio-project/gsplat, Apache-2.0,
// gsplat/cuda/csrc/IntersectTile.cu).
// Changes with respect to gsplat:
//   * single image, square surfel radii and the tile rectangle of
//     `surfel_tile_rect`
//   * per-tile [start, end) ranges instead of offsets
//   * per-tile bucket counts for the bucketed backward pass

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "common.h"
#include "ops.h"

namespace strand_surfels {

namespace cg = cooperative_groups;

// First pass: number of tiles touched by each surfel.
// Second pass (cum_tiles != nullptr): write one (key, surfel id) pair per
// touched tile; key = | tile id (upper 32 bits) | depth (lower 32 bits) |.
// Depths are positive, so their float bit patterns sort like the values.
__global__ void intersect_tiles_kernel(
    const uint32_t N,
    const float *__restrict__ means2d, // [N, 2]
    const int32_t *__restrict__ radii, // [N]
    const float *__restrict__ depths,  // [N]
    const int64_t *__restrict__ cum_tiles, // [N] inclusive prefix sum, second pass only
    const uint32_t tiles_x,
    const uint32_t tiles_y,
    int32_t *__restrict__ tiles_per_surfel, // [N] first pass only
    int64_t *__restrict__ isect_keys,       // [n_isects] second pass only
    int32_t *__restrict__ isect_ids         // [n_isects] second pass only
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    const bool first_pass = cum_tiles == nullptr;
    if (radii[idx] <= 0) {
        if (first_pass) {
            tiles_per_surfel[idx] = 0;
        }
        return;
    }

    uint2 rect_min, rect_max;
    surfel_tile_rect(
        vec2(means2d[2 * idx], means2d[2 * idx + 1]), radii[idx], tiles_x, tiles_y,
        rect_min, rect_max
    );
    if (first_pass) {
        tiles_per_surfel[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
        return;
    }

    const int64_t depth_bits = static_cast<int64_t>(__float_as_uint(depths[idx]));
    int64_t cur = (idx == 0) ? 0 : cum_tiles[idx - 1];
    for (uint32_t y = rect_min.y; y < rect_max.y; ++y) {
        for (uint32_t x = rect_min.x; x < rect_max.x; ++x) {
            const int64_t tile_id = y * tiles_x + x;
            isect_keys[cur] = (tile_id << 32) | depth_bits;
            isect_ids[cur] = static_cast<int32_t>(idx);
            ++cur;
        }
    }
}

// [start, end) range of every tile in the sorted intersection list.
__global__ void tile_ranges_kernel(
    const uint32_t n_isects,
    const int64_t *__restrict__ sorted_keys,
    int32_t *__restrict__ tile_ranges // [n_tiles, 2], zero initialised
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects) {
        return;
    }
    const int64_t tile = sorted_keys[idx] >> 32;
    if (idx == 0) {
        tile_ranges[2 * tile] = 0;
    } else {
        const int64_t prev_tile = sorted_keys[idx - 1] >> 32;
        if (tile != prev_tile) {
            tile_ranges[2 * prev_tile + 1] = idx;
            tile_ranges[2 * tile] = idx;
        }
    }
    if (idx == n_isects - 1) {
        tile_ranges[2 * tile + 1] = n_isects;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t> intersect_tiles(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const int64_t image_width,
    const int64_t image_height
) {
    SS_CHECK_FLOAT(means2d);
    SS_CHECK_INPUT(radii);
    SS_CHECK_FLOAT(depths);
    const at::cuda::OptionalCUDAGuard device_guard(means2d.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const uint32_t N = means2d.size(0);
    const uint32_t tiles_x = num_tiles_x(image_width);
    const uint32_t tiles_y = num_tiles_y(image_height);
    const uint32_t n_tiles = tiles_x * tiles_y;
    auto int_opts = means2d.options().dtype(at::kInt);
    auto long_opts = means2d.options().dtype(at::kLong);

    at::Tensor tile_ranges = at::zeros({n_tiles, 2}, int_opts);
    at::Tensor bucket_offsets = at::zeros({n_tiles}, int_opts);
    if (N == 0) {
        return {tile_ranges, at::empty({0}, int_opts), bucket_offsets, 0};
    }

    // 1. tiles per surfel and their prefix sum
    at::Tensor tiles_per_surfel = at::empty({N}, int_opts);
    const dim3 grid((N + 255) / 256);
    intersect_tiles_kernel<<<grid, 256, 0, stream>>>(
        N, means2d.data_ptr<float>(), radii.data_ptr<int32_t>(),
        depths.data_ptr<float>(), nullptr, tiles_x, tiles_y,
        tiles_per_surfel.data_ptr<int32_t>(), nullptr, nullptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor cum_tiles = at::cumsum(tiles_per_surfel, 0, at::kLong);
    const int64_t n_isects = cum_tiles[N - 1].item<int64_t>();
    if (n_isects == 0) {
        return {tile_ranges, at::empty({0}, int_opts), bucket_offsets, 0};
    }

    // 2. one (tile | depth) key per surfel-tile intersection
    at::Tensor keys = at::empty({n_isects}, long_opts);
    at::Tensor ids = at::empty({n_isects}, int_opts);
    intersect_tiles_kernel<<<grid, 256, 0, stream>>>(
        N, means2d.data_ptr<float>(), radii.data_ptr<int32_t>(),
        depths.data_ptr<float>(), cum_tiles.data_ptr<int64_t>(), tiles_x, tiles_y,
        nullptr, keys.data_ptr<int64_t>(), ids.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 3. sort by tile, then front to back (stable: ties keep the surfel order)
    at::Tensor keys_alt = at::empty_like(keys);
    at::Tensor ids_alt = at::empty_like(ids);
    cub::DoubleBuffer<int64_t> d_keys(keys.data_ptr<int64_t>(), keys_alt.data_ptr<int64_t>());
    cub::DoubleBuffer<int32_t> d_values(ids.data_ptr<int32_t>(), ids_alt.data_ptr<int32_t>());
    const int tile_bits = (int)floor(log2((double)n_tiles)) + 1;
    SS_CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs, d_keys, d_values, (int)n_isects, 0,
        32 + tile_bits, stream
    );
    at::Tensor sorted_keys = d_keys.selector == 0 ? keys : keys_alt;
    at::Tensor flatten_ids = d_values.selector == 0 ? ids : ids_alt;

    // 4. per-tile ranges and bucket counts
    tile_ranges_kernel<<<(n_isects + 255) / 256, 256, 0, stream>>>(
        n_isects, sorted_keys.data_ptr<int64_t>(), tile_ranges.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor counts = tile_ranges.select(1, 1) - tile_ranges.select(1, 0);
    at::Tensor buckets = (counts + (int)(BUCKET_SIZE - 1)).floor_divide((int)BUCKET_SIZE);
    bucket_offsets = at::cumsum(buckets, 0, at::kInt); // inclusive
    const int64_t n_buckets = bucket_offsets[n_tiles - 1].item<int64_t>();

    return {tile_ranges, flatten_ids, bucket_offsets, n_buckets};
}

} // namespace strand_surfels
