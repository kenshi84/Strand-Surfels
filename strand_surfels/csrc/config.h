// SPDX-License-Identifier: MIT
//
// Compile-time configuration of the strand surfel rasterizer (plain C++, shared
// by the CUDA sources and the host bindings).

#pragma once

#include <cstdint>

namespace strand_surfels {

// ---------------------------------------------------------------------------
// Rasterizer configuration
// ---------------------------------------------------------------------------

// Image tiles are TILE_SIZE x TILE_SIZE pixels, one thread block per tile.
constexpr uint32_t TILE_SIZE = 16;
constexpr uint32_t TILE_PIXELS = TILE_SIZE * TILE_SIZE;

// The backward pass processes the depth-sorted surfels of a tile in buckets of
// BUCKET_SIZE (one warp). The forward pass stores the blending state of every
// pixel of the tile at the start of each bucket.
constexpr uint32_t BUCKET_SIZE = 32;

// Number of color channels (RGB).
constexpr uint32_t COLOR_DIM = 3;

// Surfels closer than this to the camera are culled / not blended.
constexpr float NEAR_PLANE = 0.2f;
// Far plane of the normalised depth used by the depth distortion loss.
constexpr float FAR_PLANE = 10.0f;

// Screen space low-pass filter of 2DGS: the surfel kernel is the minimum of the
// ray-surfel intersection kernel and an isotropic 2D Gaussian of this size.
constexpr float FILTER_SIZE = 0.707106f; // sqrt(2) / 2
constexpr float FILTER_INV_SQUARE = 2.0f;

// Alpha blending limits.
constexpr float MAX_ALPHA = 0.99f;
constexpr float MIN_ALPHA = 1.0f / 255.0f;
constexpr float MIN_TRANSMITTANCE = 0.0001f;

// Layout of the auxiliary output image `render_aux` [AUX_DIM, H, W].
// Depth and direction are alpha-weighted sums (not normalised by alpha).
constexpr uint32_t AUX_DEPTH = 0;      // expected depth (camera z)
constexpr uint32_t AUX_ALPHA = 1;      // accumulated opacity
constexpr uint32_t AUX_DIR = 2;        // 3 channels: world space strand direction
constexpr uint32_t AUX_DISTORTION = 5; // depth distortion
constexpr uint32_t AUX_MEDIAN = 6;     // median depth
constexpr uint32_t AUX_DIM = 7;

// Layout of the per-pixel state kept for the backward pass.
// final_state [FINAL_STATE_DIM, H, W]: transmittance, sum(w*m), sum(w*m^2)
constexpr uint32_t FINAL_STATE_DIM = 3;
// contributors [2, H, W]: number of processed surfels up to the last blended
// one and up to the median depth surfel (1-based positions in the tile list)
constexpr uint32_t CONTRIB_DIM = 2;

// Per bucket and pixel, the forward pass samples
//   sampled_state [n_buckets, 3, TILE_PIXELS]: T, sum(w*m), sum(w*m^2)
//   sampled_color [n_buckets, COLOR_DIM, TILE_PIXELS]: accumulated color
//   sampled_aux   [n_buckets, 4, TILE_PIXELS]: accumulated depth and direction
constexpr uint32_t SAMPLED_STATE_DIM = 3;
constexpr uint32_t SAMPLED_AUX_DIM = 4;

} // namespace strand_surfels
