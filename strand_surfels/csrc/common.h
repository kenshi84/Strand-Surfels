// SPDX-License-Identifier: MIT
//
// Shared definitions of the strand surfel rasterizer: vector types, constants
// of the image formation model and small device helpers.
//
// Conventions
// -----------
// * Cameras follow OpenCV: world-to-camera matrix `viewmat` (row-major 4x4),
//   intrinsics `K` (row-major 3x3), x right, y down, camera looking along +z.
// * Pixel (i, j) covers the continuous image coordinates [j, j+1) x [i, i+1)
//   and is evaluated at its center (j + 0.5, i + 0.5), as in gsplat.
// * Image-shaped outputs are channel-first: [channels, height, width].

#pragma once

#include <cstdint>

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "config.h"

namespace strand_surfels {

using vec2 = glm::vec<2, float>;
using vec3 = glm::vec<3, float>;
using vec4 = glm::vec<4, float>;
using mat3 = glm::mat<3, 3, float>;

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

#define SS_CHECK_INPUT(x)                                                      \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                   \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define SS_CHECK_FLOAT(x)                                                      \
    SS_CHECK_INPUT(x);                                                         \
    TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

// Temporary storage handling for the two-phase cub API (as in PyTorch/gsplat).
#define SS_CUB_WRAPPER(func, ...)                                              \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

inline __host__ __device__ uint32_t num_tiles_x(uint32_t width) { return (width + TILE_SIZE - 1) / TILE_SIZE; }
inline __host__ __device__ uint32_t num_tiles_y(uint32_t height) { return (height + TILE_SIZE - 1) / TILE_SIZE; }

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Rotation matrix of a quaternion (w, x, y, z), normalised internally.
inline __device__ mat3 quat_to_rotmat(const vec4 quat) {
    const float inv_norm = rsqrtf(glm::dot(quat, quat));
    const float w = quat[0] * inv_norm;
    const float x = quat[1] * inv_norm;
    const float y = quat[2] * inv_norm;
    const float z = quat[3] * inv_norm;
    // glm matrices are column-major
    return mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y)
    );
}

// Vector-Jacobian product of quat_to_rotmat with respect to the components of
// the *normalised* quaternion (w, x, y, z). v_R is column-major.
//
// Note: the Jacobian of the normalisation is not applied; callers pass unit
// quaternions normalised in PyTorch, whose autograd applies it.
inline __device__ vec4 quat_to_rotmat_vjp(const vec4 quat, const mat3 v_R) {
    const float inv_norm = rsqrtf(glm::dot(quat, quat));
    const float w = quat[0] * inv_norm;
    const float x = quat[1] * inv_norm;
    const float y = quat[2] * inv_norm;
    const float z = quat[3] * inv_norm;
    return vec4(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
               z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]))
    );
}

// Load the world-to-camera rotation (column-major glm) and translation from a
// row-major 4x4 matrix.
inline __device__ void load_viewmat(const float *viewmat, mat3 &R, vec3 &t) {
    R = mat3(viewmat[0], viewmat[4], viewmat[8],
             viewmat[1], viewmat[5], viewmat[9],
             viewmat[2], viewmat[6], viewmat[10]);
    t = vec3(viewmat[3], viewmat[7], viewmat[11]);
}

// Opacity dependent cut-off radius of a surfel in units of its standard
// deviation: the radius at which opacity * G(r) drops to 1/255 (with 3 sigma
// at full opacity). Smaller cut-offs give tighter bounding boxes.
inline __device__ float opacity_cutoff(const float opacity) {
    return sqrtf(fmaxf(9.f + 2.f * logf(opacity), 0.000001f));
}

// Range of image tiles [rect_min, rect_max) touched by a surfel with screen
// center `mean2d` (pixel-center convention) and square pixel radius `radius`.
// The bounds are evaluated in the integer-pixel frame (center - 0.5) of the
// legacy rasterizer, which keeps the tile assignment identical to it.
inline __device__ void surfel_tile_rect(
    const vec2 mean2d, const int radius, const uint32_t tiles_x,
    const uint32_t tiles_y, uint2 &rect_min, uint2 &rect_max
) {
    const float px = mean2d.x - 0.5f;
    const float py = mean2d.y - 0.5f;
    const float ts = static_cast<float>(TILE_SIZE);
    rect_min.x = min(tiles_x, (uint32_t)max(0, (int)((px - radius) / ts)));
    rect_min.y = min(tiles_y, (uint32_t)max(0, (int)((py - radius) / ts)));
    rect_max.x = min(tiles_x, (uint32_t)max(0, (int)((px + radius + ts - 1.f) / ts)));
    rect_max.y = min(tiles_y, (uint32_t)max(0, (int)((py + radius + ts - 1.f) / ts)));
}

// Normalised (NDC like) depth used by the depth distortion loss and its
// derivative with respect to the camera depth.
inline __device__ float ndc_depth(const float depth) {
    return FAR_PLANE / (FAR_PLANE - NEAR_PLANE) * (1.f - NEAR_PLANE / depth);
}
inline __device__ float ndc_depth_grad(const float depth) {
    return (FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth * depth);
}

} // namespace strand_surfels
