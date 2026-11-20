// SPDX-License-Identifier: MIT
//
// View dependent surfel colors from real spherical harmonics (degree <= 3).
//
// color = max(0.5 + sum_k c_k Y_k(d), 0), d = normalize(mean - camera_center),
// with the basis and sign convention of 3D Gaussian Splatting (see
// strand_surfels/rendering/sh.py for the PyTorch reference implementation).
// Only visible surfels (radius > 0) are evaluated; the others get zero colors
// and zero gradients.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "common.h"
#include "ops.h"

namespace strand_surfels {

namespace cg = cooperative_groups;

namespace {

constexpr float C0 = 0.28209479177387814f;
constexpr float C1 = 0.4886025119029199f;
__device__ constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                         -1.0925484305920792f, 0.5462742152960396f};
__device__ constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                         0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                         -0.5900435899266435f};

// SH basis values Y_k(d) for k < (degree + 1)^2 and, if `grad` is given, their
// gradients with respect to the (unit) direction d.
inline __device__ void sh_basis(const uint32_t degree, const vec3 d, float *Y, vec3 *grad) {
    const float x = d.x, y = d.y, z = d.z;
    Y[0] = C0;
    if (grad) grad[0] = vec3(0.f);
    if (degree < 1) return;
    Y[1] = -C1 * y;
    Y[2] = C1 * z;
    Y[3] = -C1 * x;
    if (grad) {
        grad[1] = vec3(0.f, -C1, 0.f);
        grad[2] = vec3(0.f, 0.f, C1);
        grad[3] = vec3(-C1, 0.f, 0.f);
    }
    if (degree < 2) return;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    Y[4] = C2[0] * xy;
    Y[5] = C2[1] * yz;
    Y[6] = C2[2] * (2.f * zz - xx - yy);
    Y[7] = C2[3] * xz;
    Y[8] = C2[4] * (xx - yy);
    if (grad) {
        grad[4] = C2[0] * vec3(y, x, 0.f);
        grad[5] = C2[1] * vec3(0.f, z, y);
        grad[6] = C2[2] * vec3(-2.f * x, -2.f * y, 4.f * z);
        grad[7] = C2[3] * vec3(z, 0.f, x);
        grad[8] = C2[4] * vec3(2.f * x, -2.f * y, 0.f);
    }
    if (degree < 3) return;
    Y[9] = C3[0] * y * (3.f * xx - yy);
    Y[10] = C3[1] * xy * z;
    Y[11] = C3[2] * y * (4.f * zz - xx - yy);
    Y[12] = C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
    Y[13] = C3[4] * x * (4.f * zz - xx - yy);
    Y[14] = C3[5] * z * (xx - yy);
    Y[15] = C3[6] * x * (xx - 3.f * yy);
    if (grad) {
        grad[9] = C3[0] * vec3(6.f * xy, 3.f * xx - 3.f * yy, 0.f);
        grad[10] = C3[1] * vec3(yz, xz, xy);
        grad[11] = C3[2] * vec3(-2.f * xy, 4.f * zz - xx - 3.f * yy, 8.f * yz);
        grad[12] = C3[3] * vec3(-6.f * xz, -6.f * yz, 6.f * zz - 3.f * xx - 3.f * yy);
        grad[13] = C3[4] * vec3(4.f * zz - 3.f * xx - yy, -2.f * xy, 8.f * xz);
        grad[14] = C3[5] * vec3(2.f * xz, -2.f * yz, xx - yy);
        grad[15] = C3[6] * vec3(3.f * xx - 3.f * yy, -6.f * xy, 0.f);
    }
}

// sum_k Y_k c_k of one color channel (same summation order in both passes)
inline __device__ float sh_sum(const float *Y, const float *c, const uint32_t n_bases, const uint32_t ch) {
    float v = 0.f;
    for (uint32_t k = 0; k < n_bases; ++k) {
        v += Y[k] * c[3 * k + ch];
    }
    return v;
}

} // namespace

__global__ void sh_to_color_fwd_kernel(
    const uint32_t N,
    const uint32_t K,      // coefficients per channel stored in `coeffs`
    const uint32_t degree, // active degree, (degree + 1)^2 <= K
    const float *__restrict__ coeffs,        // [N, K, 3]
    const float *__restrict__ means,         // [N, 3]
    const float *__restrict__ camera_center, // [3]
    const int32_t *__restrict__ radii,       // [N]
    float *__restrict__ colors               // [N, 3]
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx] <= 0) {
        return;
    }
    const vec3 raw = glm::make_vec3(means + 3 * idx) - glm::make_vec3(camera_center);
    const vec3 d = raw * rsqrtf(glm::dot(raw, raw));
    float Y[16];
    sh_basis(degree, d, Y, nullptr);
    const uint32_t n_bases = (degree + 1) * (degree + 1);
    const float *c = coeffs + idx * K * 3;
#pragma unroll
    for (uint32_t ch = 0; ch < 3; ++ch) {
        colors[3 * idx + ch] = fmaxf(sh_sum(Y, c, n_bases, ch) + 0.5f, 0.f);
    }
}

__global__ void sh_to_color_bwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degree,
    const float *__restrict__ coeffs,
    const float *__restrict__ means,
    const float *__restrict__ camera_center,
    const int32_t *__restrict__ radii,
    const float *__restrict__ colors,   // [N, 3] forward output (unused)
    const float *__restrict__ v_colors, // [N, 3]
    float *__restrict__ v_coeffs,       // [N, K, 3]
    float *__restrict__ v_means         // [N, 3]
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx] <= 0) {
        return;
    }
    const vec3 raw = glm::make_vec3(means + 3 * idx) - glm::make_vec3(camera_center);
    const float inv_len = rsqrtf(glm::dot(raw, raw));
    const vec3 d = raw * inv_len;
    float Y[16];
    vec3 dY[16];
    sh_basis(degree, d, Y, dY);
    const uint32_t n_bases = (degree + 1) * (degree + 1);

    // colors clamped at zero (negative before clamping) receive no gradient
    const float *c = coeffs + idx * K * 3;
    vec3 v_c;
#pragma unroll
    for (uint32_t ch = 0; ch < 3; ++ch) {
        const bool clamped = sh_sum(Y, c, n_bases, ch) + 0.5f < 0.f;
        v_c[ch] = clamped ? 0.f : v_colors[3 * idx + ch];
    }
    float *v_c_out = v_coeffs + idx * K * 3;
    vec3 v_d(0.f);
    for (uint32_t k = 0; k < n_bases; ++k) {
        const vec3 ck = vec3(c[3 * k], c[3 * k + 1], c[3 * k + 2]);
        v_c_out[3 * k] = Y[k] * v_c.x;
        v_c_out[3 * k + 1] = Y[k] * v_c.y;
        v_c_out[3 * k + 2] = Y[k] * v_c.z;
        v_d += dY[k] * glm::dot(ck, v_c);
    }
    // through the normalisation d = raw / |raw|
    const vec3 v_raw = (v_d - d * glm::dot(d, v_d)) * inv_len;
    v_means[3 * idx] = v_raw.x;
    v_means[3 * idx + 1] = v_raw.y;
    v_means[3 * idx + 2] = v_raw.z;
}

at::Tensor sh_to_color_fwd(
    const int64_t degree,
    const at::Tensor &coeffs,
    const at::Tensor &means,
    const at::Tensor &camera_center,
    const at::Tensor &radii
) {
    SS_CHECK_FLOAT(coeffs);
    SS_CHECK_FLOAT(means);
    SS_CHECK_FLOAT(camera_center);
    SS_CHECK_INPUT(radii);
    TORCH_CHECK(degree >= 0 && degree <= 3, "SH degree must be in [0, 3]");
    TORCH_CHECK(coeffs.size(1) >= (degree + 1) * (degree + 1), "too few SH coefficients");
    const at::cuda::OptionalCUDAGuard device_guard(means.device());
    const uint32_t N = means.size(0);
    at::Tensor colors = at::zeros({N, 3}, means.options());
    if (N > 0) {
        sh_to_color_fwd_kernel<<<(N + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, coeffs.size(1), degree, coeffs.data_ptr<float>(), means.data_ptr<float>(),
            camera_center.data_ptr<float>(), radii.data_ptr<int32_t>(), colors.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return colors;
}

std::tuple<at::Tensor, at::Tensor> sh_to_color_bwd(
    const int64_t degree,
    const at::Tensor &coeffs,
    const at::Tensor &means,
    const at::Tensor &camera_center,
    const at::Tensor &radii,
    const at::Tensor &colors,
    const at::Tensor &v_colors
) {
    SS_CHECK_FLOAT(coeffs);
    SS_CHECK_FLOAT(means);
    SS_CHECK_FLOAT(camera_center);
    SS_CHECK_INPUT(radii);
    SS_CHECK_FLOAT(colors);
    SS_CHECK_FLOAT(v_colors);
    const at::cuda::OptionalCUDAGuard device_guard(means.device());
    const uint32_t N = means.size(0);
    at::Tensor v_coeffs = at::zeros_like(coeffs);
    at::Tensor v_means = at::zeros_like(means);
    if (N > 0) {
        sh_to_color_bwd_kernel<<<(N + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, coeffs.size(1), degree, coeffs.data_ptr<float>(), means.data_ptr<float>(),
            camera_center.data_ptr<float>(), radii.data_ptr<int32_t>(), colors.data_ptr<float>(),
            v_colors.data_ptr<float>(), v_coeffs.data_ptr<float>(), v_means.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {v_coeffs, v_means};
}

} // namespace strand_surfels
