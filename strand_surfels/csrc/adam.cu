// SPDX-License-Identifier: MIT
//
// Sparse Adam update: only the parameters of surfels that were visible in the
// current view are updated (moments and values); all others are left
// untouched. As in the reference implementation, no bias correction is
// applied.
//
// Adapted from Taming 3DGS (https://github.com/humansensinglab/taming-3dgs,
// MIT, Copyright (c) 2024 The Authors,
// submodules/diff-gaussian-rasterization/cuda_rasterizer/adam.cu).

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "common.h"
#include "ops.h"

namespace strand_surfels {

namespace cg = cooperative_groups;

// One thread per parameter element; the parameter tensor is viewed as
// [N, M] with N surfels and M values per surfel.
__global__ void sparse_adam_kernel(
    const uint32_t N,
    const uint32_t M,
    float *__restrict__ param,
    const float *__restrict__ grad,
    float *__restrict__ exp_avg,
    float *__restrict__ exp_avg_sq,
    const bool *__restrict__ visible, // [N]
    const float lr,
    const float beta1,
    const float beta2,
    const float eps
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * M || !visible[idx / M]) {
        return;
    }
    const float g = grad[idx];
    const float m = beta1 * exp_avg[idx] + (1.0f - beta1) * g;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g * g;
    param[idx] -= lr * m / (sqrtf(v) + eps);
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

void sparse_adam_update(
    at::Tensor &param,
    const at::Tensor &grad,
    at::Tensor &exp_avg,
    at::Tensor &exp_avg_sq,
    const at::Tensor &visible,
    const double lr,
    const double beta1,
    const double beta2,
    const double eps
) {
    SS_CHECK_FLOAT(param);
    SS_CHECK_FLOAT(grad);
    SS_CHECK_FLOAT(exp_avg);
    SS_CHECK_FLOAT(exp_avg_sq);
    SS_CHECK_INPUT(visible);
    TORCH_CHECK(visible.scalar_type() == at::kBool, "visible must be a bool tensor");
    const uint32_t N = visible.numel();
    TORCH_CHECK(N > 0 && param.numel() % N == 0, "parameter size must be a multiple of N");
    TORCH_CHECK(grad.numel() == param.numel() && exp_avg.numel() == param.numel() &&
                    exp_avg_sq.numel() == param.numel(),
                "parameter, gradient and moments must have the same size");
    const at::cuda::OptionalCUDAGuard device_guard(param.device());
    const uint32_t M = param.numel() / N;
    const uint32_t n = N * M;
    sparse_adam_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        N, M, param.data_ptr<float>(), grad.data_ptr<float>(), exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(), visible.data_ptr<bool>(), (float)lr, (float)beta1,
        (float)beta2, (float)eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace strand_surfels
