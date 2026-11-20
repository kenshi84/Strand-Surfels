// SPDX-License-Identifier: MIT
//
// Fused geometry loss between rendered strand maps and the line-based
// multi-view stereo result (SMHC, see the README).
//
// Inputs per view
// ---------------
// * depth [H, W]    rendered (expected or median) depth
// * dirs  [3, H, W] rendered strand direction (world space, alpha normalised)
// * int_map [3, H, W] intermediate map of the line-based MVS stage:
//     - multi-view consistent pixels: (depth > 0, dir.x, dir.y), the 3D strand
//       direction with dir.z = sqrt(1 - dir.x^2 - dir.y^2);
//     - all other pixels: (-1, 2D orientation angle, orientation confidence).
// * the line camera of the view (see `line_camera_matrices` in
//   strand_surfels/losses/geometry.py): M_inv [3, 3], P_col3 [3] with the
//   projection P = [M | P_col3] = K [R | t], and the Pluecker line projection
//   matrix [3, 6] (only its first two rows are used).
//
// Output loss map [3, H, W]
// -------------------------
// * channel 0: |depth - depth_mvs|                    (consistent pixels)
// * channel 1: 1 - <dir, +-dir_mvs>                    (consistent pixels)
// * channel 2: confidence-weighted difference between the 2D orientation of
//              the projected rendered strand and the measured orientation,
//              divided by 10                           (all other pixels)
// Channels 1 and 2 are zero where the rendered direction vanishes. Pixel
// (x, y) is back-projected at the integer image coordinates (x, y), the pixel
// convention of the int maps.
//
// The forward pass also produces the per-pixel derivatives of the loss map,
// which the backward pass scales by the incoming gradient.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "common.h"
#include "ops.h"

namespace strand_surfels {

namespace {

constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

// The orientation wrap-around is evaluated in double precision (pi and pi/2
// as doubles), as in the original implementation.
constexpr double PI_D = 3.14159265358979323846;
constexpr double HALF_PI_D = 1.57079632679489661923;
constexpr float ANGLE_EPS = 1e-7f;      // regularises a / b in the line angle
constexpr float NORMALIZE_EPS = 1e-6f;  // squared norms below are treated as zero
constexpr float ORIENTATION_2D_DIVISOR = 10.f; // constant down-weighting of the 2D term

// Line camera of one view in constant-size form.
struct LineCamera {
    const float *m_inv;     // [3, 3] row-major
    const float *p_col3;    // [3]
    const float *line_proj; // [3, 6] row-major
};

inline __device__ float3 cross(const float3 a, const float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ float dot(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// v / |v|, or zero for (near) zero vectors
inline __device__ float3 normalize_or_zero(float3 v) {
    const float norm_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    const float inv = norm_sq > NORMALIZE_EPS ? rsqrtf(norm_sq) : 0.f;
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

// Unit 3D direction from its stored (x, y) components, z >= 0.
inline __device__ float3 direction_from_xy(const float x, const float y) {
    float z = sqrtf(1 - x * x - y * y);
    z = (z != z) ? 0.f : z; // NaN for slightly non-unit inputs
    return normalize_or_zero(make_float3(x, y, z));
}

// World point seen at integer pixel (px, py) with the given depth:
// X = M^-1 (depth * (px, py, 1) - P_col3).
inline __device__ float3 back_project(const int px, const int py, const float depth, const LineCamera &cam) {
    const float3 p = make_float3(depth * px - cam.p_col3[0], depth * py - cam.p_col3[1], depth - cam.p_col3[2]);
    const float *m = cam.m_inv;
    return make_float3(
        m[0] * p.x + m[1] * p.y + m[2] * p.z, m[3] * p.x + m[4] * p.y + m[5] * p.z,
        m[6] * p.x + m[7] * p.y + m[8] * p.z
    );
}

// Image line a x + b y + c = 0 of the 3D line through `point` along `dir`;
// returns (a, b). The Pluecker coordinates (dir, moment = point x dir) are
// passed in reversed order to the line projection matrix (Hartley and
// Zisserman, Eqs. 3.13 and 8.4).
inline __device__ float2 project_line(const float3 point, const float3 dir, const LineCamera &cam) {
    const float3 moment = cross(point, dir);
    const float3 rdir = make_float3(moment.z, moment.y, moment.x);
    const float3 rmom = make_float3(dir.z, dir.y, dir.x);
    const float *L = cam.line_proj;
    return make_float2(
        L[0] * rdir.x + L[1] * rdir.y + L[2] * rdir.z + L[3] * rmom.x + L[4] * rmom.y + L[5] * rmom.z,
        L[6] * rdir.x + L[7] * rdir.y + L[8] * rdir.z + L[9] * rmom.x + L[10] * rmom.y + L[11] * rmom.z
    );
}

// Orientation of the image line (a, b) in [-pi/2, pi/2].
inline __device__ float line_angle(const float2 line) { return atanf(-line.x / (line.y + ANGLE_EPS)); }

} // namespace

__global__ void geometry_loss_fwd_kernel(
    const int H,
    const int W,
    const float *__restrict__ depth,   // [H, W]
    const float *__restrict__ dirs,    // [3, H, W]
    const float *__restrict__ int_map, // [3, H, W]
    const LineCamera cam,
    float *__restrict__ loss,          // [3, H, W]
    float *__restrict__ jac_depth,     // [H, W] d loss[0] / d depth, or nullptr
    float *__restrict__ jac_dir        // [3, H, W] d loss[1 or 2] / d dir, or nullptr
) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }
    const int HW = H * W;
    const int pix = y * W + x;

    const float d_est = depth[pix];
    const float3 dir_est = make_float3(dirs[pix], dirs[HW + pix], dirs[2 * HW + pix]);
    const float dir_mask = dot(dir_est, dir_est) > NORMALIZE_EPS ? 1.f : 0.f;
    const float d_mvs = int_map[pix];

    if (d_mvs > 0.f) {
        // 3D terms: depth and (sign-agnostic) strand direction
        float3 dir_mvs = direction_from_xy(int_map[HW + pix], int_map[2 * HW + pix]);
        if (dot(dir_mvs, dir_est) < 0) {
            dir_mvs = make_float3(-dir_mvs.x, -dir_mvs.y, -dir_mvs.z);
        }
        loss[pix] = fabsf(d_est - d_mvs);
        loss[HW + pix] = (1 - dot(dir_est, dir_mvs)) * dir_mask;
        if (jac_depth) {
            jac_depth[pix] = (d_est < d_mvs) ? -1.f : (d_est > d_mvs) ? 1.f : 0.f;
            jac_dir[pix] = -dir_mvs.x * dir_mask;
            jac_dir[HW + pix] = -dir_mvs.y * dir_mask;
            jac_dir[2 * HW + pix] = -dir_mvs.z * dir_mask;
        }
        return;
    }

    // 2D term: orientation of the projected rendered strand vs the measured
    // orientation (an undirected angle, compared modulo pi)
    const float3 point = back_project(x, y, d_est, cam);
    const float2 line = project_line(point, dir_est, cam);
    const float angle = line_angle(line);
    float ori = int_map[HW + pix];
    const float conf = int_map[2 * HW + pix];

    float cost, d_cost_d_angle;
    if (fabsf(angle - ori) >= HALF_PI_D && ori >= 0) {
        ori -= PI_D;
        cost = (angle - ori) * conf;
        d_cost_d_angle = conf;
    } else if (fabsf(angle - ori) >= HALF_PI_D && ori < 0) {
        ori += PI_D;
        cost = (ori - angle) * conf;
        d_cost_d_angle = -conf;
    } else if (fabsf(angle - ori) < HALF_PI_D && ori >= angle) {
        cost = (ori - angle) * conf;
        d_cost_d_angle = -conf;
    } else {
        cost = (angle - ori) * conf;
        d_cost_d_angle = conf;
    }
    cost /= ORIENTATION_2D_DIVISOR;
    d_cost_d_angle /= ORIENTATION_2D_DIVISOR;
    loss[2 * HW + pix] = cost * dir_mask;

    if (jac_depth) {
        // through angle = atan(-a / b) to the line coefficients (a, b)
        const float ab_sq = line.x * line.x + line.y * line.y + ANGLE_EPS;
        const float v_a = d_cost_d_angle * (-line.y / ab_sq);
        const float v_b = d_cost_d_angle * (line.x / ab_sq);
        // transpose of the line projection, un-reversing the Pluecker order
        const float *L = cam.line_proj;
        const float3 v_moment = make_float3(L[2] * v_a + L[8] * v_b, L[1] * v_a + L[7] * v_b, L[0] * v_a + L[6] * v_b);
        float3 v_dir = make_float3(L[5] * v_a + L[11] * v_b, L[4] * v_a + L[10] * v_b, L[3] * v_a + L[9] * v_b);
        // component-wise contribution through the moment = point x dir
        v_dir.x += v_moment.x * (point.z - point.y);
        v_dir.y += v_moment.y * (point.x - point.z);
        v_dir.z += v_moment.z * (point.y - point.x);
        jac_dir[pix] = v_dir.x * dir_mask;
        jac_dir[HW + pix] = v_dir.y * dir_mask;
        jac_dir[2 * HW + pix] = v_dir.z * dir_mask;
    }
}

__global__ void geometry_loss_bwd_kernel(
    const int H,
    const int W,
    const float *__restrict__ int_map,   // [3, H, W]
    const float *__restrict__ jac_depth, // [H, W]
    const float *__restrict__ jac_dir,   // [3, H, W]
    const float *__restrict__ v_loss,    // [3, H, W]
    float *__restrict__ v_depth,         // [H, W]
    float *__restrict__ v_dirs           // [3, H, W]
) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }
    const int HW = H * W;
    const int pix = y * W + x;
    const bool consistent = int_map[pix] > 0.f;
    // the direction enters loss channel 1 (3D term) or 2 (2D term)
    const float v_dir_term = consistent ? v_loss[HW + pix] : v_loss[2 * HW + pix];
    v_depth[pix] = consistent ? v_loss[pix] * jac_depth[pix] : 0.f;
    v_dirs[pix] = v_dir_term * jac_dir[pix];
    v_dirs[HW + pix] = v_dir_term * jac_dir[HW + pix];
    v_dirs[2 * HW + pix] = v_dir_term * jac_dir[2 * HW + pix];
}

__global__ void orientation_2d_kernel(
    const int H,
    const int W,
    const float *__restrict__ depth, // [H, W]
    const float *__restrict__ dirs,  // [3, H, W]
    const LineCamera cam,
    float *__restrict__ angles       // [H, W]
) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }
    const int HW = H * W;
    const int pix = y * W + x;
    const float3 dir = normalize_or_zero(make_float3(dirs[pix], dirs[HW + pix], dirs[2 * HW + pix]));
    const float3 point = back_project(x, y, depth[pix], cam);
    angles[pix] = line_angle(project_line(point, dir, cam));
}

namespace {

void check_maps(const at::Tensor &depth, const at::Tensor &dirs) {
    SS_CHECK_FLOAT(depth);
    SS_CHECK_FLOAT(dirs);
    TORCH_CHECK(depth.dim() == 2, "depth must have shape [H, W]");
    TORCH_CHECK(dirs.dim() == 3 && dirs.size(0) == 3 && dirs.size(1) == depth.size(0) &&
                    dirs.size(2) == depth.size(1),
                "dirs must have shape [3, H, W]");
}

LineCamera make_line_camera(const at::Tensor &m_inv, const at::Tensor &p_col3, const at::Tensor &line_proj) {
    SS_CHECK_FLOAT(m_inv);
    SS_CHECK_FLOAT(p_col3);
    SS_CHECK_FLOAT(line_proj);
    TORCH_CHECK(m_inv.numel() == 9 && p_col3.numel() == 3 && line_proj.numel() == 18,
                "line camera must be M_inv [3, 3], P_col3 [3], line projection [3, 6]");
    return {m_inv.data_ptr<float>(), p_col3.data_ptr<float>(), line_proj.data_ptr<float>()};
}

dim3 image_grid(const int H, const int W) { return dim3((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y); }

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> geometry_loss_fwd(
    const at::Tensor &depth,
    const at::Tensor &dirs,
    const at::Tensor &int_map,
    const at::Tensor &m_inv,
    const at::Tensor &p_col3,
    const at::Tensor &line_proj,
    const bool need_grad
) {
    check_maps(depth, dirs);
    SS_CHECK_FLOAT(int_map);
    TORCH_CHECK(int_map.sizes() == dirs.sizes(), "int_map must have shape [3, H, W]");
    const LineCamera cam = make_line_camera(m_inv, p_col3, line_proj);
    const at::cuda::OptionalCUDAGuard device_guard(depth.device());
    const int H = depth.size(0), W = depth.size(1);

    at::Tensor loss = at::zeros({3, H, W}, depth.options());
    at::Tensor jac_depth = need_grad ? at::zeros({H, W}, depth.options()) : at::Tensor();
    at::Tensor jac_dir = need_grad ? at::zeros({3, H, W}, depth.options()) : at::Tensor();
    if (H * W > 0) {
        geometry_loss_fwd_kernel<<<image_grid(H, W), dim3(BLOCK_X, BLOCK_Y), 0, at::cuda::getCurrentCUDAStream()>>>(
            H, W, depth.data_ptr<float>(), dirs.data_ptr<float>(), int_map.data_ptr<float>(), cam,
            loss.data_ptr<float>(), need_grad ? jac_depth.data_ptr<float>() : nullptr,
            need_grad ? jac_dir.data_ptr<float>() : nullptr
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {loss, jac_depth, jac_dir};
}

std::tuple<at::Tensor, at::Tensor> geometry_loss_bwd(
    const at::Tensor &int_map,
    const at::Tensor &jac_depth,
    const at::Tensor &jac_dir,
    const at::Tensor &v_loss
) {
    check_maps(jac_depth, jac_dir);
    SS_CHECK_FLOAT(int_map);
    SS_CHECK_FLOAT(v_loss);
    TORCH_CHECK(int_map.sizes() == jac_dir.sizes() && v_loss.sizes() == jac_dir.sizes(),
                "int_map and the loss gradient must have shape [3, H, W]");
    const at::cuda::OptionalCUDAGuard device_guard(v_loss.device());
    const int H = jac_depth.size(0), W = jac_depth.size(1);

    at::Tensor v_depth = at::empty({H, W}, v_loss.options());
    at::Tensor v_dirs = at::empty({3, H, W}, v_loss.options());
    if (H * W > 0) {
        geometry_loss_bwd_kernel<<<image_grid(H, W), dim3(BLOCK_X, BLOCK_Y), 0, at::cuda::getCurrentCUDAStream()>>>(
            H, W, int_map.data_ptr<float>(), jac_depth.data_ptr<float>(), jac_dir.data_ptr<float>(),
            v_loss.data_ptr<float>(), v_depth.data_ptr<float>(), v_dirs.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {v_depth, v_dirs};
}

at::Tensor orientation_2d(
    const at::Tensor &depth,
    const at::Tensor &dirs,
    const at::Tensor &m_inv,
    const at::Tensor &p_col3,
    const at::Tensor &line_proj
) {
    check_maps(depth, dirs);
    const LineCamera cam = make_line_camera(m_inv, p_col3, line_proj);
    const at::cuda::OptionalCUDAGuard device_guard(depth.device());
    const int H = depth.size(0), W = depth.size(1);
    at::Tensor angles = at::zeros({H, W}, depth.options());
    if (H * W > 0) {
        orientation_2d_kernel<<<image_grid(H, W), dim3(BLOCK_X, BLOCK_Y), 0, at::cuda::getCurrentCUDAStream()>>>(
            H, W, depth.data_ptr<float>(), dirs.data_ptr<float>(), cam, angles.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return angles;
}

} // namespace strand_surfels
