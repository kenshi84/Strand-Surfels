// SPDX-License-Identifier: MIT
//
// Fused geometry loss between rendered strand maps and the line-based
// multi-view stereo result (SMHC, see the README).
//
// Inputs per view
// ---------------
// * depth [H, W]    rendered (expected or median) depth
// * tensors [6, H, W] alpha-normalised structure tensor E[t t^T], stored as
//   (xx, xy, xz, yy, yz, zz)
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
// * channel 1: 1 - dir_mvs^T T dir_mvs                 (consistent pixels)
// * channel 2: confidence * (sqrt(n^T T n + eps^2) - eps) / 10, where n is
//              normal to the plane of 3D directions projecting to the
//              measured image orientation                (all other pixels)
// Channels 1 and 2 are zero where the rendered tensor vanishes. Pixel
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

constexpr float ANGLE_EPS = 1e-7f;      // regularises a / b in the line angle
constexpr float NORMALIZE_EPS = 1e-6f;  // squared norms below are treated as zero
constexpr float PLANE_NORMAL_EPS = 1e-20f;
constexpr float ORIENTATION_2D_DIVISOR = 10.f; // constant down-weighting of the 2D term
constexpr float ORIENTATION_2D_EPS = 1e-3f; // smooth |sin(angle)| near zero

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

// Quadratic form v^T T v for symmetric T stored as xx, xy, xz, yy, yz, zz.
inline __device__ float quadratic(const float *T, const int stride, const int pix, const float3 v) {
    return T[pix] * v.x * v.x + 2.f * T[stride + pix] * v.x * v.y +
           2.f * T[2 * stride + pix] * v.x * v.z + T[3 * stride + pix] * v.y * v.y +
           2.f * T[4 * stride + pix] * v.y * v.z + T[5 * stride + pix] * v.z * v.z;
}

inline __device__ void quadratic_basis(const float3 v, float out[6]) {
    out[0] = v.x * v.x;
    out[1] = 2.f * v.x * v.y;
    out[2] = 2.f * v.x * v.z;
    out[3] = v.y * v.y;
    out[4] = 2.f * v.y * v.z;
    out[5] = v.z * v.z;
}

// v / |v|, or zero for (near) zero vectors
inline __device__ float3 normalize_or_zero(float3 v) {
    const float norm_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    const float inv = norm_sq > NORMALIZE_EPS ? rsqrtf(norm_sq) : 0.f;
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

inline __device__ float3 normalize_plane_or_zero(float3 v) {
    const float norm_sq = dot(v, v);
    const float inv = norm_sq > PLANE_NORMAL_EPS ? rsqrtf(norm_sq) : 0.f;
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

// Unit 3D direction from its stored (x, y) components, z >= 0.
inline __device__ float3 direction_from_xy(const float x, const float y) {
    float z = sqrtf(1 - x * x - y * y);
    z = (z != z) ? 0.f : z; // NaN for slightly non-unit inputs
    return normalize_or_zero(make_float3(x, y, z));
}

// World-space ray direction M^-1 (px, py, 1).
inline __device__ float3 pixel_ray(const float px, const float py, const LineCamera &cam) {
    const float *m = cam.m_inv;
    return make_float3(
        m[0] * px + m[1] * py + m[2], m[3] * px + m[4] * py + m[5],
        m[6] * px + m[7] * py + m[8]
    );
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
    const float *__restrict__ tensors, // [6, H, W]
    const float *__restrict__ int_map, // [3, H, W]
    const LineCamera cam,
    float *__restrict__ loss,          // [3, H, W]
    float *__restrict__ jac_depth,     // [H, W] d loss[0] / d depth, or nullptr
    float *__restrict__ jac_tensor     // [6, H, W] d loss[1 or 2] / d T, or nullptr
) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }
    const int HW = H * W;
    const int pix = y * W + x;

    const float d_est = depth[pix];
    const float trace = tensors[pix] + tensors[3 * HW + pix] + tensors[5 * HW + pix];
    const float tensor_mask = trace > NORMALIZE_EPS ? 1.f : 0.f;
    const float d_mvs = int_map[pix];

    if (d_mvs > 0.f) {
        // 3D direction term is sign invariant: for rank-one T=t t^T this is
        // 1 - (t . t_mvs)^2.
        const float3 dir_mvs = direction_from_xy(int_map[HW + pix], int_map[2 * HW + pix]);
        float basis[6];
        quadratic_basis(dir_mvs, basis);
        loss[pix] = fabsf(d_est - d_mvs);
        loss[HW + pix] = (1.f - quadratic(tensors, HW, pix, dir_mvs)) * tensor_mask;
        if (jac_depth) {
            jac_depth[pix] = (d_est < d_mvs) ? -1.f : (d_est > d_mvs) ? 1.f : 0.f;
#pragma unroll
            for (int c = 0; c < 6; ++c) {
                jac_tensor[c * HW + pix] = -basis[c] * tensor_mask;
            }
        }
        return;
    }

    // The measured undirected image orientation defines a plane through the
    // camera centre. Every 3D direction that projects to that orientation lies
    // in the plane, so n^T T n is a differentiable, sign-invariant residual.
    const float ori = int_map[HW + pix];
    const float conf = int_map[2 * HW + pix];
    const float3 ray0 = pixel_ray((float)x, (float)y, cam);
    const float3 ray1 = pixel_ray((float)x + cosf(ori), (float)y + sinf(ori), cam);
    const float3 normal = normalize_plane_or_zero(cross(ray0, ray1));
    float basis[6];
    quadratic_basis(normal, basis);
    const float q = fmaxf(quadratic(tensors, HW, pix, normal), 0.f);
    const float root = sqrtf(q + ORIENTATION_2D_EPS * ORIENTATION_2D_EPS);
    const float scale = conf / ORIENTATION_2D_DIVISOR * tensor_mask;
    loss[2 * HW + pix] = (root - ORIENTATION_2D_EPS) * scale;

    if (jac_depth) {
        const float jac_scale = scale * 0.5f / root;
#pragma unroll
        for (int c = 0; c < 6; ++c) {
            jac_tensor[c * HW + pix] = basis[c] * jac_scale;
        }
    }
}

__global__ void geometry_loss_bwd_kernel(
    const int H,
    const int W,
    const float *__restrict__ int_map,   // [3, H, W]
    const float *__restrict__ jac_depth, // [H, W]
    const float *__restrict__ jac_tensor, // [6, H, W]
    const float *__restrict__ v_loss,    // [3, H, W]
    float *__restrict__ v_depth,         // [H, W]
    float *__restrict__ v_tensors        // [6, H, W]
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
    const float v_tensor_term = consistent ? v_loss[HW + pix] : v_loss[2 * HW + pix];
    v_depth[pix] = consistent ? v_loss[pix] * jac_depth[pix] : 0.f;
#pragma unroll
    for (int c = 0; c < 6; ++c) {
        v_tensors[c * HW + pix] = v_tensor_term * jac_tensor[c * HW + pix];
    }
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

void check_direction_maps(const at::Tensor &depth, const at::Tensor &dirs) {
    SS_CHECK_FLOAT(depth);
    SS_CHECK_FLOAT(dirs);
    TORCH_CHECK(depth.dim() == 2, "depth must have shape [H, W]");
    TORCH_CHECK(dirs.dim() == 3 && dirs.size(0) == 3 && dirs.size(1) == depth.size(0) &&
                    dirs.size(2) == depth.size(1),
                "dirs must have shape [3, H, W]");
}

void check_tensor_maps(const at::Tensor &depth, const at::Tensor &tensors) {
    SS_CHECK_FLOAT(depth);
    SS_CHECK_FLOAT(tensors);
    TORCH_CHECK(depth.dim() == 2, "depth must have shape [H, W]");
    TORCH_CHECK(tensors.dim() == 3 && tensors.size(0) == 6 && tensors.size(1) == depth.size(0) &&
                    tensors.size(2) == depth.size(1),
                "tensors must have shape [6, H, W]");
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
    const at::Tensor &tensors,
    const at::Tensor &int_map,
    const at::Tensor &m_inv,
    const at::Tensor &p_col3,
    const at::Tensor &line_proj,
    const bool need_grad
) {
    check_tensor_maps(depth, tensors);
    SS_CHECK_FLOAT(int_map);
    TORCH_CHECK(int_map.dim() == 3 && int_map.size(0) == 3 && int_map.size(1) == depth.size(0) &&
                    int_map.size(2) == depth.size(1),
                "int_map must have shape [3, H, W]");
    const LineCamera cam = make_line_camera(m_inv, p_col3, line_proj);
    const at::cuda::OptionalCUDAGuard device_guard(depth.device());
    const int H = depth.size(0), W = depth.size(1);

    at::Tensor loss = at::zeros({3, H, W}, depth.options());
    at::Tensor jac_depth = need_grad ? at::zeros({H, W}, depth.options()) : at::Tensor();
    at::Tensor jac_tensor = need_grad ? at::zeros({6, H, W}, depth.options()) : at::Tensor();
    if (H * W > 0) {
        geometry_loss_fwd_kernel<<<image_grid(H, W), dim3(BLOCK_X, BLOCK_Y), 0, at::cuda::getCurrentCUDAStream()>>>(
            H, W, depth.data_ptr<float>(), tensors.data_ptr<float>(), int_map.data_ptr<float>(), cam,
            loss.data_ptr<float>(), need_grad ? jac_depth.data_ptr<float>() : nullptr,
            need_grad ? jac_tensor.data_ptr<float>() : nullptr
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {loss, jac_depth, jac_tensor};
}

std::tuple<at::Tensor, at::Tensor> geometry_loss_bwd(
    const at::Tensor &int_map,
    const at::Tensor &jac_depth,
    const at::Tensor &jac_tensor,
    const at::Tensor &v_loss
) {
    check_tensor_maps(jac_depth, jac_tensor);
    SS_CHECK_FLOAT(int_map);
    SS_CHECK_FLOAT(v_loss);
    TORCH_CHECK(int_map.dim() == 3 && int_map.size(0) == 3 && int_map.size(1) == jac_depth.size(0) &&
                    int_map.size(2) == jac_depth.size(1) && v_loss.sizes() == int_map.sizes(),
                "int_map and the loss gradient must have shape [3, H, W]");
    const at::cuda::OptionalCUDAGuard device_guard(v_loss.device());
    const int H = jac_depth.size(0), W = jac_depth.size(1);

    at::Tensor v_depth = at::empty({H, W}, v_loss.options());
    at::Tensor v_tensors = at::empty({6, H, W}, v_loss.options());
    if (H * W > 0) {
        geometry_loss_bwd_kernel<<<image_grid(H, W), dim3(BLOCK_X, BLOCK_Y), 0, at::cuda::getCurrentCUDAStream()>>>(
            H, W, int_map.data_ptr<float>(), jac_depth.data_ptr<float>(), jac_tensor.data_ptr<float>(),
            v_loss.data_ptr<float>(), v_depth.data_ptr<float>(), v_tensors.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {v_depth, v_tensors};
}

at::Tensor orientation_2d(
    const at::Tensor &depth,
    const at::Tensor &dirs,
    const at::Tensor &m_inv,
    const at::Tensor &p_col3,
    const at::Tensor &line_proj
) {
    check_direction_maps(depth, dirs);
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
