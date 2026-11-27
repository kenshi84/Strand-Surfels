"""Pure PyTorch reference of the strand surfel rasterizer.

Implements the same image formation model as the CUDA kernels (projection,
tile assignment, depth order, blending limits) with plain tensor operations, so
that autograd provides exact gradients to test the hand written backward
passes against. It is slow and only meant for small images in tests.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import math

import torch
from torch import Tensor

TILE = 16
NEAR, FAR = 0.2, 10.0
FILTER_SIZE, FILTER_INV_SQUARE = 0.707106, 2.0
MAX_ALPHA, MIN_ALPHA, MIN_T = 0.99, 1.0 / 255.0, 0.0001


def rotmat_unit(q: Tensor) -> Tensor:
    """Rotation matrices [N, 3, 3] of unit quaternions (w, x, y, z).

    The quaternions are assumed normalised (the CUDA kernel does not
    differentiate through the normalisation either)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def project(means, quats, scales, opacities, viewmat, K, width, height):
    """Reference of ``project_surfels_fwd``: radii, means2d, depths, ray transforms, dirs."""
    R_cam, t_cam = viewmat[:3, :3], viewmat[:3, 3]
    mean_c = means @ R_cam.T + t_cam
    R = rotmat_unit(quats)
    axis_u = (R[:, :, 0] * scales[:, 0:1]) @ R_cam.T
    axis_v = (R[:, :, 1] * scales[:, 1:2]) @ R_cam.T
    fx, cx, fy, cy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
    Tw = torch.stack([axis_u[:, 2], axis_v[:, 2], mean_c[:, 2]], -1)
    Tu = fx * torch.stack([axis_u[:, 0], axis_v[:, 0], mean_c[:, 0]], -1) + cx * Tw
    Tv = fy * torch.stack([axis_u[:, 1], axis_v[:, 1], mean_c[:, 1]], -1) + cy * Tw

    # bounding box (the opacity enters through a constant cut-off)
    cutoff = torch.sqrt(torch.clamp(9.0 + 2.0 * torch.log(opacities.detach()), min=1e-6))
    t = torch.stack([cutoff**2, cutoff**2, -torch.ones_like(cutoff)], -1)
    d = (t * Tw * Tw).sum(-1, keepdim=True)
    f = t / d
    center = torch.stack([(f * Tu * Tw).sum(-1), (f * Tv * Tw).sum(-1)], -1)
    half_sq = (
        center.detach() ** 2
        - torch.stack([(f * Tu * Tu).sum(-1), (f * Tv * Tv).sum(-1)], -1).detach()
    )
    half = torch.sqrt(torch.clamp(half_sq, min=1e-4))
    radius = torch.ceil(torch.maximum(half.max(-1).values, cutoff * FILTER_SIZE)).int()

    rect_min, rect_max = tile_rect(center.detach(), radius, width, height)
    visible = (mean_c[:, 2] > NEAR) & ((rect_max - rect_min).prod(-1) > 0)
    radii = torch.where(visible, radius, torch.zeros_like(radius))
    return radii, center, mean_c[:, 2].detach(), torch.cat([Tu, Tv, Tw], -1), R[:, :, 0]


def tile_rect(center: Tensor, radius: Tensor, width: int, height: int):
    """Tile rectangle [min, max) of each surfel (legacy integer-pixel frame)."""
    tiles = torch.tensor([math.ceil(width / TILE), math.ceil(height / TILE)], device=center.device)
    p = center - 0.5
    r = radius[:, None].float()
    lo = torch.clamp(torch.trunc((p - r) / TILE).int(), min=0)
    hi = torch.clamp(torch.trunc((p + r + TILE - 1.0) / TILE).int(), min=0)
    return torch.minimum(lo, tiles), torch.minimum(hi, tiles)


def ndc(depth: Tensor) -> Tensor:
    return FAR / (FAR - NEAR) * (1.0 - NEAR / depth)


def rasterize(
    radii, means2d, depths, ray_transforms, dirs, colors, opacities, pse, background, width, height
):
    """Reference: render [3,H,W], aux [10,H,W], error map [1,H,W]."""
    device = means2d.device
    rect_min, rect_max = tile_rect(means2d.detach(), radii, width, height)
    render = torch.zeros(3, height, width, device=device)
    aux = torch.zeros(10, height, width, device=device)
    error = torch.zeros(1, height, width, device=device)
    n_tx, n_ty = math.ceil(width / TILE), math.ceil(height / TILE)
    order = torch.argsort(depths, stable=True)  # front to back, ties by index

    for ty in range(n_ty):
        for tx in range(n_tx):
            in_tile = (
                (radii > 0)
                & (rect_min[:, 0] <= tx)
                & (tx < rect_max[:, 0])
                & (rect_min[:, 1] <= ty)
                & (ty < rect_max[:, 1])
            )
            ids = order[in_tile[order]]
            ys = torch.arange(ty * TILE, min((ty + 1) * TILE, height), device=device)
            xs = torch.arange(tx * TILE, min((tx + 1) * TILE, width), device=device)
            py, px = torch.meshgrid(ys.float() + 0.5, xs.float() + 0.5, indexing="ij")
            shape = py.shape
            T = torch.ones(shape, device=device)
            color = torch.zeros(3, *shape, device=device)
            tensor = torch.zeros(6, *shape, device=device)
            depth_sum = torch.zeros(shape, device=device)
            m1 = torch.zeros(shape, device=device)
            m2 = torch.zeros(shape, device=device)
            distortion = torch.zeros(shape, device=device)
            median = torch.zeros(shape, device=device)
            err = torch.zeros(shape, device=device)
            done = torch.zeros(shape, dtype=torch.bool, device=device)

            for g in ids.tolist():
                Tu, Tv, Tw = ray_transforms[g, 0:3], ray_transforms[g, 3:6], ray_transforms[g, 6:9]
                h_u = px[..., None] * Tw - Tu
                h_v = py[..., None] * Tw - Tv
                p = torch.cross(h_u, h_v, dim=-1)
                pz = torch.where(p[..., 2] == 0, torch.ones_like(p[..., 2]), p[..., 2])
                s = p[..., :2] / pz[..., None]
                rho3d = (s * s).sum(-1)
                dxy = means2d[g] - torch.stack([px, py], -1)
                rho2d = FILTER_INV_SQUARE * (dxy * dxy).sum(-1)
                rho = torch.minimum(rho3d, rho2d)
                depth = s[..., 0] * Tw[0] + s[..., 1] * Tw[1] + Tw[2]
                G = torch.exp(-0.5 * rho)
                alpha_raw = opacities[g] * G
                # clamp the value but pass the gradient (as the CUDA kernel)
                alpha = alpha_raw - torch.clamp_min(alpha_raw - MAX_ALPHA, 0.0).detach()
                valid = (
                    (~done) & (p[..., 2] != 0) & (depth >= NEAR) & (rho >= 0) & (alpha >= MIN_ALPHA)
                )
                next_T = T * (1 - alpha)
                saturate = valid & (next_T < MIN_T)
                done = done | saturate
                blend = valid & ~saturate

                w = torch.where(blend, alpha * T, torch.zeros_like(T))
                m = ndc(torch.where(blend, depth, torch.ones_like(depth)))
                distortion = distortion + (m * m * (1 - T) + m2 - 2 * m * m1) * w
                depth_sum = depth_sum + depth * w
                m1 = m1 + m * w
                m2 = m2 + m * m * w
                median = torch.where(blend & (T > 0.5), depth, median)
                dx, dy, dz = dirs[g]
                direction_tensor = torch.stack(
                    (dx * dx, dx * dy, dx * dz, dy * dy, dy * dz, dz * dz)
                )
                tensor = tensor + direction_tensor[:, None, None] * w
                color = color + colors[g][:, None, None] * w
                err = err + pse[g] * w
                T = torch.where(blend, next_T, T)

            sl = (slice(None), slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
            render[sl] = color + T * background[:, None, None]
            aux[sl] = torch.stack([depth_sum, 1 - T, *tensor, distortion, median])
            error[sl] = err[None]
    return render, aux, error
