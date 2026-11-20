"""Geometry loss kernels against a PyTorch reference."""

# pylint: disable=missing-function-docstring

import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

# pylint: disable=wrong-import-position
from strand_surfels.losses import (  # noqa: E402
    geometry_loss,
    line_camera_matrices,
    orientation_2d,
)

H, W = 24, 30


def random_view(seed: int):
    """Rendered maps, an int map with both pixel kinds and a camera."""
    g = torch.Generator().manual_seed(seed)
    K = np.array([[40.0, 0.0, W / 2], [0.0, 42.0, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    angle = 0.3 + 0.1 * seed
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = [
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ]
    viewmat[:3, 3] = [0.1, -0.2, 2.5]
    depth = torch.empty(H, W).uniform_(1.5, 3.5, generator=g)
    dirs = torch.randn(3, H, W, generator=g) * 0.7
    dirs[:, :3, :3] = 0.0  # pixels without a rendered strand

    consistent = torch.rand(H, W, generator=g) < 0.4
    mvs_dir = torch.nn.functional.normalize(torch.randn(3, H, W, generator=g), dim=0)
    mvs_dir = mvs_dir * torch.where(mvs_dir[2:3] < 0, -1.0, 1.0)  # stored with z >= 0
    int_map = torch.stack(
        [
            torch.where(consistent, torch.empty(H, W).uniform_(1.5, 3.5, generator=g), -1.0),
            torch.where(consistent, mvs_dir[0], torch.empty(H, W).uniform_(-1.5, 1.5, generator=g)),
            torch.where(consistent, mvs_dir[1], torch.empty(H, W).uniform_(0.0, 5.0, generator=g)),
        ]
    )
    cam = line_camera_matrices(K, viewmat)
    return depth.cuda(), dirs.cuda(), int_map.cuda(), cam.to("cuda")


def line_angle_reference(depth, dirs, cam):
    """Back-projection, Pluecker line projection and angle; returns (angle, (a, b), point)."""
    ys, xs = torch.meshgrid(
        torch.arange(H, device="cuda").float(),
        torch.arange(W, device="cuda").float(),
        indexing="ij",
    )
    pix = torch.stack([depth * xs, depth * ys, depth]) - cam.p_col3[:, None, None]
    point = torch.einsum("ij,jhw->ihw", cam.m_inv, pix)
    moment = torch.cross(point, dirs, dim=0)
    pluecker = torch.cat([moment.flip(0), dirs.flip(0)])  # reversed order
    ab = torch.einsum("ij,jhw->ihw", cam.line_proj[:2], pluecker)
    return torch.atan(-ab[0] / (ab[1] + 1e-7)), ab, point


def reference(depth, dirs, int_map, cam):
    """Loss map and per-pixel derivatives (d/d depth, d/d dirs of the active term)."""
    consistent = int_map[0] > 0
    dir_mask = ((dirs * dirs).sum(0) > 1e-6).float()
    x, y = int_map[1], int_map[2]
    mvs = torch.stack([x, y, torch.nan_to_num(torch.sqrt(1 - x * x - y * y), nan=0.0)])
    mvs = mvs * torch.where((mvs * mvs).sum(0) > 1e-6, torch.rsqrt((mvs * mvs).sum(0)), 0.0)
    mvs = torch.where((mvs * dirs).sum(0) < 0, -mvs, mvs)

    angle, ab, point = line_angle_reference(depth, dirs, cam)
    ori, conf = int_map[1], int_map[2]
    wrap = (angle - ori).abs() >= math.pi / 2
    ori = torch.where(wrap, torch.where(ori >= 0, ori - math.pi, ori + math.pi), ori)
    sign = torch.where(angle >= ori, 1.0, -1.0)
    sign = torch.where(~wrap & (ori >= angle), -1.0, sign)
    cost = sign * (angle - ori) * conf / 10.0

    zero = torch.zeros_like(depth)
    loss = torch.stack(
        [
            torch.where(consistent, (depth - int_map[0]).abs(), zero),
            torch.where(consistent, (1 - (dirs * mvs).sum(0)) * dir_mask, zero),
            torch.where(consistent, zero, cost * dir_mask),
        ]
    )
    # derivative of the 2D term (moment contribution taken component-wise)
    d_angle = sign * conf / 10.0
    ab_sq = ab[0] ** 2 + ab[1] ** 2 + 1e-7
    v_ab = torch.stack([d_angle * -ab[1] / ab_sq, d_angle * ab[0] / ab_sq])
    v_pluecker = torch.einsum("ij,ihw->jhw", cam.line_proj[:2], v_ab)
    v_moment, v_dir = v_pluecker[:3].flip(0), v_pluecker[3:].flip(0)
    p = point
    v_dir = v_dir + v_moment * torch.stack([p[2] - p[1], p[0] - p[2], p[1] - p[0]])
    jac_dir = torch.where(consistent, -mvs * dir_mask, v_dir * dir_mask)
    jac_depth = torch.where(consistent, torch.sign(depth - int_map[0]), zero)
    return loss, jac_depth, jac_dir


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_geometry_loss_matches_reference(seed):
    depth, dirs, int_map, cam = random_view(seed)
    up = torch.randn(3, H, W, device="cuda", generator=torch.Generator("cuda").manual_seed(seed))
    d = depth.clone().requires_grad_(True)
    r = dirs.clone().requires_grad_(True)
    loss = geometry_loss(d, r, int_map, cam)
    (loss * up).sum().backward()

    ref_loss, jac_depth, jac_dir = reference(depth, dirs, int_map, cam)
    consistent = int_map[0] > 0
    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(d.grad, up[0] * jac_depth)
    v_term = torch.where(consistent, up[1], up[2])
    torch.testing.assert_close(r.grad, v_term * jac_dir, atol=1e-4, rtol=1e-3)


def test_3d_terms_match_autograd():
    depth, dirs, int_map, cam = random_view(3)
    int_map[0] = int_map[0].abs() + 1.0  # all pixels consistent
    xy = torch.nn.functional.normalize(torch.randn(3, H, W, device="cuda"), dim=0)
    int_map[1:] = (xy * torch.where(xy[2:3] < 0, -1.0, 1.0))[:2]
    d = depth.clone().requires_grad_(True)
    r = dirs.clone().requires_grad_(True)
    geometry_loss(d, r, int_map, cam)[:2].sum().backward()

    d2 = depth.clone().requires_grad_(True)
    r2 = dirs.clone().requires_grad_(True)
    ref_loss, _, _ = reference(d2, r2, int_map, cam)
    ref_loss[:2].sum().backward()
    torch.testing.assert_close(d.grad, d2.grad)
    torch.testing.assert_close(r.grad, r2.grad, atol=1e-5, rtol=1e-4)


def test_orientation_2d_matches_reference():
    depth, dirs, _, cam = random_view(4)
    norm_sq = (dirs * dirs).sum(0, keepdim=True)
    unit = dirs * torch.where(norm_sq > 1e-6, torch.rsqrt(norm_sq), 0.0)
    ref, _, _ = line_angle_reference(depth, unit, cam)
    torch.testing.assert_close(orientation_2d(depth, dirs, cam), ref, atol=1e-4, rtol=1e-4)


def test_orientation_of_projected_line_is_its_image_angle():
    """The angle is that of the image segment between two points of the 3D line."""
    depth, dirs, _, cam = random_view(5)
    angle = orientation_2d(depth, dirs, cam)
    m_inv, p_col3 = cam.m_inv.cpu().double(), cam.p_col3.cpu().double()
    M = np.linalg.inv(m_inv.numpy())
    y, x = 10, 17
    X = m_inv.numpy() @ (depth[y, x].item() * np.array([x, y, 1.0]) - p_col3.numpy())

    def proj(Q):
        return M @ Q + p_col3.numpy()

    a, b = proj(X), proj(X + 1e-3 * dirs[:, y, x].cpu().double().numpy())
    du, dv = b[0] / b[2] - a[0] / a[2], b[1] / b[2] - a[1] / a[2]
    expected = math.atan(dv / du)  # image line direction, modulo pi
    diff = (angle[y, x].item() - expected + math.pi / 2) % math.pi - math.pi / 2
    assert abs(diff) < 1e-3
