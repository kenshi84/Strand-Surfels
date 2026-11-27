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
    direction_to_structure_tensor,
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
    return (
        depth.cuda(),
        dirs.cuda(),
        direction_to_structure_tensor(dirs).cuda(),
        int_map.cuda(),
        cam.to("cuda"),
    )


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


def quadratic(tensors, vectors):
    x, y, z = vectors
    return (
        tensors[0] * x * x
        + 2 * tensors[1] * x * y
        + 2 * tensors[2] * x * z
        + tensors[3] * y * y
        + 2 * tensors[4] * y * z
        + tensors[5] * z * z
    )


def basis(vectors):
    x, y, z = vectors
    return torch.stack((x * x, 2 * x * y, 2 * x * z, y * y, 2 * y * z, z * z))


def reference(depth, tensors, int_map, cam):
    """Sign-invariant tensor loss and its per-pixel derivatives."""
    consistent = int_map[0] > 0
    tensor_mask = (tensors[[0, 3, 5]].sum(0) > 1e-6).float()
    x, y = int_map[1], int_map[2]
    mvs = torch.stack([x, y, torch.nan_to_num(torch.sqrt(1 - x * x - y * y), nan=0.0)])
    mvs = mvs * torch.where((mvs * mvs).sum(0) > 1e-6, torch.rsqrt((mvs * mvs).sum(0)), 0.0)

    ys, xs = torch.meshgrid(
        torch.arange(H, device="cuda").float(),
        torch.arange(W, device="cuda").float(),
        indexing="ij",
    )
    ori, conf = int_map[1], int_map[2]
    ray0 = torch.einsum("ij,jhw->ihw", cam.m_inv, torch.stack((xs, ys, torch.ones_like(xs))))
    ray1 = torch.einsum(
        "ij,jhw->ihw",
        cam.m_inv,
        torch.stack((xs + torch.cos(ori), ys + torch.sin(ori), torch.ones_like(xs))),
    )
    normal = torch.nn.functional.normalize(torch.cross(ray0, ray1, dim=0), dim=0)
    q_2d = quadratic(tensors, normal).clamp_min(0.0)
    root_2d = torch.sqrt(q_2d + 1e-6)

    zero = torch.zeros_like(depth)
    loss = torch.stack(
        [
            torch.where(consistent, (depth - int_map[0]).abs(), zero),
            torch.where(consistent, (1 - quadratic(tensors, mvs)) * tensor_mask, zero),
            torch.where(
                consistent, zero, (root_2d - 1e-3) * conf / 10.0 * tensor_mask
            ),
        ]
    )
    jac_tensor = torch.where(
        consistent.unsqueeze(0),
        -basis(mvs),
        basis(normal) * conf.unsqueeze(0) / (20.0 * root_2d.unsqueeze(0)),
    ) * tensor_mask.unsqueeze(0)
    jac_depth = torch.where(consistent, torch.sign(depth - int_map[0]), zero)
    return loss, jac_depth, jac_tensor


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_geometry_loss_matches_reference(seed):
    depth, _, tensors, int_map, cam = random_view(seed)
    up = torch.randn(3, H, W, device="cuda", generator=torch.Generator("cuda").manual_seed(seed))
    d = depth.clone().requires_grad_(True)
    r = tensors.clone().requires_grad_(True)
    loss = geometry_loss(d, r, int_map, cam)
    (loss * up).sum().backward()

    ref_loss, jac_depth, jac_tensor = reference(depth, tensors, int_map, cam)
    consistent = int_map[0] > 0
    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(d.grad, up[0] * jac_depth)
    v_term = torch.where(consistent, up[1], up[2])
    # The Charbonnier derivative contains 1 / sqrt(q + eps^2); close to exact
    # alignment it amplifies the small CUDA/PyTorch rounding difference in the
    # viewing-plane normal. The dedicated 3D test below retains tighter bounds.
    torch.testing.assert_close(r.grad, v_term * jac_tensor, atol=1e-4, rtol=2e-2)


def test_3d_terms_match_autograd():
    depth, _, tensors, int_map, cam = random_view(3)
    int_map[0] = int_map[0].abs() + 1.0  # all pixels consistent
    xy = torch.nn.functional.normalize(torch.randn(3, H, W, device="cuda"), dim=0)
    int_map[1:] = (xy * torch.where(xy[2:3] < 0, -1.0, 1.0))[:2]
    d = depth.clone().requires_grad_(True)
    r = tensors.clone().requires_grad_(True)
    geometry_loss(d, r, int_map, cam)[:2].sum().backward()

    d2 = depth.clone().requires_grad_(True)
    r2 = tensors.clone().requires_grad_(True)
    ref_loss, _, _ = reference(d2, r2, int_map, cam)
    ref_loss[:2].sum().backward()
    torch.testing.assert_close(d.grad, d2.grad)
    torch.testing.assert_close(r.grad, r2.grad, atol=1e-5, rtol=1e-4)


def test_structure_tensor_is_direction_sign_invariant():
    dirs = torch.nn.functional.normalize(torch.randn(3, H, W, device="cuda"), dim=0)
    signs = torch.where(torch.rand(1, H, W, device="cuda") > 0.5, 1.0, -1.0)
    torch.testing.assert_close(
        direction_to_structure_tensor(dirs), direction_to_structure_tensor(dirs * signs)
    )


def test_2d_term_with_large_focal_length_is_not_masked():
    K = np.array([[2000.0, 0.0, W / 2], [0.0, 2000.0, H / 2], [0.0, 0.0, 1.0]])
    camera = line_camera_matrices(K, np.eye(4, dtype=np.float32)).to("cuda")
    depth = torch.ones(H, W, device="cuda")
    dirs = torch.zeros(3, H, W, device="cuda")
    dirs[1] = 1.0  # orthogonal to the viewing plane of a horizontal image line
    int_map = torch.empty(3, H, W, device="cuda")
    int_map[0] = -1.0
    int_map[1] = 0.0
    int_map[2] = 1.0
    loss = geometry_loss(depth, direction_to_structure_tensor(dirs), int_map, camera)
    expected = (math.sqrt(1.0 + 1e-6) - 1e-3) / 10.0
    torch.testing.assert_close(
        loss[2], torch.full_like(depth, expected), atol=1e-5, rtol=1e-5
    )


def test_orientation_2d_matches_reference():
    depth, dirs, _, _, cam = random_view(4)
    norm_sq = (dirs * dirs).sum(0, keepdim=True)
    unit = dirs * torch.where(norm_sq > 1e-6, torch.rsqrt(norm_sq), 0.0)
    ref, _, _ = line_angle_reference(depth, unit, cam)
    torch.testing.assert_close(orientation_2d(depth, dirs, cam), ref, atol=1e-4, rtol=1e-4)


def test_orientation_of_projected_line_is_its_image_angle():
    """The angle is that of the image segment between two points of the 3D line."""
    depth, dirs, _, _, cam = random_view(5)
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
