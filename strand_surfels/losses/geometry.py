"""Geometry loss against the line-based multi-view stereo result (SMHC).

The first stage of the pipeline (see the README) reconstructs, per view, an
*int map* ``[3, H, W]``:

* multi-view consistent pixels store ``(depth, dir.x, dir.y)`` of a 3D strand
  point (``dir.z >= 0`` is implied by the unit length);
* all other hair pixels store ``(-1, angle, confidence)``: the 2D strand
  orientation measured in the image and its confidence.

:func:`geometry_loss` compares the rendered depth and strand direction with
these measurements and returns a per-pixel loss map with three channels
(:data:`GEOMETRY_LOSS_CHANNELS`): the 3D depth and direction errors on the
consistent pixels and the confidence weighted 2D orientation error of the
projected rendered strands elsewhere. Forward and backward are fused CUDA
kernels (``strand_surfels/csrc/geometry_loss.cu``).

2D orientations are computed with Pluecker line coordinates: the 3D strand line
through the back-projected pixel is projected with the *line projection matrix*
of the camera (Hartley and Zisserman, Sec. 8.1.2), see
:func:`line_camera_matrices`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from strand_surfels._backend import _C

# torch.autograd.Function subclasses override forward/backward with explicit
# signatures by design
# pylint: disable=abstract-method,arguments-differ

#: Channels of the loss map returned by :func:`geometry_loss` (index -> name).
GEOMETRY_LOSS_CHANNELS = ("depth_3d", "direction_3d", "orientation_2d")


class LineCamera(NamedTuple):
    """Camera matrices used by the geometry loss (float32 tensors).

    With the projection ``P = K [R | t] = [M | P_col3]``:

    * ``m_inv`` ``[3, 3]``: ``M^-1``, to back-project pixels,
    * ``p_col3`` ``[3]``: last column of ``P``,
    * ``line_proj`` ``[3, 6]``: line projection matrix acting on Pluecker
      coordinates in reversed order (see :func:`line_camera_matrices`).
    """

    m_inv: Float[Tensor, "3 3"]
    p_col3: Float[Tensor, "3"]
    line_proj: Float[Tensor, "3 6"]

    def to(self, device: torch.device | str) -> "LineCamera":
        """Copy of the camera with all matrices on ``device``."""
        return LineCamera(
            self.m_inv.to(device).contiguous(),
            self.p_col3.to(device).contiguous(),
            self.line_proj.to(device).contiguous(),
        )


def _pluecker(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pluecker coordinates (direction, moment) of the line joining the
    homogeneous points (or, dually, the intersection of the planes) ``a`` and
    ``b`` (4-vectors)."""
    out = np.zeros(6, dtype=np.float32)
    out[0] = a[3] * b[0] - b[3] * a[0]
    out[1] = a[3] * b[1] - b[3] * a[1]
    out[2] = a[3] * b[2] - b[3] * a[2]
    out[3] = a[1] * b[2] - b[1] * a[2]
    out[4] = a[2] * b[0] - b[2] * a[0]
    out[5] = a[0] * b[1] - b[0] * a[1]
    return out


def line_camera_matrices(K: np.ndarray, viewmat: np.ndarray) -> LineCamera:
    """Geometry loss matrices of a camera.

    Args:
        K: ``[3, 3]`` intrinsics (OpenCV).
        viewmat: ``[4, 4]`` world-to-camera transform (OpenCV).

    The computation runs in float32 numpy, as in the original implementation,
    so that the matrices are reproduced exactly.
    """
    K = np.asarray(K, dtype=np.float32)
    viewmat = np.asarray(viewmat, dtype=np.float32)
    P = K @ np.eye(3, 4, dtype=np.float32) @ viewmat  # [3, 4]
    M_inv = np.linalg.inv(P[:, :3].copy())
    # rows: lines of the planes P[1] ^ P[2], P[2] ^ P[0], P[0] ^ P[1], each
    # with the order of its coordinates reversed
    line_proj = np.stack(
        [
            np.flip(_pluecker(P[1], P[2])),
            np.flip(_pluecker(P[2], P[0])),
            np.flip(_pluecker(P[0], P[1])),
        ]
    )
    return LineCamera(
        m_inv=torch.from_numpy(M_inv).float(),
        p_col3=torch.from_numpy(P[:, 3].copy()).float(),
        line_proj=torch.from_numpy(np.ascontiguousarray(line_proj)).float(),
    )


class _GeometryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, dirs, int_map, m_inv, p_col3, line_proj):
        need_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        loss, jac_depth, jac_dir = _C.geometry_loss_fwd(
            depth, dirs, int_map, m_inv, p_col3, line_proj, need_grad
        )
        if need_grad:
            ctx.save_for_backward(int_map, jac_depth, jac_dir)
        return loss

    @staticmethod
    def backward(ctx, v_loss):
        int_map, jac_depth, jac_dir = ctx.saved_tensors
        v_depth, v_dirs = _C.geometry_loss_bwd(int_map, jac_depth, jac_dir, v_loss.contiguous())
        return v_depth, v_dirs, None, None, None, None


def geometry_loss(
    depth: Float[Tensor, "H W"],
    dirs: Float[Tensor, "3 H W"],
    int_map: Float[Tensor, "3 H W"],
    camera: LineCamera,
) -> Float[Tensor, "3 H W"]:
    """Per-pixel geometry loss map (channels :data:`GEOMETRY_LOSS_CHANNELS`).

    Args:
        depth: rendered depth (expected depth, i.e. normalised by alpha, or
            median depth).
        dirs: rendered world space strand direction, normalised by alpha.
        int_map: int map of the view (see the module documentation).
        camera: matrices of the view on the same device.

    Gradients flow to ``depth`` (3D depth term) and ``dirs`` (3D direction and
    2D orientation terms). Weighting, masking and reduction are left to the
    caller.
    """
    return _GeometryLoss.apply(depth.contiguous(), dirs.contiguous(), int_map.contiguous(), *camera)


@torch.no_grad()
def orientation_2d(
    depth: Float[Tensor, "H W"], dirs: Float[Tensor, "3 H W"], camera: LineCamera
) -> Float[Tensor, "H W"]:
    """Image space orientation angle in ``[-pi/2, pi/2]`` of the rendered strands.

    Each pixel's strand (through the back-projected pixel along ``dirs``) is
    projected into the image; the angle is the one of the resulting 2D line,
    in the convention of the measured orientations in the int maps.
    """
    return _C.orientation_2d(depth.contiguous(), dirs.contiguous(), *camera)


def orientation_to_rgb(angle: Float[Tensor, "H W"]) -> Float[Tensor, "3 H W"]:
    """Color coding of 2D orientations: ``((cos + 1) / 2, (sin + 1) / 2, 0)``."""
    rgb = torch.zeros(3, *angle.shape, dtype=angle.dtype, device=angle.device)
    rgb[0] = (torch.cos(angle) + 1.0) * 0.5
    rgb[1] = (torch.sin(angle) + 1.0) * 0.5
    return rgb
