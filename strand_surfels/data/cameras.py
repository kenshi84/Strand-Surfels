"""Cameras of a dataset (OpenCV conventions) and the scene extent.

``calibration.json`` lists, per camera,

* ``camera_id``: e.g. ``"C000012"`` (the number identifies the image files),
* ``intrinsics.camera_matrix``: row-major 3x3 ``K``,
* ``intrinsics.resolution``: ``[height, width]``,
* ``extrinsics.view_matrix``: row-major 4x4 world-to-camera transform.

Cameras look along +z with x to the right and y down. Pixel ``(i, j)`` covers
``[j, j+1) x [i, i+1)`` in the coordinates of ``K``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from strand_surfels.losses.geometry import LineCamera, line_camera_matrices


@dataclass
class Camera:
    """A pinhole camera of the dataset."""

    uid: int  # number in the camera id and the image file names
    name: str  # image name, e.g. "img_000012"
    K: np.ndarray  # [3, 3] float32 intrinsics
    viewmat: np.ndarray  # [4, 4] float32 world-to-camera transform
    width: int
    height: int

    @property
    def R(self) -> np.ndarray:
        """World-to-camera rotation."""
        return self.viewmat[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """World-to-camera translation."""
        return self.viewmat[:3, 3]

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates, ``-R^T t``."""
        return -self.R.T @ self.t

    @property
    def fov(self) -> Tuple[float, float]:
        """Horizontal and vertical field of view in radians."""
        return (
            2.0 * math.atan(0.5 * self.width / float(self.K[0, 0])),
            2.0 * math.atan(0.5 * self.height / float(self.K[1, 1])),
        )

    def line_camera(self) -> LineCamera:
        """Matrices of the geometry loss (see :func:`line_camera_matrices`)."""
        return line_camera_matrices(self.K, self.viewmat)

    def tensors(self, device: torch.device | str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """``(viewmat, K)`` as float32 tensors on ``device``, as the rasterizer takes them."""
        return (
            torch.from_numpy(self.viewmat).to(device),
            torch.from_numpy(self.K).to(device),
        )


def load_calibration(path: str, downscale: float = 1.0) -> List[Camera]:
    """Cameras of a ``calibration.json`` file, with images downscaled by ``downscale``.

    Image sizes are rounded; the intrinsics are divided by ``downscale``.
    """
    cameras, _ = load_scene_calibration(path, downscale)
    return cameras


def load_scene_calibration(
    path: str, downscale: float = 1.0
) -> tuple[List[Camera], np.ndarray | None]:
    """Read the world-space cameras and optional rotation from strand to world coordinates."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rotation = None
    if "strands_to_world_rotation" in data:
        rotation = np.asarray(data["strands_to_world_rotation"], dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ):
            raise ValueError(f"{path}: strands_to_world_rotation must be a 3x3 rotation matrix")
    cameras = []
    for entry in data["cameras"]:
        uid = int(entry["camera_id"].lstrip("C"))
        K = np.array(entry["intrinsics"]["camera_matrix"], dtype=np.float32).reshape(3, 3)
        K[0, 0] /= downscale
        K[0, 2] /= downscale
        K[1, 1] /= downscale
        K[1, 2] /= downscale
        height, width = entry["intrinsics"]["resolution"]
        viewmat = np.array(entry["extrinsics"]["view_matrix"], dtype=np.float32).reshape(4, 4)
        cameras.append(
            Camera(
                uid=uid,
                name=f"img_{uid:06d}",
                K=K,
                viewmat=viewmat,
                width=round(width / downscale),
                height=round(height / downscale),
            )
        )
    return cameras, rotation


def camera_extent(cameras: Sequence[Camera]) -> Tuple[np.ndarray, float]:
    """Centre and radius of the camera rig, as in NeRF++ / 3D Gaussian Splatting.

    Returns the mean camera centre and 1.1 times the largest distance of a
    camera centre from it. The radius sets the spatial scale of the
    optimisation (position learning rate, densification thresholds).
    """
    centers = np.stack([c.center for c in cameras]).astype(np.float64)
    mean = centers.mean(axis=0)
    radius = 1.1 * float(np.linalg.norm(centers - mean, axis=1).max())
    return mean, radius
