"""Cameras of the viewer and conversions between OpenCV and Polyscope (OpenGL).

The reconstruction uses OpenCV cameras (x right, y down, looking along +z).
Polyscope uses OpenGL view matrices (x right, y up, looking along -z); both are
world-to-camera transforms and differ by a flip of the y and z axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

#: diag(1, -1, -1, 1): OpenCV <-> OpenGL camera axes
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


@dataclass
class ViewerCamera:
    """A pinhole camera (OpenCV conventions)."""

    name: str
    K: np.ndarray  # [3, 3] intrinsics
    viewmat: np.ndarray  # [4, 4] world-to-camera
    width: int
    height: int

    @property
    def fov_vertical_deg(self) -> float:
        """Vertical field of view in degrees."""
        return math.degrees(2.0 * math.atan(0.5 * self.height / float(self.K[1, 1])))

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        R, t = self.viewmat[:3, :3], self.viewmat[:3, 3]
        return -R.T @ t

    @classmethod
    def from_camera(cls, camera) -> "ViewerCamera":
        """Viewer camera of a :class:`strand_surfels.data.Camera`."""
        return cls(camera.name, camera.K, camera.viewmat, camera.width, camera.height)

    def polyscope_parameters(self):
        """``polyscope.CameraParameters`` of this camera (principal point ignored)."""
        import polyscope as ps  # pylint: disable=import-outside-toplevel

        return ps.CameraParameters(
            ps.CameraIntrinsics(
                fov_vertical_deg=self.fov_vertical_deg, aspect=self.width / self.height
            ),
            ps.CameraExtrinsics(mat=CV_TO_GL @ self.viewmat.astype(np.float32)),
        )


def opencv_from_polyscope(
    view_gl: np.ndarray, fov_vertical_deg: float, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV ``(K, viewmat)`` of the Polyscope view camera rendered at ``width x height``."""
    viewmat = (CV_TO_GL @ np.asarray(view_gl, dtype=np.float32)).astype(np.float32)
    focal = 0.5 * height / math.tan(0.5 * math.radians(fov_vertical_deg))
    K = np.array(
        [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return K, viewmat


def load_calibration(
    path: str, downscale: float = 1.0, *, strand_frame: bool = False, in_mm: bool = False
) -> List[ViewerCamera]:
    """Dataset cameras expressed in the coordinate frame of the displayed PLY."""
    # pylint: disable-next=import-outside-toplevel
    from strand_surfels.data.cameras import load_scene_calibration

    cameras, strands_to_world = load_scene_calibration(path, downscale)
    if strand_frame and strands_to_world is not None:
        for camera in cameras:
            camera.viewmat[:3, :3] = camera.R @ strands_to_world
    if in_mm:
        for camera in cameras:
            camera.viewmat[:3, 3] *= 1000.0
    return [ViewerCamera.from_camera(c) for c in cameras]


def up_direction(cameras: Sequence[ViewerCamera]) -> str:
    """Polyscope up direction (``"y_up"``, ``"neg_z_up"``, ...) closest to the
    average up vector of the cameras (``-y`` of OpenCV cameras)."""
    up = -np.mean([c.viewmat[1, :3] for c in cameras], axis=0)
    axis = int(np.argmax(np.abs(up)))
    return ("" if up[axis] > 0 else "neg_") + "xyz"[axis] + "_up"
