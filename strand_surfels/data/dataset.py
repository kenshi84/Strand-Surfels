"""Multi-view hair dataset: images, masks and int maps of every camera.

Layout (see the README, *Data layout*)::

    data/<scene>/calibration.json
    data/<scene>/image/img_XXXXXX.png    RGB images
    data/<scene>/hair/img_XXXXXX.png     hair masks (optional)
    data/<scene>/body/img_XXXXXX.png     foreground masks (optional)
    output/<scene>/int/int_map_XXXXXX.npy   SMHC int maps (optional)
    output/<scene>/filtered/...             SMHC line samples (initial points)

Image names use 4 to 6 digit camera numbers (``img_000012.png``). Masks are
binarised at 0.01. Without a foreground mask, the foreground is derived from
the image: pixels whose brightest channel exceeds ``foreground_threshold``
(by default: every pixel that is not exactly black, which suits scenes
rendered or photographed on a black background). Without a hair mask the whole
image is hair; without an int map no geometry measurement is available for the
view.

The views are kept in host memory (images as 8-bit, masks as booleans at the
original resolution) and moved to the GPU one at a time with
:meth:`View.load`.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from strand_surfels.data.cameras import Camera, camera_extent, load_scene_calibration
from strand_surfels.data.points import InitialPoints, load_initial_points
from strand_surfels.losses.geometry import LineCamera

logger = logging.getLogger(__name__)

MASK_THRESHOLD = 0.01


@dataclass
class ViewData:
    """Training data of one view on the GPU (float32, channel-first)."""

    camera: Camera
    viewmat: Tensor  # [4, 4]
    K: Tensor  # [3, 3]
    line_camera: LineCamera
    image: Tensor  # [3, H, W] in [0, 1], multiplied by the foreground mask
    alpha_mask: Tensor  # [1, H, W] foreground
    hair_mask: Tensor  # [1, H, W]
    int_map: Tensor  # [3, H, W]


@dataclass
class View:
    """A camera with its images and maps in host memory.

    Images and masks are stored as ``uint8`` / ``bool`` at full resolution
    and as float32 when downscaled (where resampling makes them fractional).
    """

    camera: Camera
    image: Tensor  # [H, W, 3] uint8, or [3, H, W] float32 (premultiplied by alpha)
    alpha_mask: Tensor  # [H, W] bool, or [1, H, W] float32
    hair_mask: Tensor  # [H, W] bool, or [1, H, W] float32
    int_map: Tensor  # [3, H, W] float32
    mask_image: bool = True  # multiply the (8-bit) image by the foreground mask

    def load(self, device: torch.device | str = "cuda") -> ViewData:
        """The view on ``device`` in float32."""
        line = self.camera.line_camera().to(device)
        viewmat, K = self.camera.tensors(device)
        if self.image.dtype == torch.uint8:
            alpha = self.alpha_mask.to(device, non_blocking=True).float()[None]
            image = self.image.to(device, non_blocking=True).permute(2, 0, 1).float() * (
                1.0 / 255.0
            )
            if self.mask_image:
                image = image * alpha
            hair = self.hair_mask.to(device, non_blocking=True).float()[None]
        else:
            alpha = self.alpha_mask.to(device, non_blocking=True)
            image = self.image.to(device, non_blocking=True)
            hair = self.hair_mask.to(device, non_blocking=True)
        return ViewData(
            camera=self.camera,
            viewmat=viewmat,
            K=K,
            line_camera=line,
            image=image.clamp(0.0, 1.0),
            alpha_mask=alpha.clamp(0.0, 1.0),
            hair_mask=hair.clamp(0.0, 1.0),
            int_map=self.int_map.to(device, non_blocking=True),
        )


def _find_image(folder: str, uid: int, prefix: str = "img_", ext: str = ".png") -> Optional[str]:
    for digits in (6, 5, 4):
        path = os.path.join(folder, f"{prefix}{uid:0{digits}d}{ext}")
        if os.path.exists(path):
            return path
    return None


def _load_mask(path: str) -> np.ndarray:
    """Binary mask (float32 0/1) of a grayscale image."""
    mask = np.array(Image.open(path).convert("L"), dtype=np.float32) * (1.0 / 255.0)
    return (mask > MASK_THRESHOLD).astype(np.float32)


def _resize(channel: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bicubic resize of a float32 image channel (PIL)."""
    resized = Image.fromarray(channel).resize((width, height), Image.Resampling.BICUBIC)
    return np.array(resized)


class HairDataset:
    """Multi-view hair dataset (see the module documentation).

    Args:
        data_dir: ``data/<scene>``.
        output_dir: ``output/<scene>`` with the SMHC results.
        downscale: image downscale factor (the ``images_resolution`` option).
        images_dir: image folder inside ``data_dir``.
        mask_images: multiply the images by the foreground masks.
        foreground_threshold: without foreground masks, pixels whose largest
            color value (0-255) exceeds this are foreground; ``None`` makes
            every pixel foreground.
        every_nth_camera: use only every n-th camera (all if <= 1).
        initial_points_stride: subsampling of the SMHC line samples.
        num_workers: threads used to read the files.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data_dir: str,
        output_dir: str,
        *,
        downscale: float = 1.0,
        images_dir: str = "image",
        calibration_file: str = "calibration.json",
        mask_images: bool = True,
        foreground_threshold: Optional[int] = 0,
        every_nth_camera: int = 1,
        initial_points_stride: int = 100,
        num_workers: int = 8,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.downscale = downscale
        self.images_dir = images_dir
        self.mask_images = mask_images
        self.foreground_threshold = foreground_threshold

        cameras, self.strands_to_world_rotation = load_scene_calibration(
            os.path.join(data_dir, calibration_file), downscale
        )
        if every_nth_camera > 1:
            cameras = cameras[::every_nth_camera]
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
            views = list(pool.map(self._load_view, cameras))
        self.views: List[View] = [v for v in views if v is not None]
        if not self.views:
            raise ValueError(f"no views could be read from {data_dir}")
        logger.info("Loaded %d of %d views", len(self.views), len(cameras))

        self.scene_center, self.cameras_extent = camera_extent([v.camera for v in self.views])
        self._initial_points_stride = initial_points_stride
        self._initial_points: Optional[InitialPoints] = None

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> View:
        return self.views[index]

    @property
    def cameras(self) -> List[Camera]:
        """Cameras of the loaded views."""
        return [v.camera for v in self.views]

    @property
    def initial_points(self) -> InitialPoints:
        """Initial surfel positions from the SMHC line samples (loaded lazily)."""
        if self._initial_points is None:
            self._initial_points = load_initial_points(self.output_dir, self._initial_points_stride)
        return self._initial_points

    # ----------------------------------------------------------------- loading

    def _load_view(self, camera: Camera) -> Optional[View]:  # pylint: disable=too-many-locals
        image_path = _find_image(os.path.join(self.data_dir, self.images_dir), camera.uid)
        if image_path is None:
            logger.warning("No image for camera %d, skipping it", camera.uid)
            return None
        image = np.array(Image.open(image_path).convert("RGB"))  # [H, W, 3] uint8
        height, width = image.shape[:2]
        full = (width, height) == (camera.width, camera.height)

        # foreground mask: body mask, else from the image colors, else everything
        body_path = _find_image(os.path.join(self.data_dir, "body"), camera.uid)
        if body_path is not None:
            alpha = _load_mask(body_path)
        elif self.foreground_threshold is not None:
            alpha = (image.max(axis=-1) > self.foreground_threshold).astype(np.float32)
        else:
            alpha = np.ones((height, width), dtype=np.float32)

        hair_path = _find_image(os.path.join(self.data_dir, "hair"), camera.uid)
        hair = _load_mask(hair_path) if hair_path else np.ones((height, width), np.float32)

        int_path = os.path.join(self.output_dir, "int", f"int_map_{camera.uid:06d}.npy")
        if os.path.exists(int_path):
            int_map = np.load(int_path).astype(np.float32)
        else:
            # no measurement: inconsistent pixels with zero orientation confidence
            int_map = np.zeros((height, width, 3), dtype=np.float32)
            int_map[..., 0] = -1.0

        if full:
            return View(
                camera=camera,
                image=torch.from_numpy(image),
                alpha_mask=torch.from_numpy(alpha > 0.5),
                hair_mask=torch.from_numpy(hair > 0.5),
                int_map=torch.from_numpy(int_map).permute(2, 0, 1).contiguous(),
                mask_image=self.mask_images,
            )

        # downscaled: resample float32 channels (the image after masking)
        w, h = camera.width, camera.height
        image_f = image.astype(np.float32) * (1.0 / 255.0)
        if self.mask_images:
            image_f = image_f * alpha[..., None]

        def stack(channels):
            return torch.from_numpy(np.stack([_resize(c, w, h) for c in channels]))

        return View(
            camera=camera,
            image=stack([image_f[..., i] for i in range(3)]),
            alpha_mask=stack([alpha]),
            hair_mask=stack([hair]),
            int_map=stack([int_map[..., i] for i in range(3)]),
        )
