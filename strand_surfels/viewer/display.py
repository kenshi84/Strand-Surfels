"""Display images of the rendered surfels (independent of the GUI).

:func:`display_image` turns a :class:`~strand_surfels.rendering.RasterOutput`
into a premultiplied RGBA image and a depth image for Polyscope, in one of the
:class:`DisplayMode` visualisations:

* ``RGB``: the rendered colors,
* ``ALPHA``: the accumulated opacity,
* ``DEPTH``: the expected (or median) depth, color mapped,
* ``DIRECTION_3D``: the world space strand direction, ``|d|`` as RGB (sign
  agnostic) or ``(d + 1) / 2``,
* ``ORIENTATION_2D``: the image space orientation of the projected strands
  (the quantity compared with the measured orientations in the geometry loss),
* ``LABEL``: the hair labels (a later addition), from blue (other) to orange
  (hair); rendered by the viewer with the label colors in place of the SH
  colors.

All images are ``[H, W, C]`` CUDA tensors, row 0 at the top.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from strand_surfels.losses.geometry import line_camera_matrices, orientation_2d, orientation_to_rgb
from strand_surfels.rendering import RasterOutput


class DisplayMode(enum.Enum):
    """Visualisations of the viewer."""

    RGB = "RGB"
    ALPHA = "Alpha"
    DEPTH = "Depth"
    DIRECTION_3D = "Direction 3D"
    ORIENTATION_2D = "Orientation 2D"
    LABEL = "Hair label"


#: Surfel filters by hair label (:attr:`DisplaySettings.label_filter`).
LABEL_FILTERS = ("All surfels", "Hair only", "Other only")


# Viridis colormap (matplotlib, CC0), 17 samples, linearly interpolated.
_VIRIDIS = (
    (0.267004, 0.004874, 0.329415),
    (0.282327, 0.094955, 0.417331),
    (0.278826, 0.175490, 0.483397),
    (0.258965, 0.251537, 0.524736),
    (0.229739, 0.322361, 0.545706),
    (0.199430, 0.387607, 0.554642),
    (0.172719, 0.448791, 0.557885),
    (0.149039, 0.508051, 0.557250),
    (0.127568, 0.566949, 0.550556),
    (0.120638, 0.625828, 0.533488),
    (0.157851, 0.683765, 0.501686),
    (0.246070, 0.738910, 0.452024),
    (0.369214, 0.788888, 0.382914),
    (0.515992, 0.831158, 0.294279),
    (0.678489, 0.863742, 0.189503),
    (0.845561, 0.887322, 0.099702),
    (0.993248, 0.906157, 0.143936),
)


def colormap(values: Tensor) -> Tensor:
    """Viridis colors ``[..., 3]`` of values in ``[0, 1]`` (clamped)."""
    lut = torch.tensor(_VIRIDIS, dtype=values.dtype, device=values.device)
    x = values.clamp(0.0, 1.0) * (len(_VIRIDIS) - 1)
    lo = x.floor().long().clamp(max=len(_VIRIDIS) - 2)
    t = (x - lo)[..., None]
    return lut[lo] * (1 - t) + lut[lo + 1] * t


def hue_to_rgb(hue: Tensor) -> Tensor:
    """Fully saturated colors ``[..., 3]`` of hues in ``[0, 1)``."""
    k = (torch.stack([hue + 0.0, hue - 1 / 3, hue + 1 / 3], dim=-1) * 6.0) % 6.0
    return (2.0 - (k - 3.0).abs()).clamp(0.0, 1.0)  # piecewise linear hue wheel


@dataclass
class DisplaySettings:
    """Options of :func:`display_image`."""

    mode: DisplayMode = DisplayMode.RGB
    #: blend semi-transparent pixels with the Polyscope scene behind them
    #: instead of with the background color
    composite: bool = False
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: pixels with less accumulated opacity show the background
    alpha_threshold: float = 0.05
    median_depth: bool = False
    #: depth color range; None: 1st to 99th percentile of the visible pixels
    depth_range: Optional[Tuple[float, float]] = None
    direction_abs: bool = True
    #: 2D orientation colors: ((cos + 1) / 2, (sin + 1) / 2, 0) as in the
    #: training logs, or a hue wheel periodic in the (undirected) angle
    orientation_hue: bool = False
    #: one of LABEL_FILTERS; hair: label probability above label_threshold
    label_filter: str = LABEL_FILTERS[0]
    label_threshold: float = 0.5
    last_depth_range: Tuple[float, float] = field(default=(0.0, 1.0), compare=False)


def _ray_lengths(K: np.ndarray, width: int, height: int, device) -> Tensor:
    """|K^-1 (u, v, 1)| at the pixel centres: converts z depth to ray distance."""
    u = (torch.arange(width, device=device, dtype=torch.float32) + 0.5 - float(K[0, 2])) / float(
        K[0, 0]
    )
    v = (torch.arange(height, device=device, dtype=torch.float32) + 0.5 - float(K[1, 2])) / float(
        K[1, 1]
    )
    return torch.sqrt(1.0 + v[:, None] ** 2 + u[None, :] ** 2)


def _percentile_range(values: Tensor, lo: float = 0.01, hi: float = 0.99) -> Tuple[float, float]:
    if values.numel() == 0:
        return 0.0, 1.0
    if values.numel() > 1 << 20:  # quantile limit; a random subset suffices
        values = values[torch.randint(values.numel(), (1 << 20,), device=values.device)]
    q = torch.quantile(values, torch.tensor([lo, hi], device=values.device))
    a, b = q.tolist()
    return a, max(b, a + 1e-6)


@torch.no_grad()
def display_image(
    out: RasterOutput, settings: DisplaySettings, K: np.ndarray, viewmat: np.ndarray
) -> Tuple[Tensor, Tensor]:
    """Premultiplied RGBA ``[H, W, 4]`` and ray distance ``[H, W]`` (``inf``
    where nothing is drawn: opacity below the threshold) of a rendering
    without background (black).

    ``K`` and ``viewmat`` are the (OpenCV) camera of the rendering.
    """
    aux = out.aux
    H, W = aux.shape[-2:]
    alpha = aux[1]
    covered = alpha > settings.alpha_threshold
    safe_alpha = torch.where(alpha > 0, alpha, torch.ones_like(alpha))
    depth = aux[6] if settings.median_depth else aux[0] / safe_alpha
    dirs = aux[2:5] / safe_alpha

    mode = settings.mode
    if mode in (DisplayMode.RGB, DisplayMode.LABEL):
        rgb = out.render.permute(1, 2, 0)  # already weighted by alpha
    elif mode == DisplayMode.ALPHA:
        rgb = alpha[..., None].expand(H, W, 3)
    elif mode == DisplayMode.DEPTH:
        lo, hi = settings.depth_range or _percentile_range(depth[covered])
        settings.last_depth_range = (lo, hi)
        rgb = colormap((depth - lo) / (hi - lo)) * alpha[..., None]
    elif mode == DisplayMode.DIRECTION_3D:
        d = torch.nn.functional.normalize(dirs, dim=0).permute(1, 2, 0)
        rgb = (d.abs() if settings.direction_abs else (d + 1.0) * 0.5) * alpha[..., None]
    elif mode == DisplayMode.ORIENTATION_2D:
        camera = line_camera_matrices(K, viewmat).to(aux.device)
        angle = orientation_2d(depth.contiguous(), dirs.contiguous(), camera)
        if settings.orientation_hue:
            rgb = hue_to_rgb((angle + 0.5 * math.pi) / math.pi)
        else:
            rgb = orientation_to_rgb(angle).permute(1, 2, 0)
        rgb = rgb * alpha[..., None]
    else:
        raise ValueError(f"unknown display mode {mode}")

    # the alpha view shows coverage, so it is opaque wherever it is drawn
    opacity = torch.ones_like(alpha) if mode == DisplayMode.ALPHA else alpha
    if settings.composite:
        rgba = torch.cat([rgb, opacity[..., None]], dim=-1)
    else:
        bg = torch.tensor(settings.background, dtype=rgb.dtype, device=rgb.device)
        rgb = rgb + (1.0 - opacity)[..., None] * bg
        rgba = torch.cat([rgb, torch.ones_like(alpha)[..., None]], dim=-1)

    # Polyscope composites by ray distance and does not draw pixels at
    # infinity: those show its background (set to the same color by the viewer)
    ray_depth = torch.where(covered, depth, torch.full_like(depth, float("inf")))
    ray_depth = ray_depth * _ray_lengths(K, W, H, aux.device)
    return rgba.contiguous(), ray_depth.contiguous()
