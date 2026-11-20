"""Differentiable rendering of strand surfels."""

from strand_surfels.rendering.rasterization import (
    AUX_CHANNELS,
    RasterOutput,
    SampleBuffer,
    rasterize_surfels,
)
from strand_surfels.rendering.sh import eval_sh, rgb_to_sh, sh_to_color
from strand_surfels.rendering.surfels import Surfels

__all__ = [
    "AUX_CHANNELS",
    "RasterOutput",
    "SampleBuffer",
    "Surfels",
    "eval_sh",
    "rasterize_surfels",
    "rgb_to_sh",
    "sh_to_color",
]
