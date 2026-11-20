"""Renderable surfels: activated surfel parameters plus a rendering shortcut."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from strand_surfels.rendering.rasterization import RasterOutput, SampleBuffer, rasterize_surfels


@dataclass
class Surfels:
    """Activated parameters of a set of 2D Gaussian surfels.

    The optimised (raw) parameters are mapped as in 3D Gaussian Splatting: unit
    quaternions by normalisation, scales by ``exp`` and opacities by a sigmoid
    (see :meth:`from_raw`). Colors are spherical harmonics coefficients, of
    which the first ``(sh_degree + 1)^2`` per channel are used.

    ``labels`` (a later addition, optional) is the probability of every surfel
    to belong to the hair, see :mod:`strand_surfels.losses.labels`.
    """

    means: Float[Tensor, "N 3"]
    quats: Float[Tensor, "N 4"]  # unit, (w, x, y, z)
    scales: Float[Tensor, "N 2"]
    opacities: Float[Tensor, "N"]
    sh_coeffs: Float[Tensor, "N K 3"]
    sh_degree: int
    labels: Optional[Float[Tensor, "N"]] = None  # hair probability

    @classmethod
    def from_raw(
        cls,
        xyz: Tensor,
        rotation: Tensor,
        scaling: Tensor,
        opacity: Tensor,
        features_dc: Tensor,
        features_rest: Tensor,
        sh_degree: Optional[int] = None,
        label: Optional[Tensor] = None,
    ) -> "Surfels":
        """Surfels from the raw parameters as stored in the point cloud files.

        Args:
            xyz: ``[N, 3]`` centres.
            rotation: ``[N, 4]`` unnormalised quaternions (w, x, y, z).
            scaling: ``[N, 2]`` log scales.
            opacity: ``[N, 1]`` or ``[N]`` opacity logits.
            features_dc: ``[N, 1, 3]`` SH coefficients of degree 0.
            features_rest: ``[N, K - 1, 3]`` higher degree SH coefficients.
            sh_degree: active SH degree, by default the highest one stored.
            label: optional ``[N, 1]`` or ``[N]`` hair label logits.
        """
        sh_coeffs = torch.cat([features_dc, features_rest], dim=1)
        max_degree = int(math.isqrt(sh_coeffs.shape[1])) - 1
        return cls(
            means=xyz,
            quats=F.normalize(rotation, dim=-1),
            scales=torch.exp(scaling),
            opacities=torch.sigmoid(opacity).reshape(-1),
            sh_coeffs=sh_coeffs,
            sh_degree=max_degree if sh_degree is None else min(sh_degree, max_degree),
            labels=None if label is None else torch.sigmoid(label).reshape(-1),
        )

    @property
    def max_sh_degree(self) -> int:
        """Highest SH degree the coefficients allow."""
        return int(math.isqrt(self.sh_coeffs.shape[1])) - 1

    def __len__(self) -> int:
        return self.means.shape[0]

    def to(self, device: torch.device | str) -> "Surfels":
        """Copy of the surfels on ``device``."""
        return Surfels(
            self.means.to(device),
            self.quats.to(device),
            self.scales.to(device),
            self.opacities.to(device),
            self.sh_coeffs.to(device),
            self.sh_degree,
            None if self.labels is None else self.labels.to(device),
        )

    def select(self, mask: Tensor) -> "Surfels":
        """The surfels where ``mask`` (bool ``[N]``) is set."""
        return Surfels(
            self.means[mask],
            self.quats[mask],
            self.scales[mask],
            self.opacities[mask],
            self.sh_coeffs[mask],
            self.sh_degree,
            None if self.labels is None else self.labels[mask],
        )

    def render(
        self,
        viewmat: Float[Tensor, "4 4"],
        K: Float[Tensor, "3 3"],
        width: int,
        height: int,
        background: Optional[Tensor] = None,
        scale_modifier: float = 1.0,
        sh_degree: Optional[int] = None,
        pse: Optional[Tensor] = None,
        sample_buffer: Optional[SampleBuffer] = None,
        colors: Optional[Tensor] = None,
    ) -> RasterOutput:
        """Render the surfels from a camera (OpenCV ``viewmat`` and ``K``).

        Differentiable like :func:`~strand_surfels.rendering.rasterize_surfels`;
        wrap in ``torch.no_grad()`` for display only. ``pse`` (per-surfel
        error, zeros requiring grad) and ``sample_buffer`` are passed on to the
        rasterizer. ``colors`` (``[N, 3]``) replaces the SH colors.
        """
        degree = self.sh_degree if sh_degree is None else min(sh_degree, self.max_sh_degree)
        return rasterize_surfels(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            colors,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
            background=background,
            scale_modifier=scale_modifier,
            sh_coeffs=self.sh_coeffs,
            sh_degree=degree,
            pse=pse,
            sample_buffer=sample_buffer,
        )
