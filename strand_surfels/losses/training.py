"""Training objective of the strand surfels.

Per pixel, with ``I`` the rendered and ``Î`` the ground truth image::

    photometric = (1 - λ_ssim) |I - Î| + λ_ssim (1 - SSIM(I, Î))
                + λ_alpha |alpha - foreground mask|
    geometry    = (λ_depth  geometry[depth_3d]
                 + λ_dir    geometry[direction_3d]
                 + λ_2d     geometry[orientation_2d]) * hair mask
    total       = mean(photometric + geometry
                       + photometric.detach() * error_map
                       + λ_dist distortion)

``geometry`` is :func:`~strand_surfels.losses.geometry.geometry_loss` of the
expected (or median) depth and the strand direction. The term
``photometric.detach() * error_map`` has no effect on the parameters; its
gradient with respect to the per-surfel errors gives the densification scores
(Taming 3DGS). Regularisation weights are zero before their start iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch
from torch import Tensor

from strand_surfels.losses.geometry import geometry_loss
from strand_surfels.rendering import RasterOutput

if TYPE_CHECKING:  # strand_surfels.data imports this package
    from strand_surfels.data.dataset import ViewData


def normalized_maps(out: RasterOutput) -> Dict[str, Tensor]:
    """Alpha, expected depth, median depth, strand direction (normalised by
    alpha) and distortion of a rendering, each ``[C, H, W]``."""
    aux = out.aux
    alpha = aux[1:2]
    return {
        "alpha": alpha,
        "depth": torch.nan_to_num(aux[0:1] / alpha, 0, 0, 0),
        "median_depth": torch.nan_to_num(aux[6:7], 0, 0),
        "direction": torch.nan_to_num(aux[2:5] / alpha, 0, 0, 0),
        "distortion": aux[5:6],
    }


def loss_weight(cfg: Dict, name: str, iteration: int) -> float:
    """Weight ``cfg[name]``, or zero up to ``cfg[name + '_start_iter']``."""
    start = cfg.get(f"{name}_start_iter")
    return cfg[name] if start is None or iteration > start else 0.0


def training_loss(
    out: RasterOutput, view: "ViewData", cfg: Dict, iteration: int
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Total loss and its (per-pixel mean) terms for logging.

    ``cfg`` is the ``optimization`` section of the configuration.
    """
    from fused_ssim import fused_ssim  # pylint: disable=import-outside-toplevel

    maps = normalized_maps(out)
    image, gt = out.render, view.image

    l1 = torch.abs(image - gt)
    ssim = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0))
    loss = (1.0 - cfg["lambda_dssim"]) * l1 + cfg["lambda_dssim"] * (1.0 - ssim.squeeze(0))
    alpha_loss = loss_weight(cfg, "lambda_alpha", iteration) * torch.abs(
        maps["alpha"] - view.alpha_mask
    )
    loss += alpha_loss

    depth = maps["median_depth"] if cfg["use_median_depths"] else maps["depth"]
    geom = geometry_loss(depth[0], maps["direction"], view.int_map, view.line_camera)
    geom_depth, geom_dir3d, geom_dir2d = geom * view.hair_mask

    # densification scores: gradient of the error-weighted photometric loss
    pse_loss = loss.clone().detach() * out.error_map

    loss += loss_weight(cfg, "lambda_3d_depth", iteration) * geom_depth
    loss += loss_weight(cfg, "lambda_3d_direction", iteration) * geom_dir3d
    loss += loss_weight(cfg, "lambda_2d_direction", iteration) * geom_dir2d
    dist_loss = loss_weight(cfg, "lambda_dist", iteration) * maps["distortion"]
    total = (loss + pse_loss + dist_loss).mean()

    terms = {
        "l1": l1.mean(),
        "ssim": ssim.mean(),
        "alpha": alpha_loss.mean(),
        "geom_depth": geom_depth.mean(),
        "geom_dir3d": geom_dir3d.mean(),
        "geom_dir2d": geom_dir2d.mean(),
        "dist": maps["distortion"].mean(),
        "total": total,
    }
    return total, terms


def psnr(image: Tensor, gt: Tensor) -> Tensor:
    """Peak signal-to-noise ratio of ``[C, H, W]`` images in ``[0, 1]``, averaged over channels."""
    mse = ((image - gt) ** 2).reshape(image.shape[0], -1).mean(1)
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).mean()
