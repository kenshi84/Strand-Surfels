"""Hair labels of the surfels (a later addition).

*This was added after the STITCH publication; it is not part of the method
described in the paper and was not used for its results.* It makes it easy to
separate the surfels that represent the hair from the others (body, head,
clothing), e.g. to export or view the hair alone.

Every surfel has a label logit ``l`` (``SurfelModel.params["label"]``); its
hair probability is ``p = sigmoid(l)``. The probabilities are learned from
the hair masks of the training views: every ``interval`` iterations, the
current view is rendered a second time with the colors ``(p, 0, 0)`` and the
geometry (positions, rotations, scales, opacities) detached, so that the
first channel is the alpha-blended probability ``Σ w_i p_i``, and::

    label loss = weight * mean |Σ w_i p_i - alpha * hair mask|

with ``alpha = Σ w_i`` (detached). Only the label logits receive gradients:
the labels do not influence the other parameters or the densification, they
only follow them (cloned and split surfels inherit their label).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from strand_surfels.rendering import RasterOutput, SampleBuffer, Surfels

if TYPE_CHECKING:  # strand_surfels.data imports this package
    from strand_surfels.data.dataset import ViewData

#: Default ``labels`` section of the configuration (not part of
#: ``configs/splatting_cfgs.yaml``, which predates the labels).
LABEL_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "weight": 1.0,
    "lr": 0.05,  # Adam learning rate of the logits
    "interval": 10,  # label update every this many iterations
    "init": 0.0,  # initial logit (probability 0.5)
}


def label_config(cfg: Dict) -> Dict[str, Any]:
    """The ``labels`` section of a configuration, completed with the defaults."""
    out = copy.deepcopy(LABEL_DEFAULTS)
    out.update(cfg.get("labels") or {})
    return out


def render_labels(
    surfels: Surfels,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    sample_buffer: Optional[SampleBuffer] = None,
) -> RasterOutput:
    """Rendering whose first color channel is the blended hair probability;
    differentiable with respect to ``surfels.labels`` only."""
    assert surfels.labels is not None, "the surfels have no labels"
    p = surfels.labels.reshape(-1, 1)
    colors = torch.cat([p, torch.zeros_like(p).expand(-1, 2)], dim=1)
    geometry = Surfels(
        surfels.means.detach(),
        surfels.quats.detach(),
        surfels.scales.detach(),
        surfels.opacities.detach(),
        surfels.sh_coeffs.detach(),
        surfels.sh_degree,
    )
    return geometry.render(viewmat, K, width, height, sample_buffer=sample_buffer, colors=colors)


def label_loss(out: RasterOutput, hair_mask: Tensor) -> Tensor:
    """Mean ``|Σ w_i p_i - alpha * hair mask|`` of a :func:`render_labels` output."""
    alpha = out.alpha.detach()
    return torch.abs(out.render[0:1] - alpha * hair_mask).mean()


@torch.no_grad()
def label_iou(out: RasterOutput, hair_mask: Tensor, min_alpha: float = 0.5) -> Tuple[float, float]:
    """Intersection and union (pixel counts) of the rendered hair (blended
    probability above 0.5 where alpha exceeds ``min_alpha``) and the hair mask."""
    alpha = out.alpha
    predicted = (alpha > min_alpha) & (out.render[0:1] > 0.5 * alpha)
    target = hair_mask > 0.5
    return float((predicted & target).sum()), float((predicted | target).sum())


def label_step(
    surfels: Surfels, view: "ViewData", weight: float, sample_buffer: Optional[SampleBuffer] = None
) -> Tensor:
    """Label rendering, loss and backward pass for one view; returns the
    (weighted, detached) loss. Gradients accumulate on the label logits."""
    out = render_labels(
        surfels, view.viewmat, view.K, view.camera.width, view.camera.height, sample_buffer
    )
    loss = weight * label_loss(out, view.hair_mask)
    loss.backward()
    return loss.detach()


#: Display colors of hair and other surfels (viewer).
HAIR_COLOR = (0.95, 0.45, 0.1)
OTHER_COLOR = (0.35, 0.55, 0.9)


def label_colors(labels: Tensor) -> Tensor:
    """Colors ``[N, 3]`` blending between :data:`OTHER_COLOR` (probability 0)
    and :data:`HAIR_COLOR` (probability 1)."""
    hair = torch.tensor(HAIR_COLOR, device=labels.device)
    other = torch.tensor(OTHER_COLOR, device=labels.device)
    p = labels.reshape(-1, 1)
    return p * hair + (1 - p) * other
