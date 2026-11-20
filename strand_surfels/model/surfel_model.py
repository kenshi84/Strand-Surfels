"""Trainable set of strand surfels with budgeted densification.

Parameters (raw, optimised values; activations as in 3D Gaussian Splatting):

========= =========== ==========================================
name      shape       activation
========= =========== ==========================================
xyz       [N, 3]      centre
f_dc      [N, 1, 3]   SH coefficients of degree 0
f_rest    [N, K-1, 3] higher degree SH coefficients
opacity   [N, 1]      sigmoid
scaling   [N, 2]      exp; the surfel's two tangent axis lengths
rotation  [N, 4]      normalised quaternion (w, x, y, z); the first
                      tangent axis is the strand direction
label     [N, 1]      sigmoid; probability of belonging to the hair
                      (a later addition, see
                      :mod:`strand_surfels.losses.labels`)
========= =========== ==========================================

Densification follows *Taming 3DGS* (Mallick et al., 2024) and *Revising
Densification in Gaussian Splatting* (Rota Bulò et al., 2024):

* the score of a surfel is the maximum, over the densification interval, of
  its per-surfel error (the gradient of the error-weighted loss with respect
  to the surfel's error, see ``rasterize_surfels(pse=...)``);
* surfels whose score exceeds a threshold are candidates; small ones are
  cloned, large ones split, and the number of new surfels follows a budget
  schedule (:func:`~strand_surfels.model.schedules.densification_budgets`),
  candidates being sampled with probabilities proportional to their score;
* a cloned surfel and its copy get the opacity ``alpha' = 1 - sqrt(1 - alpha)``,
  so that together they are as opaque as the original;
* instead of periodic opacity resets, opacities are reduced by a small amount
  after every densification step.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from scipy.spatial.transform import Rotation
from torch import Tensor, nn

from strand_surfels.data.points import InitialPoints, mean_squared_neighbor_distance
from strand_surfels.io import load_surfels_ply, save_surfels_ply
from strand_surfels.model.schedules import exponential_lr
from strand_surfels.optim import SparseAdam
from strand_surfels.rendering import Surfels
from strand_surfels.rendering.sh import rgb_to_sh

PARAM_NAMES = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation", "label")
# file / Surfels.from_raw names of the parameters
_FILE_NAMES = {
    "xyz": "xyz",
    "f_dc": "features_dc",
    "f_rest": "features_rest",
    "opacity": "opacity",
    "scaling": "scaling",
    "rotation": "rotation",
    "label": "label",
}


def inverse_sigmoid(x: Tensor) -> Tensor:
    """Logit, the inverse of the opacity activation."""
    return torch.log(x / (1 - x))


def quaternion_to_rotmat(q: Tensor) -> Tensor:
    """Rotation matrices [N, 3, 3] of quaternions (w, x, y, z), normalised first."""
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    w, x, y, z = (q / norm[:, None]).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)  # fmt: skip


def frames_from_directions(dirs: Tensor) -> Tensor:
    """Rotation matrices [N, 3, 3] whose first column is the (normalised)
    direction; the other two columns complete a right-handed frame."""
    d = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    helper = torch.tensor([1.0, 0.0, 0.0], dtype=d.dtype, device=d.device).expand_as(d).clone()
    collinear = torch.all(torch.abs(d - helper) < 1e-5, dim=-1)
    helper[collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=d.dtype, device=d.device)
    v1 = torch.linalg.cross(helper, d)  # pylint: disable=not-callable
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = torch.linalg.cross(d, v1)  # pylint: disable=not-callable
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    return torch.stack((d, v1, v2), dim=-1)


class SurfelModel:  # pylint: disable=too-many-instance-attributes
    """Surfel parameters, their optimizer and the densification statistics."""

    def __init__(self, params: Dict[str, Tensor], max_sh_degree: int, spatial_lr_scale: float):
        if params.get("label") is None:  # point clouds and checkpoints without labels
            params = {**params, "label": params["opacity"].new_zeros((len(params["xyz"]), 1))}
        self.params: Dict[str, nn.Parameter] = {
            k: nn.Parameter(params[k].contiguous().requires_grad_(True)) for k in PARAM_NAMES
        }
        self.max_sh_degree = max_sh_degree
        self.active_sh_degree = 0
        self.spatial_lr_scale = spatial_lr_scale
        self.optimizer: Optional[SparseAdam] = None
        self.xyz_lr = None
        self.percent_dense = 0.0
        # densification statistics (see add_densification_stats)
        self.max_pse_grad = torch.empty(0)
        self.max_radii2d = torch.empty(0)
        self._reset_stats()

    # -------------------------------------------------------------- creation

    @classmethod
    def from_points(
        cls,
        points: InitialPoints,
        spatial_lr_scale: float,
        max_sh_degree: int = 3,
        device: str = "cuda",
        label_init: float = 0.0,
    ) -> "SurfelModel":
        """Surfels at the SMHC line samples: oriented along the strands, sized by
        the distance to the 3 nearest samples, gray, with opacity 0.1 and the
        hair label logit ``label_init``."""
        xyz = torch.from_numpy(points.xyz).float().to(device)
        n = xyz.shape[0]
        n_coeffs = (max_sh_degree + 1) ** 2
        features = torch.zeros((n, 3, n_coeffs), device=device)
        features[:, :, 0:1] = rgb_to_sh(torch.from_numpy(points.gray).float().to(device))[:, None]

        dist2 = torch.clamp_min(
            torch.from_numpy(mean_squared_neighbor_distance(points.xyz)).to(device), 1e-7
        )
        scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)

        frames = frames_from_directions(torch.from_numpy(points.dirs).float())
        q = Rotation.from_matrix(frames.numpy()).as_quat()  # (x, y, z, w)
        rotation = torch.from_numpy(q[:, [3, 0, 1, 2]].copy()).float().to(device)

        opacity = inverse_sigmoid(0.1 * torch.ones((n, 1), dtype=torch.float, device=device))
        return cls(
            {
                "xyz": xyz,
                "f_dc": features[:, :, 0:1].transpose(1, 2),
                "f_rest": features[:, :, 1:].transpose(1, 2),
                "opacity": opacity,
                "scaling": scaling,
                "rotation": rotation,
                "label": torch.full((n, 1), label_init, dtype=torch.float, device=device),
            },
            max_sh_degree,
            spatial_lr_scale,
        )

    @classmethod
    def from_ply(
        cls, path: str, spatial_lr_scale: float = 1.0, device: str = "cuda"
    ) -> "SurfelModel":
        """Surfels of a point cloud file, with all SH degrees active."""
        raw = load_surfels_ply(path)
        params = {
            k: torch.from_numpy(raw[_FILE_NAMES[k]]).float().to(device)
            for k in PARAM_NAMES
            if _FILE_NAMES[k] in raw
        }
        degree = int(math.isqrt(params["f_rest"].shape[1] + 1)) - 1
        model = cls(params, degree, spatial_lr_scale)
        model.active_sh_degree = degree
        return model

    def save_ply(self, path: str) -> None:
        """Write the raw parameters to a point cloud file."""
        save_surfels_ply(
            path, {_FILE_NAMES[k]: v.detach().cpu().numpy() for k, v in self.params.items()}
        )

    # ------------------------------------------------------------ accessors

    def __len__(self) -> int:
        return self.params["xyz"].shape[0]

    @property
    def opacity(self) -> Tensor:
        """Activated opacities [N, 1]."""
        return torch.sigmoid(self.params["opacity"])

    @property
    def scaling(self) -> Tensor:
        """Activated scales [N, 2]."""
        return torch.exp(self.params["scaling"])

    def surfels(self) -> Surfels:
        """Activated (differentiable) parameters for rendering."""
        p = self.params
        return Surfels.from_raw(
            p["xyz"], p["rotation"], p["scaling"], p["opacity"], p["f_dc"], p["f_rest"],
            sh_degree=self.active_sh_degree, label=p["label"],
        )  # fmt: skip

    def oneup_sh_degree(self) -> None:
        """Activate the next SH degree (up to the maximum)."""
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    # ---------------------------------------------------------- optimisation

    def setup_optimizer(self, cfg: Dict, label_lr: float = 0.05) -> None:
        """Sparse Adam with one group per parameter and the position schedule.

        ``cfg`` keys: ``position_lr_init``, ``position_lr_final``,
        ``position_lr_delay_mult``, ``position_lr_max_steps``, ``feature_lr``,
        ``opacity_lr``, ``scaling_lr``, ``rotation_lr``, ``percent_dense``.
        Position learning rates are scaled by the scene extent. ``label_lr``
        is the learning rate of the hair label logits (they only receive
        gradients from the label loss).
        """
        self.percent_dense = cfg["percent_dense"]
        lrs = {
            "xyz": cfg["position_lr_init"] * self.spatial_lr_scale,
            "f_dc": cfg["feature_lr"],
            "f_rest": cfg["feature_lr"] / 20.0,
            "opacity": cfg["opacity_lr"],
            "scaling": cfg["scaling_lr"],
            "rotation": cfg["rotation_lr"],
            "label": label_lr,
        }
        groups = [{"params": [self.params[k]], "lr": lrs[k], "name": k} for k in PARAM_NAMES]
        self.optimizer = SparseAdam(groups, lr=0.0, eps=1e-15)
        self.xyz_lr = exponential_lr(
            cfg["position_lr_init"] * self.spatial_lr_scale,
            cfg["position_lr_final"] * self.spatial_lr_scale,
            max_steps=cfg["position_lr_max_steps"],
            delay_mult=cfg["position_lr_delay_mult"],
        )
        self._reset_stats()

    def update_learning_rate(self, iteration: int) -> float:
        """Set the position learning rate of ``iteration``; returns it."""
        lr = self.xyz_lr(iteration)
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = lr
        return lr

    def step(self, visibility: Tensor) -> None:
        """Optimizer step on the visible surfels, then clear the gradients."""
        self.optimizer.step(visibility)
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------- parameter bookkeeping

    def _sync_params(self) -> None:
        for group in self.optimizer.param_groups:
            self.params[group["name"]] = group["params"][0]

    def _replace_param(self, name: str, tensor: Tensor) -> None:
        """New values for one parameter; its Adam moments are reset."""
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old = group["params"][0]
            state = self.optimizer.state.pop(old, None)
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            if state is not None:
                state["exp_avg"] = torch.zeros_like(tensor)
                state["exp_avg_sq"] = torch.zeros_like(tensor)
                self.optimizer.state[group["params"][0]] = state
        self._sync_params()

    def _select(self, keep: Tensor) -> None:
        """Keep the surfels (and their optimizer state) where ``keep`` is set."""
        for group in self.optimizer.param_groups:
            old = group["params"][0]
            state = self.optimizer.state.pop(old, None)
            group["params"][0] = nn.Parameter(old[keep].requires_grad_(True))
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                self.optimizer.state[group["params"][0]] = state
        self._sync_params()
        self.max_pse_grad = self.max_pse_grad[keep]
        self.max_radii2d = self.max_radii2d[keep]

    def _append(self, new: Dict[str, Tensor]) -> None:
        """Add surfels (with zero optimizer moments); resets the statistics."""
        for group in self.optimizer.param_groups:
            old, ext = group["params"][0], new[group["name"]]
            state = self.optimizer.state.pop(old, None)
            group["params"][0] = nn.Parameter(torch.cat((old, ext), dim=0).requires_grad_(True))
            if state is not None:
                state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(ext)), dim=0)
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(ext)), dim=0)
                self.optimizer.state[group["params"][0]] = state
        self._sync_params()
        self._reset_stats()

    # --------------------------------------------------------- densification

    def _reset_stats(self) -> None:
        device = self.params["xyz"].device
        self.max_pse_grad = torch.zeros((len(self), 1), device=device)
        self.max_radii2d = torch.zeros((len(self),), device=device)

    @torch.no_grad()
    def add_densification_stats(self, pse_grad: Tensor, radii: Tensor, visible: Tensor) -> None:
        """Track the maximum per-surfel error gradient and screen radius of the
        visible surfels over the densification interval."""
        self.max_radii2d[visible] = torch.max(self.max_radii2d[visible], radii[visible].float())
        grad = pse_grad.reshape(-1, 1)
        self.max_pse_grad[visible] = torch.maximum(self.max_pse_grad[visible], grad[visible])

    @torch.no_grad()
    def densify_and_prune(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        score_threshold: float,
        min_opacity: float,
        extent: float,
        max_screen_size: Optional[float],
        budget: int,
    ) -> Dict[str, int]:
        """One densification step; returns the numbers of cloned, split and pruned surfels.

        Args:
            score_threshold: minimum score of densification candidates.
            min_opacity: surfels below are pruned.
            extent: scene extent; surfels larger than ``percent_dense * extent``
                are split, smaller ones cloned, and (with ``max_screen_size``)
                those larger than ``0.1 * extent`` are pruned.
            max_screen_size: prune surfels whose screen radius exceeded this
                (pixels) during the interval, if given.
            budget: total number of surfels allowed after this step.
        """
        scores = self.max_pse_grad
        scores[scores.isnan()] = 0.0
        candidates = torch.norm(scores, dim=-1) >= score_threshold
        size = self.scaling.max(dim=1).values
        clones = candidates & (size <= self.percent_dense * extent)
        splits = candidates & (size > self.percent_dense * extent)
        n_clones, n_splits = int(clones.sum()), int(splits.sum())

        cloned = split = 0
        if n_clones + n_splits > 0:
            n = len(self)
            budget = min(budget, n_clones + n_splits + n)
            clone_budget = ((budget - n) * n_clones) // (n_clones + n_splits)
            split_budget = ((budget - n) * n_splits) // (n_clones + n_splits)
            if budget > 0:
                cloned = self._clone(scores.clone(), clone_budget, clones)
                split = self._split(scores.clone(), split_budget, splits)

        prune = (self.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_on_screen = self.max_radii2d > max_screen_size
            big_in_world = self.scaling.max(dim=1).values > 0.1 * extent
            prune = prune | big_on_screen | big_in_world
        pruned = int(prune.sum())
        self._select(~prune)
        torch.cuda.empty_cache()
        return {"cloned": cloned, "split": split, "pruned": pruned}

    def _clone(self, scores: Tensor, budget: int, candidates: Tensor) -> int:
        """Duplicate ``budget`` candidates sampled by score."""
        scores[~candidates] = 0
        selected = torch.zeros((len(self),), dtype=torch.bool, device=scores.device)
        if budget > 0:
            selected[torch.multinomial(scores.squeeze(), budget, replacement=False)] = True
        p = self.params
        new = {k: p[k][selected] for k in ("xyz", "f_dc", "f_rest", "scaling", "rotation", "label")}
        # Opacity compensation (Revising Densification, Eq. 9): the cloned
        # surfel and its copy, each with alpha' = 1 - sqrt(1 - alpha), together
        # let through as much light as the original surfel did.
        compensated = inverse_sigmoid(1 - torch.sqrt(1 - self.opacity[selected]))
        p["opacity"].data[selected] = compensated
        new["opacity"] = compensated
        self._append(new)
        return int(selected.sum())

    def _split(self, scores: Tensor, budget: int, candidates: Tensor, n_children: int = 2) -> int:
        """Replace ``budget`` candidates sampled by score with ``n_children``
        smaller surfels each, placed randomly in the surfel's tangent plane."""
        scores[~candidates] = 0
        n = len(self)  # includes surfels cloned in this step (score 0)
        padded = torch.zeros((n,), dtype=torch.float32)  # sampled on the CPU
        padded[: scores.shape[0]] = scores.squeeze()
        selected = torch.zeros_like(padded, dtype=torch.bool, device=self.params["xyz"].device)
        if budget > 0:
            selected[torch.multinomial(padded, budget, replacement=False)] = True

        p = self.params
        stds = self.scaling[selected].repeat(n_children, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = quaternion_to_rotmat(p["rotation"][selected]).repeat(n_children, 1, 1)
        new = {
            "xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + p["xyz"][selected].repeat(n_children, 1),
            "scaling": torch.log(self.scaling[selected].repeat(n_children, 1) / (0.8 * n_children)),
            "rotation": p["rotation"][selected].repeat(n_children, 1),
            "f_dc": p["f_dc"][selected].repeat(n_children, 1, 1),
            "f_rest": p["f_rest"][selected].repeat(n_children, 1, 1),
            "opacity": p["opacity"][selected].repeat(n_children, 1),
            "label": p["label"][selected].repeat(n_children, 1),
        }
        self._append(new)
        n_new = n_children * int(selected.sum())
        keep = ~torch.cat((selected, torch.zeros(n_new, device=selected.device, dtype=torch.bool)))
        self._select(keep)
        return int(selected.sum())

    @torch.no_grad()
    def reduce_opacity(self, amount: float = 0.001) -> None:
        """Lower all opacities by ``amount`` (not below ``amount``)."""
        reduced = torch.clamp(self.opacity - amount, min=amount)
        self._replace_param("opacity", inverse_sigmoid(reduced))

    # ----------------------------------------------------------- checkpoints

    def state_dict(self) -> Dict:
        """Parameters, optimizer state and statistics."""
        return {
            "params": {k: v.detach() for k, v in self.params.items()},
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "max_pse_grad": self.max_pse_grad,
            "max_radii2d": self.max_radii2d,
        }

    @classmethod
    def from_state_dict(
        cls, state: Dict, cfg: Optional[Dict] = None, label_lr: float = 0.05
    ) -> "SurfelModel":
        """Model of a :meth:`state_dict`; with ``cfg`` the optimizer is restored too."""
        model = cls(state["params"], state["max_sh_degree"], state["spatial_lr_scale"])
        model.active_sh_degree = state["active_sh_degree"]
        if cfg is not None:
            model.setup_optimizer(cfg, label_lr)
            if state["optimizer"] is not None:
                model.optimizer.load_state_dict(state["optimizer"])
        model.max_pse_grad = state["max_pse_grad"]
        model.max_radii2d = state["max_radii2d"]
        return model
