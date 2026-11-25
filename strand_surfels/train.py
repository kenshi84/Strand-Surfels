"""Optimisation of the strand surfels of a scene.

Example::

    strand-surfels-train --data data/jewfro --smhc output/jewfro --run output/jewfro/run1

``--data`` is the input dataset, ``--smhc`` the folder with the SMHC results
(``int/`` and ``filtered/``) and ``--run`` the folder for the results:

* ``point_cloud/iteration_<i>/point_cloud.ply``: surfels at the save iterations,
* ``checkpoint/chkpnt<i>.pth``: model and optimizer state for resuming,
* ``config.yaml``: the configuration used, and TensorBoard logs.

The surfels also learn hair labels (a later addition, see
:mod:`strand_surfels.losses.labels`; ``--labels.enabled false`` turns it off).

The optimisation settings come from ``configs/splatting_cfgs.yaml``; every
entry can be overridden on the command line (``--optimization.iterations
30000``, see :mod:`strand_surfels.config`).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from strand_surfels.config import load_config, save_config
from strand_surfels.data import HairDataset
from strand_surfels.losses.labels import (
    LABEL_DEFAULTS,
    label_config,
    label_iou,
    label_step,
    render_labels,
)
from strand_surfels.losses.training import loss_weight, psnr, training_loss
from strand_surfels.model import SurfelModel, densification_budgets
from strand_surfels.rendering import SampleBuffer

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "splatting_cfgs.yaml"
)
#: training views shown in the evaluation reports
EVAL_VIEWS = (5, 10, 15, 20, 25)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:  # pylint: disable=too-many-instance-attributes
    """Optimisation loop: rendering, losses, densification, logging, saving.

    Args:
        dataset: the scene.
        cfg: the full configuration (``optimization`` and ``labels`` sections
            used here).
        run_dir: output folder.
        viewer: optional :class:`~strand_surfels.viewer.SurfelViewer`-like
            object with ``tick(iteration, stats)`` (shows the optimisation).
        log_tensorboard: write TensorBoard logs to ``run_dir``.
    """

    def __init__(
        self,
        dataset: HairDataset,
        cfg: Dict,
        run_dir: str,
        viewer=None,
        log_tensorboard: bool = True,
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.opt = cfg["optimization"]
        self.labels = label_config(cfg)
        self.run_dir = run_dir
        self.viewer = viewer
        self.model = SurfelModel.from_points(
            dataset.initial_points, dataset.cameras_extent, self.opt["sh_degree"],
            label_init=self.labels["init"],
        )  # fmt: skip
        self.model.setup_optimizer(self.opt, self.labels["lr"])
        self.initial_count = len(self.model)  # start of the budget schedule
        self._budgets: List[int] = []
        self._densify_step = 0
        self.first_iteration = 1
        self.background = torch.zeros(3, device="cuda")
        self.sample_buffer = SampleBuffer()
        self.history: List[Dict[str, float]] = []
        self.tb = None
        if log_tensorboard:
            try:
                from torch.utils.tensorboard import (  # pylint: disable=import-outside-toplevel
                    SummaryWriter,
                )

                self.tb = SummaryWriter(run_dir)
            except ImportError:
                logger.warning("TensorBoard is not available, not logging")

    # -------------------------------------------------------------- resuming

    def save_checkpoint(self, iteration: int) -> str:
        """Write model and optimizer state; returns the file path."""
        folder = os.path.join(self.run_dir, "checkpoint")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"chkpnt{iteration}.pth")
        state = {
            "model": self.model.state_dict(),
            "iteration": iteration,
            "initial_count": self.initial_count,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume from a checkpoint written by :meth:`save_checkpoint`."""
        state = torch.load(path, map_location="cuda", weights_only=False)
        self.model = SurfelModel.from_state_dict(state["model"], self.opt, self.labels["lr"])
        self.first_iteration = state["iteration"] + 1
        self.initial_count = state["initial_count"]

    # -------------------------------------------------------------- training

    def train(self) -> SurfelModel:  # pylint: disable=too-many-locals,too-many-statements
        """Run the optimisation; returns the final model."""
        opt, model = self.opt, self.model
        iterations = opt["iterations"]
        self._budgets = densification_budgets(
            self.initial_count,
            opt["final_splat_budget"],
            opt["densify_from_iter"],
            opt["densify_until_iter"],
            opt["densification_interval"],
        )
        # densification steps already done (when resuming)
        self._densify_step = sum(
            1
            for i in range(
                opt["densify_from_iter"] + 1, min(self.first_iteration, opt["densify_until_iter"])
            )
            if i % opt["densification_interval"] == 0
        )
        stack: List[int] = []
        ema = {"loss": 0.0, "dist": 0.0, "geometry": 0.0}
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.time()
        progress = tqdm(range(self.first_iteration, iterations + 1), desc="Training", ncols=110)

        for iteration in range(self.first_iteration, iterations + 1):
            start.record()
            model.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                model.oneup_sh_degree()

            # views in random order, each once per epoch
            if not stack:
                stack = list(range(len(self.dataset)))
            view = self.dataset[stack.pop(random.randint(0, len(stack) - 1))].load("cuda")

            # per-surfel errors: their gradient is the densification score
            pse = torch.zeros(len(model), device="cuda", requires_grad=True)
            out = model.surfels().render(
                view.viewmat, view.K, view.camera.width, view.camera.height,
                background=self.background, pse=pse, sample_buffer=self.sample_buffer,
            )  # fmt: skip
            loss, terms = training_loss(out, view, opt, iteration)
            loss.backward()

            # hair labels (a later addition): gradients on the label logits only
            label_loss = None
            if self.labels["enabled"] and iteration % self.labels["interval"] == 0:
                label_loss = label_step(
                    model.surfels(), view, self.labels["weight"], self.sample_buffer
                )
            end.record()

            with torch.no_grad():
                visible = out.visibility
                geometry = (
                    terms["geom_depth"] * loss_weight(opt, "lambda_3d_depth", iteration)
                    + terms["geom_dir3d"] * loss_weight(opt, "lambda_3d_direction", iteration)
                    + terms["geom_dir2d"] * loss_weight(opt, "lambda_2d_direction", iteration)
                )
                names = [k for k in terms if k != "total"]
                values = torch.stack([loss, geometry, pse.grad.max()] + [terms[k] for k in names])
                loss_v, geometry_v, pse_max, *term_values = values.tolist()
                record = {
                    "iteration": iteration,
                    "loss": loss_v,
                    "geometry": geometry_v,
                    "pse_max": pse_max,
                    "surfels": len(model),
                    "buckets": out.n_buckets,
                    **dict(zip(names, term_values)),
                }
                if label_loss is not None:
                    record["label"] = label_loss.item()
                dist_weight = loss_weight(opt, "lambda_dist", iteration)
                ema["loss"] = 0.4 * loss_v + 0.6 * ema["loss"]
                ema["dist"] = 0.4 * record["dist"] * dist_weight + 0.6 * ema["dist"]
                ema["geometry"] = 0.4 * geometry_v + 0.6 * ema["geometry"]
                if iteration % 10 == 0:
                    progress.set_postfix(
                        {"loss": f"{ema['loss']:.5f}", "geom": f"{ema['geometry']:.5f}",
                         "surfels": len(model)}  # fmt: skip
                    )
                    progress.update(10)

                self._densify(iteration, pse.grad, out.radii, visible)

                # optimizer step (skipped for parameters replaced by densification)
                if iteration < iterations:
                    model.step(visible)

                end.synchronize()
                record["iter_ms"] = start.elapsed_time(end)
                self.history.append(record)
                self._log(iteration, record)

            # drop this iteration's tensors before saving / evaluating
            del out, loss, terms, pse, view
            self._save_outputs(iteration)

            if self.viewer is not None:
                self.viewer.tick(iteration, {"loss": ema["loss"], "surfels": len(model)})

        progress.close()
        logger.info("Training finished in %.1f s", time.time() - wall_start)
        return model

    def _densify(self, iteration: int, pse_grad, radii, visible) -> None:
        """Densification statistics and, at the densification iterations, a
        densification step followed by the opacity decay."""
        opt, model = self.opt, self.model
        if iteration >= opt["densify_until_iter"]:
            if iteration % 100 == 0:
                torch.cuda.empty_cache()
            return
        model.add_densification_stats(pse_grad, radii, visible)
        if iteration <= opt["densify_from_iter"] or iteration % opt["densification_interval"]:
            return
        size_threshold = 20 if iteration > opt["opacity_reset_interval"] else None
        counts = model.densify_and_prune(
            opt["max_per_splat_error"],
            opt["min_opacity_to_prune"],
            self.dataset.cameras_extent,
            size_threshold,
            self._budgets[self._densify_step + 1],
        )
        self._densify_step += 1
        model.reduce_opacity()
        if self.tb:
            for k, v in counts.items():
                self.tb.add_scalar(f"densification/{k}", v, iteration)

    def _save_outputs(self, iteration: int) -> None:
        """Point cloud, checkpoint and evaluation at the configured iterations."""
        opt = self.opt
        if iteration in opt["save_iterations"]:
            folder = os.path.join(
                self.run_dir, "point_cloud", f"iteration_{iteration}"
            )
            path = os.path.join(folder, "point_cloud.ply")
            self.model.save_ply(path)
            tqdm.write(f"[{iteration}] saved {len(self.model):,} surfels to {path}")
            if self.labels["enabled"]:
                path = os.path.join(folder, "hair_point_cloud_mm.ply")
                self.model.export_hair_point_cloud_in_mm(path)
        if iteration in opt["checkpoint_iterations"]:
            tqdm.write(f"[{iteration}] checkpoint {self.save_checkpoint(iteration)}")
        if iteration in opt["test_iterations"]:
            # the backward samples (up to several GB) are re-allocated afterwards
            self.sample_buffer.release()
            self.evaluate(iteration)

    # ------------------------------------------------------------- reporting

    def _log(self, iteration: int, record: Dict[str, float]) -> None:
        if not self.tb:
            return
        tb = self.tb
        for k in ("l1", "alpha", "geom_depth", "geom_dir3d", "geom_dir2d", "dist"):
            tb.add_scalar(f"train/{k}", record[k], iteration)
        tb.add_scalar("train/total_loss", record["loss"], iteration)
        tb.add_scalar("train/geometry_loss", record["geometry"], iteration)
        tb.add_scalar("iter_time", record["iter_ms"], iteration)
        tb.add_scalar("total_points", record["surfels"], iteration)
        tb.add_scalar("buckets", record["buckets"], iteration)
        tb.add_scalar("pse_max", record["pse_max"], iteration)
        if "label" in record:
            tb.add_scalar("train/label", record["label"], iteration)

    @torch.no_grad()
    def evaluate(self, iteration: Optional[int] = None, views=EVAL_VIEWS) -> Dict[str, float]:
        """L1, PSNR and SSIM of renderings of some training views, and the
        intersection over union of the rendered hair labels and the hair masks."""
        from fused_ssim import fused_ssim  # pylint: disable=import-outside-toplevel

        torch.cuda.empty_cache()
        surfels = self.model.surfels()
        totals = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
        intersection = union = 0.0
        indices = [i % len(self.dataset) for i in views]
        for n, idx in enumerate(indices):
            view = self.dataset[idx].load("cuda")
            out = surfels.render(
                view.viewmat, view.K, view.camera.width, view.camera.height,
                background=self.background,
            )  # fmt: skip
            image = out.render.clamp(0.0, 1.0)
            gt = view.image.clamp(0.0, 1.0)
            totals["l1"] += torch.abs(image - gt).mean().double().item()
            totals["psnr"] += psnr(image, gt).double().item()
            totals["ssim"] += fused_ssim(image[None], gt[None], train=False).double().item()
            if self.labels["enabled"]:
                labels = render_labels(
                    surfels, view.viewmat, view.K, view.camera.width, view.camera.height
                )
                i, u = label_iou(labels, view.hair_mask)
                intersection, union = intersection + i, union + u
            if self.tb and n < 5 and iteration is not None:
                self.tb.add_images(f"train_view_{view.camera.name}/render", image[None], iteration)
        metrics = {k: v / len(indices) for k, v in totals.items()}
        if self.labels["enabled"]:
            metrics["hair_iou"] = intersection / max(union, 1.0)
        if iteration is not None:
            tqdm.write(
                f"[{iteration}] evaluation on {len(indices)} training views: "
                f"L1 {metrics['l1']:.5f} PSNR {metrics['psnr']:.3f} SSIM {metrics['ssim']:.4f}"
                + (f" hair IoU {metrics['hair_iou']:.4f}" if "hair_iou" in metrics else "")
            )
            if self.tb:
                for k, v in metrics.items():
                    self.tb.add_scalar(f"eval/{k}", v, iteration)
                self.tb.add_histogram("scene/opacity", self.model.opacity, iteration)
        torch.cuda.empty_cache()
        return metrics


def main(argv: Optional[List[str]] = None) -> None:
    """Command line entry point."""
    # The per-bucket samples of the backward pass vary in size from view to view
    # (up to several GB); expandable segments avoid fragmenting the CUDA memory
    # pool with them. Must be set before the first CUDA allocation.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    known, _ = pre.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], parents=[pre])
    parser.add_argument("--data", required=True, help="input dataset folder (data/<scene>)")
    parser.add_argument("--smhc", required=True, help="SMHC results folder (output/<scene>)")
    parser.add_argument("--run", required=True, help="folder for the results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--foreground-threshold", type=int, default=0,
        help="without foreground masks: pixels with a color value above this (0-255) are "
        "foreground; negative: all pixels are foreground",
    )  # fmt: skip
    parser.add_argument("--start-checkpoint", help="resume from this checkpoint")
    parser.add_argument(
        "--viewer", choices=["auto", "on", "off"], default="auto",
        help="show the optimisation in the Polyscope viewer (auto: if a display is available)",
    )  # fmt: skip
    cfg, args = load_config(known.config, argv, parser, defaults={"labels": LABEL_DEFAULTS})
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    seed_everything(args.seed)
    os.makedirs(args.run, exist_ok=True)
    save_config(os.path.join(args.run, "config.yaml"), cfg)
    dataset = HairDataset(
        args.data, args.smhc,
        downscale=cfg.get("images_resolution", 1),
        images_dir=cfg.get("images_dir", "image"),
        calibration_file=cfg.get("calibration_json_file", "calibration") + ".json",
        foreground_threshold=args.foreground_threshold if args.foreground_threshold >= 0 else None,
    )  # fmt: skip
    logger.info(
        "%d views, scene extent %.4f, %d initial surfels",
        len(dataset), dataset.cameras_extent, len(dataset.initial_points),
    )  # fmt: skip

    trainer = Trainer(dataset, cfg, args.run)
    if args.start_checkpoint:
        trainer.load_checkpoint(args.start_checkpoint)

    viewer = None
    if args.viewer != "off":
        # pylint: disable-next=import-outside-toplevel
        from strand_surfels.viewer import SurfelViewer, ViewerCamera, display_available

        if args.viewer == "on" or display_available():
            cameras = [ViewerCamera.from_camera(c) for c in dataset.cameras]

            def current_surfels():
                # looked up at every frame: the trainer's model changes
                return trainer.model.surfels()

            viewer = SurfelViewer(current_surfels, cameras)
            trainer.viewer = viewer
    trainer.train()
    if viewer is not None and viewer.enabled:
        logger.info("Close the viewer window to exit")
        viewer.run()


if __name__ == "__main__":
    main()


__all__ = ["Trainer", "main", "seed_everything"]
