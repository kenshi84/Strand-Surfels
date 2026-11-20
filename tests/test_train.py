"""End-to-end training on a tiny synthetic scene: loop, densification, saving, resuming."""

# pylint: disable=missing-function-docstring

import copy
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
pytest.importorskip("fused_ssim")

# pylint: disable=wrong-import-position
from test_data import make_scene  # noqa: E402

from strand_surfels.config import load_yaml  # noqa: E402
from strand_surfels.data import HairDataset  # noqa: E402
from strand_surfels.losses import label_step  # noqa: E402
from strand_surfels.model import SurfelModel  # noqa: E402
from strand_surfels.train import DEFAULT_CONFIG, Trainer, main, seed_everything  # noqa: E402


def small_config():
    cfg = copy.deepcopy(load_yaml(DEFAULT_CONFIG))
    cfg["optimization"].update(
        {
            "iterations": 40,
            "position_lr_max_steps": 40,
            "densify_from_iter": 5,
            "densify_until_iter": 30,
            "densification_interval": 5,
            "opacity_reset_interval": 20,
            "final_splat_budget": 400,
            "max_per_splat_error": 0.0,
            "lambda_alpha_start_iter": 2,
            "lambda_dist_start_iter": 2,
            "lambda_3d_depth_start_iter": 2,
            "lambda_3d_direction_start_iter": 2,
            "test_iterations": [40],
            "save_iterations": [40],
            "checkpoint_iterations": [20],
        }
    )
    return cfg


def scene_dataset(tmp_path):
    scene = make_scene(str(tmp_path))
    # place the initial samples in front of the cameras
    positions = np.load(os.path.join(scene["output"], "filtered", "filtered_positions.npy"))
    positions[:, :3] = positions[:, :3] * 0.2 + np.array([0.1, 0.2, 0.3])
    np.save(os.path.join(scene["output"], "filtered", "filtered_positions.npy"), positions)
    return scene, HairDataset(scene["data"], scene["output"], initial_points_stride=5)


def test_training_runs_densifies_and_saves(tmp_path):
    _, dataset = scene_dataset(tmp_path)
    seed_everything(0)
    run = str(tmp_path / "run")
    trainer = Trainer(dataset, small_config(), run, log_tensorboard=False)
    n0 = len(trainer.model)
    model = trainer.train()
    assert len(trainer.history) == 40
    assert all(np.isfinite(h["loss"]) for h in trainer.history)
    assert len(model) != n0  # densification and pruning changed the surfels
    assert max(h["surfels"] for h in trainer.history) <= 400 + n0
    ply = os.path.join(run, "point_cloud", "iteration_40", "point_cloud.ply")
    assert len(SurfelModel.from_ply(ply)) == len(model)
    assert os.path.exists(os.path.join(run, "checkpoint", "chkpnt20.pth"))
    assert set(trainer.evaluate()) == {"l1", "psnr", "ssim", "hair_iou"}
    assert sum("label" in h for h in trainer.history) == 4  # every 10 iterations


def test_resume_from_checkpoint(tmp_path):
    _, dataset = scene_dataset(tmp_path)
    cfg = small_config()
    run = str(tmp_path / "run")
    seed_everything(0)
    Trainer(dataset, cfg, run, log_tensorboard=False).train()
    resumed = Trainer(dataset, cfg, str(tmp_path / "resumed"), log_tensorboard=False)
    resumed.load_checkpoint(os.path.join(run, "checkpoint", "chkpnt20.pth"))
    assert resumed.first_iteration == 21
    resumed.train()
    assert [h["iteration"] for h in resumed.history] == list(range(21, 41))


def test_command_line(tmp_path):
    scene, _ = scene_dataset(tmp_path)
    run = str(tmp_path / "cli")
    main(
        [
            "--data", scene["data"], "--smhc", scene["output"], "--run", run, "--viewer", "off",
            "--optimization.iterations", "12", "--optimization.save_iterations", "12",
            "--optimization.test_iterations", "--optimization.checkpoint_iterations",
            "--optimization.densify_until_iter", "0", "--labels.enabled", "false",
        ]
    )  # fmt: skip
    assert os.path.exists(os.path.join(run, "point_cloud", "iteration_12", "point_cloud.ply"))
    saved = load_yaml(os.path.join(run, "config.yaml"))
    assert saved["optimization"]["iterations"] == 12
    assert saved["labels"]["enabled"] is False  # defaults can be overridden too


def test_labels_learn_the_hair_masks_and_nothing_else(tmp_path):
    """The label loss reaches the label logits only, and lowers itself."""
    _, dataset = scene_dataset(tmp_path)
    seed_everything(0)
    cfg = small_config()
    trainer = Trainer(dataset, cfg, str(tmp_path / "run"), log_tensorboard=False)
    model, view = trainer.model, dataset[0].load("cuda")
    weight = 1.0
    first = label_step(model.surfels(), view, weight)
    grads = {k: p.grad for k, p in model.params.items()}
    assert grads["label"] is not None and grads["label"].abs().max() > 0
    assert all(g is None for k, g in grads.items() if k != "label")

    model.optimizer.zero_grad(set_to_none=True)
    for _ in range(50):
        label_step(model.surfels(), view, weight)
        model.step(torch.ones(len(model), dtype=torch.bool, device="cuda"))
    assert label_step(model.surfels(), view, weight) < first


def test_training_without_labels(tmp_path):
    _, dataset = scene_dataset(tmp_path)
    cfg = small_config()
    cfg["labels"] = {"enabled": False}
    trainer = Trainer(dataset, cfg, str(tmp_path / "run"), log_tensorboard=False)
    trainer.train()
    assert not any("label" in h for h in trainer.history)
    assert set(trainer.evaluate()) == {"l1", "psnr", "ssim"}
