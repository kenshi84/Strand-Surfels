"""Surfel model: initialisation, densification, bookkeeping, schedules."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from strand_surfels.data import InitialPoints
from strand_surfels.io import load_surfels_ply, save_surfels_ply
from strand_surfels.model import (
    PARAM_NAMES,
    SurfelModel,
    densification_budgets,
    exponential_lr,
    frames_from_directions,
    quaternion_to_rotmat,
)

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

CFG = {
    "position_lr_init": 1.6e-5,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 1000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "percent_dense": 0.01,
}


def initial_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3)).astype(np.float32)
    dirs[0] = [1.0, 0.0, 0.0]  # collinear with the helper axis
    return InitialPoints(
        xyz=rng.normal(size=(n, 3)).astype(np.float32),
        dirs=dirs,
        gray=rng.uniform(size=(n, 1)).astype(np.float32),
    )


def trained_model(seed=0, n=500):
    model = SurfelModel.from_points(initial_points(n, seed), spatial_lr_scale=1.0)
    model.setup_optimizer(CFG)
    g = torch.Generator(device="cuda").manual_seed(seed)
    for p in model.params.values():  # populate the optimizer state
        p.grad = torch.randn(p.shape, device="cuda", generator=g) * 1e-2
    model.step(torch.ones(len(model), dtype=torch.bool, device="cuda"))
    return model


def check_consistent(model):
    n = len(model)
    for k in PARAM_NAMES:
        p = model.params[k]
        assert p.shape[0] == n and p.requires_grad
        group = next(g for g in model.optimizer.param_groups if g["name"] == k)
        assert group["params"][0] is p
        state = model.optimizer.state[p]
        assert state["exp_avg"].shape == p.shape and state["exp_avg_sq"].shape == p.shape
    assert model.max_pse_grad.shape == (n, 1) and model.max_radii2d.shape == (n,)


def test_initialisation_follows_the_strands():
    points = initial_points()
    model = SurfelModel.from_points(points, spatial_lr_scale=1.0)
    R = quaternion_to_rotmat(model.params["rotation"])
    d = F.normalize(torch.from_numpy(points.dirs).cuda(), dim=-1)
    torch.testing.assert_close(R[:, :, 0], d, atol=1e-5, rtol=0)
    eye = torch.eye(3, device="cuda").expand(len(model), 3, 3)
    torch.testing.assert_close(R.transpose(1, 2) @ R, eye, atol=1e-5, rtol=0)
    torch.testing.assert_close(model.opacity, torch.full((len(model), 1), 0.1, device="cuda"))
    assert model.params["f_rest"].abs().max() == 0
    colors = model.surfels().sh_coeffs[:, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(colors, torch.from_numpy(points.gray).cuda().expand(-1, 3))
    frames = frames_from_directions(torch.from_numpy(points.dirs))
    assert torch.allclose(
        torch.linalg.det(frames), torch.ones(len(points)), atol=1e-5  # pylint: disable=not-callable
    )


def test_densification_respects_the_budget():
    model = trained_model()
    n = len(model)
    model.max_pse_grad[: n // 2] = 1.0  # half of the surfels are candidates
    budget = n + 60
    counts = model.densify_and_prune(0.5, 0.0, extent=1.0, max_screen_size=None, budget=budget)
    assert counts["cloned"] + counts["split"] <= 60
    assert len(model) == n + counts["cloned"] + counts["split"] - counts["pruned"]
    assert counts["pruned"] == 0
    check_consistent(model)
    # the statistics restart after densification
    assert model.max_pse_grad.abs().max() == 0


def test_clone_compensates_opacity_and_split_stays_in_tangent_plane():
    model = trained_model()
    n = len(model)
    with torch.no_grad():
        model.params["scaling"][0] = np.log(1e-3)  # small: cloned
        model.params["scaling"][1] = np.log(0.1)  # large: split
    xyz0 = model.params["xyz"].detach().clone()
    opacity0 = model.opacity.detach().clone()
    scaling0 = model.scaling.detach().clone()
    label0 = model.params["label"].detach().clone()
    R0 = quaternion_to_rotmat(model.params["rotation"].detach())
    model.max_pse_grad[:2] = 1.0
    counts = model.densify_and_prune(0.5, 0.0, extent=1.0, max_screen_size=None, budget=n + 10)
    assert counts == {"cloned": 1, "split": 1, "pruned": 0}
    check_consistent(model)
    # layout: originals without the split one, the clone, the two children
    assert len(model) == n + 2
    xyz = model.params["xyz"].detach()
    assert torch.equal(xyz[0], xyz0[0]) and torch.equal(xyz[n - 1], xyz0[0])
    torch.testing.assert_close(xyz[1 : n - 1], xyz0[2:n], rtol=0, atol=0)

    # opacity compensation alpha' = 1 - sqrt(1 - alpha) of the cloned surfel
    # and its copy only; the other surfels keep their opacity
    compensated = 1 - torch.sqrt(1 - opacity0[0])
    torch.testing.assert_close(model.opacity[0], compensated, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(model.opacity[n - 1], compensated, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(model.opacity[1 : n - 1], opacity0[2:n], rtol=1e-6, atol=0)
    # transmittance through the pair equals that through the original surfel
    torch.testing.assert_close((1 - model.opacity[0]) ** 2, 1 - opacity0[0], rtol=1e-5, atol=1e-7)

    # the split surfel is replaced by two children on its tangent plane
    children = xyz[-2:]
    offsets = (children - xyz0[1]) @ R0[1]
    assert offsets[:, 2].abs().max() < 1e-5
    torch.testing.assert_close(
        model.scaling[-2:], (scaling0[1] / 1.6).expand(2, 2), rtol=1e-5, atol=0
    )
    torch.testing.assert_close(model.opacity[-2:], opacity0[1].expand(2, 1))

    # hair labels are inherited by the clone and the children
    label = model.params["label"].detach()
    assert torch.equal(label[n - 1], label0[0]) and torch.equal(label[0], label0[0])
    assert torch.equal(label[-2:], label0[1].expand(2, 1))


def test_pruning_and_opacity_reduction():
    model = trained_model()
    with torch.no_grad():
        model.params["opacity"][:10] = -10.0  # nearly transparent
    model.max_radii2d[10] = 100.0
    n = len(model)
    counts = model.densify_and_prune(10.0, 0.05, extent=1.0, max_screen_size=20, budget=n)
    assert counts["pruned"] >= 11 and len(model) == n - counts["pruned"]
    check_consistent(model)
    before = model.opacity.detach().clone()
    model.reduce_opacity(0.001)
    torch.testing.assert_close(
        model.opacity, torch.clamp(before - 0.001, min=0.001), atol=1e-6, rtol=0
    )


def test_ply_and_state_dict_round_trip(tmp_path):
    model = trained_model()
    model.active_sh_degree = 3
    path = str(tmp_path / "surfels.ply")
    model.save_ply(path)
    loaded = SurfelModel.from_ply(path)
    for k in PARAM_NAMES:
        assert torch.equal(loaded.params[k], model.params[k].detach())
    assert loaded.active_sh_degree == 3

    restored = SurfelModel.from_state_dict(model.state_dict(), CFG)
    for k in PARAM_NAMES:
        assert torch.equal(restored.params[k], model.params[k].detach())
        a = restored.optimizer.state[restored.params[k]]["exp_avg"]
        assert torch.equal(a, model.optimizer.state[model.params[k]]["exp_avg"])


def test_schedules():
    budgets = densification_budgets(1000, 5000, 500, 8000, 100)
    assert len(budgets) == 75 and budgets[0] == 1000
    assert all(b2 >= b1 for b1, b2 in zip(budgets, budgets[1:])) and budgets[-1] < 5000
    lr = exponential_lr(1e-3, 1e-5, max_steps=100)
    assert lr(0) == pytest.approx(1e-3) and lr(100) == pytest.approx(1e-5)
    assert lr(50) == pytest.approx(1e-4) and lr(1000) == pytest.approx(1e-5)


def test_point_clouds_without_labels_load(tmp_path):
    model = trained_model()
    path = str(tmp_path / "surfels.ply")
    model.save_ply(path)
    raw = load_surfels_ply(path)
    del raw["label"]
    save_surfels_ply(path, raw)
    loaded = SurfelModel.from_ply(path)
    assert torch.equal(loaded.params["label"], torch.zeros(len(model), 1, device="cuda"))
    assert loaded.surfels().labels is not None
