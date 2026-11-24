"""Viewer building blocks: PLY files, camera conversions, display images, GPU uploads."""

# pylint: disable=missing-function-docstring

import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fury.colormap import boys2rgb
from scipy.spatial.transform import Rotation
from test_data import make_scene

from strand_surfels.io import load_surfels_ply, read_ply, save_surfels_ply
from strand_surfels.viewer.cameras import (
    CV_TO_GL,
    ViewerCamera,
    opencv_from_polyscope,
    up_direction,
)


def random_raw_params(n: int = 50, degree: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2
    return {
        "xyz": rng.normal(size=(n, 3)).astype(np.float32),
        "features_dc": rng.normal(size=(n, 1, 3)).astype(np.float32),
        "features_rest": rng.normal(size=(n, k - 1, 3)).astype(np.float32),
        "opacity": rng.normal(size=(n, 1)).astype(np.float32),
        "scaling": rng.normal(size=(n, 2)).astype(np.float32),
        "rotation": rng.normal(size=(n, 4)).astype(np.float32),
    }


@pytest.mark.parametrize("strands_to_world", [None, np.eye(3)], ids=["world", "identity"])
def test_ply_round_trip(tmp_path, strands_to_world):
    params = random_raw_params()
    # Non-unit wxyz quaternions taking local +x to +x, -x, +y, and -z.
    params["rotation"][:4] = [[2, 0, 0, 0], [0, 0, 0, 3], [2, 0, 0, 2], [1, 0, 1, 0]]
    path = str(tmp_path / "surfels.ply")
    save_surfels_ply(path, params, strands_to_world_rotation=strands_to_world)
    loaded = load_surfels_ply(path)
    for key, value in params.items():
        np.testing.assert_array_equal(loaded[key], value)
    # SH coefficients are stored channel-major, as in 3D Gaussian Splatting
    raw = read_ply(path)
    np.testing.assert_array_equal(raw["f_rest_0"], params["features_rest"][:, 0, 0])
    np.testing.assert_array_equal(raw["f_rest_15"], params["features_rest"][:, 0, 1])
    tangents = np.column_stack([raw[name] for name in ("nx", "ny", "nz")])
    np.testing.assert_allclose(
        tangents[:4], [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, -1]], atol=1e-7
    )
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=1e-7)
    colors = np.column_stack([raw[name] for name in ("red", "green", "blue")])
    assert colors.dtype == np.uint8
    np.testing.assert_allclose(colors, np.clip(boys2rgb(tangents), 0, 1) * 255, rtol=0, atol=0.5)


def test_ply_in_strand_coordinates(tmp_path):
    from strand_surfels.rendering import eval_sh  # pylint: disable=import-outside-toplevel

    params = random_raw_params()
    params["label"] = params["opacity"].copy()
    strands_to_world = Rotation.from_rotvec([0.4, -0.7, 0.3]).as_matrix()
    path = str(tmp_path / "strands.ply")
    save_surfels_ply(path, params, strands_to_world_rotation=strands_to_world)
    comments = []
    saved = load_surfels_ply(path, comments=comments)
    assert "coordinate_frame strands" in comments
    np.testing.assert_allclose(saved["xyz"], params["xyz"] @ strands_to_world, atol=3e-7)
    for key in ("features_dc", "label", "opacity", "scaling"):
        np.testing.assert_array_equal(saved[key], params[key])

    world_frames = Rotation.from_quat(params["rotation"][:, [1, 2, 3, 0]]).as_matrix()
    frames = Rotation.from_quat(saved["rotation"][:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(frames, strands_to_world.T @ world_frames, atol=1e-7)
    np.testing.assert_allclose(
        np.linalg.norm(saved["rotation"], axis=1),
        np.linalg.norm(params["rotation"], axis=1),
        rtol=2e-7,
    )
    raw = read_ply(path)
    tangents = np.column_stack([raw[k] for k in ("nx", "ny", "nz")])
    np.testing.assert_allclose(tangents, frames[:, :, 0], atol=1e-7)
    colors = np.column_stack([raw[k] for k in ("red", "green", "blue")])
    np.testing.assert_array_equal(colors, np.rint(boys2rgb(tangents).clip(0, 1) * 255))

    dirs = F.normalize(torch.from_numpy(params["xyz"]), dim=1)
    strand_dirs = dirs @ torch.from_numpy(strands_to_world).float()
    world_sh = torch.from_numpy(
        np.concatenate((params["features_dc"], params["features_rest"]), axis=1)
    )
    strand_sh = torch.from_numpy(
        np.concatenate((saved["features_dc"], saved["features_rest"]), axis=1)
    )
    for degree in range(4):
        torch.testing.assert_close(
            eval_sh(degree, strand_sh, strand_dirs), eval_sh(degree, world_sh, dirs)
        )


def test_polyscope_camera_round_trip():
    angle = 0.4
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    viewmat[:3, 3] = [0.1, 0.2, 3.0]
    K, back = opencv_from_polyscope(CV_TO_GL @ viewmat, 40.0, 320, 240)
    np.testing.assert_allclose(back, viewmat, atol=1e-6)
    assert K[0, 0] == pytest.approx(120.0 / math.tan(math.radians(20.0)))
    assert (K[0, 2], K[1, 2]) == (160.0, 120.0)

    cam = ViewerCamera("c", K, viewmat, 320, 240)
    assert cam.fov_vertical_deg == pytest.approx(40.0)
    # OpenCV y points down, so the world up of an identity camera is -y
    assert up_direction([ViewerCamera("c", K, np.eye(4, dtype=np.float32), 320, 240)]) == "neg_y_up"


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("output_in_mm", [False, True], ids=["meters", "millimeters"])
def test_viewer_uses_ply_coordinate_frame(tmp_path, monkeypatch, output_in_mm):
    # pylint: disable-next=import-outside-toplevel
    from strand_surfels.viewer import __main__ as viewer_cli

    rotation = Rotation.from_rotvec([0.4, -0.7, 0.3]).as_matrix()
    scene = make_scene(str(tmp_path), strands_to_world_rotation=rotation)
    calibration = str(tmp_path / "data" / "calibration.json")
    path = str(tmp_path / "surfels.ply")
    viewer = Mock()
    monkeypatch.setattr(viewer_cli, "SurfelViewer", viewer)
    for export_rotation in (None, rotation):
        save_surfels_ply(
            path,
            random_raw_params(),
            strands_to_world_rotation=export_rotation,
            output_in_mm=output_in_mm,
        )
        viewer_cli.main([path, "--calibration", calibration])
        _, cameras = viewer.call_args.args
        expected = scene["centers"] if export_rotation is None else scene["centers"] @ rotation
        np.testing.assert_allclose(
            [c.center for c in cameras], expected * (1000.0 if output_in_mm else 1.0), atol=1e-3
        )


def small_scene():
    from strand_surfels.rendering import Surfels  # pylint: disable=import-outside-toplevel

    g = torch.Generator().manual_seed(0)
    n = 200
    means = torch.empty(n, 3).uniform_(-0.5, 0.5, generator=g) + torch.tensor([0.0, 0.0, 3.0])
    surfels = Surfels(
        means=means.cuda(),
        quats=F.normalize(torch.randn(n, 4, generator=g), dim=-1).cuda(),
        scales=torch.empty(n, 2).uniform_(0.02, 0.1, generator=g).cuda(),
        opacities=torch.full((n,), 0.8).cuda(),
        sh_coeffs=(torch.randn(n, 16, 3, generator=g) * 0.3).cuda(),
        sh_degree=3,
    )
    K = np.array([[60.0, 0.0, 40.0], [0.0, 60.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return surfels, K, np.eye(4, dtype=np.float32)


@cuda
@pytest.mark.cuda
def test_display_modes():
    # pylint: disable=import-outside-toplevel
    from strand_surfels.losses import line_camera_matrices, orientation_2d
    from strand_surfels.viewer.display import DisplayMode, DisplaySettings, display_image

    surfels, K, viewmat = small_scene()
    with torch.no_grad():
        out = surfels.render(torch.from_numpy(viewmat).cuda(), torch.from_numpy(K).cuda(), 80, 60)
    alpha = out.aux[1]
    for mode in DisplayMode:
        settings = DisplaySettings(mode=mode, composite=True)
        rgba, depth = display_image(out, settings, K, viewmat)
        assert rgba.shape == (60, 80, 4) and depth.shape == (60, 80)
        assert torch.isfinite(rgba).all()
        covered = alpha > settings.alpha_threshold
        assert torch.isinf(depth[~covered]).all() and (depth[covered] > 0).all()
        if mode == DisplayMode.RGB:
            torch.testing.assert_close(rgba[..., :3], out.render.permute(1, 2, 0))
            torch.testing.assert_close(rgba[..., 3], alpha)
        if mode == DisplayMode.ORIENTATION_2D:
            angle = orientation_2d(
                out.aux[0] / alpha.clamp_min(1e-12),
                out.aux[2:5] / alpha.clamp_min(1e-12),
                line_camera_matrices(K, viewmat).to("cuda"),
            )
            expected_red = (torch.cos(angle) + 1) * 0.5 * alpha
            torch.testing.assert_close(rgba[..., 0][covered], expected_red[covered])

    # solid background: opaque everywhere, background where nothing is rendered
    settings = DisplaySettings(background=(0.2, 0.4, 0.6))
    rgba, _ = display_image(out, settings, K, viewmat)
    assert (rgba[..., 3] == 1).all()
    empty = alpha == 0
    torch.testing.assert_close(
        rgba[..., :3][empty], torch.tensor([0.2, 0.4, 0.6]).cuda().expand(int(empty.sum()), 3)
    )


@cuda
@pytest.mark.gui
def test_gpu_and_host_uploads_show_the_same_image():
    # pylint: disable=import-outside-toplevel
    from strand_surfels.viewer import SurfelViewer, display_available

    if not display_available():
        pytest.skip("no display")
    import polyscope as ps

    surfels, K, viewmat = small_scene()
    cam = ViewerCamera("test", K, viewmat, 80, 60)
    viewer = SurfelViewer(surfels, [cam])
    if not viewer.enabled:
        pytest.skip("the viewer could not be started")
    ps.set_window_size(320, 240)
    images = []
    try:
        for interop in (True, False):
            viewer._interop = interop  # pylint: disable=protected-access
            viewer._release_image()  # pylint: disable=protected-access
            viewer.notify_changed()
            for _ in range(3):
                viewer._frame()  # pylint: disable=protected-access
            images.append(ps.screenshot_to_buffer(transparent_bg=False).astype(np.int32))
    finally:
        viewer.close()
    assert images[0].std() > 0  # something is drawn
    assert np.abs(images[0] - images[1]).max() <= 1


@cuda
@pytest.mark.cuda
def test_label_colors_and_filter():
    # pylint: disable=import-outside-toplevel
    from strand_surfels.losses import label_colors, render_labels
    from strand_surfels.losses.labels import HAIR_COLOR, OTHER_COLOR

    surfels, K, viewmat = small_scene()
    n = len(surfels)
    surfels.labels = (torch.arange(n, device="cuda") % 2).float()  # every other one is hair
    Kt, vt = torch.from_numpy(K).cuda(), torch.from_numpy(viewmat).cuda()
    with torch.no_grad():
        for keep, color in (
            (surfels.labels > 0.5, HAIR_COLOR),
            (surfels.labels < 0.5, OTHER_COLOR),
        ):
            part = surfels.select(keep)
            assert len(part) == int(keep.sum())
            out = part.render(vt, Kt, 80, 60, colors=label_colors(part.labels))
            expected = torch.tensor(color).cuda()[:, None, None] * out.alpha
            torch.testing.assert_close(out.render, expected, rtol=1e-5, atol=1e-6)
        # the label rendering blends the probabilities: hair fraction <= alpha
        out = render_labels(surfels, vt, Kt, 80, 60)
        assert (out.render[0] <= out.alpha[0] + 1e-6).all() and out.render[0].max() > 0
