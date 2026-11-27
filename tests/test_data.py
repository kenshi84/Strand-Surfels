"""Dataset loading, scene extent and neighbour distances on a synthetic scene."""

# pylint: disable=missing-function-docstring

import json
import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

from strand_surfels.data import (
    HairDataset,
    camera_extent,
    load_calibration,
    load_initial_points,
    mean_squared_neighbor_distance,
)

W, H = 32, 24


def rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def make_scene(root: str, n_cameras: int = 4) -> dict:
    """data/ and output/ folders of a tiny scene; returns the camera centres."""
    data, out = os.path.join(root, "data"), os.path.join(root, "output")
    for d in ("image", "hair", "body"):
        os.makedirs(os.path.join(data, d))
    for d in ("int", "filtered"):
        os.makedirs(os.path.join(out, d))
    rng = np.random.default_rng(0)
    cameras, centers = [], []
    for i in range(n_cameras):
        # cameras on a circle, looking at the origin, with a roll so that
        # R is far from symmetric
        R = rotation("z", 0.3 * i) @ rotation("y", -2 * math.pi * i / n_cameras)
        center = R.T @ np.array([0.0, 0.0, -2.0]) + np.array([0.1, 0.2, 0.3])
        t = -R @ center
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = R, t
        K = [40.0, 0, W / 2, 0, 40.0, H / 2, 0, 0, 1]
        cameras.append(
            {
                "camera_id": f"C{i:06d}",
                "extrinsics": {"view_matrix": view.ravel().tolist()},
                "intrinsics": {"camera_matrix": K, "resolution": [H, W]},
            }
        )
        centers.append(center)
        Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(
            os.path.join(data, "image", f"img_{i:06d}.png")
        )
        Image.fromarray(rng.integers(0, 256, (H, W), dtype=np.uint8)).save(
            os.path.join(data, "hair", f"img_{i:06d}.png")
        )
        if i != 1:  # camera 1 has no body mask
            body = np.zeros((H, W), np.uint8)
            body[:, : W // 2] = 255
            Image.fromarray(body).save(os.path.join(data, "body", f"img_{i:06d}.png"))
        if i != 2:  # camera 2 has no int map
            int_map = rng.normal(size=(H, W, 3)).astype(np.float32)
            np.save(os.path.join(out, "int", f"int_map_{i:06d}.npy"), int_map)
    with open(os.path.join(data, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"cameras": cameras}, f)

    positions = rng.normal(size=(1000, 4)).astype(np.float32)
    directions = rng.normal(size=(1000, 4)).astype(np.float32)
    np.save(os.path.join(out, "filtered", "filtered_positions.npy"), positions)
    np.save(os.path.join(out, "filtered", "filtered_directions.npy"), directions)
    return {"data": data, "output": out, "centers": np.stack(centers), "positions": positions}


def test_camera_centres_and_extent(tmp_path):
    scene = make_scene(str(tmp_path))
    cameras = load_calibration(os.path.join(scene["data"], "calibration.json"))
    np.testing.assert_allclose(np.stack([c.center for c in cameras]), scene["centers"], atol=1e-5)
    center, radius = camera_extent(cameras)
    expected = scene["centers"].mean(0)
    np.testing.assert_allclose(center, expected, atol=1e-5)
    assert radius == pytest.approx(
        1.1 * np.linalg.norm(scene["centers"] - expected, axis=1).max(), rel=1e-5
    )
    # the rotation is not symmetric: the transposed rotation gives other centres
    wrong = np.stack([-c.R @ c.t for c in cameras])
    assert np.abs(wrong - scene["centers"]).max() > 0.1


def test_views(tmp_path):
    scene = make_scene(str(tmp_path))
    dataset = HairDataset(scene["data"], scene["output"], num_workers=2)
    assert len(dataset) == 4
    for i, view in enumerate(dataset.views):
        data = view.load("cpu")
        image = np.array(Image.open(os.path.join(scene["data"], "image", f"img_{i:06d}.png")))
        hair = np.array(Image.open(os.path.join(scene["data"], "hair", f"img_{i:06d}.png")))
        body = (np.arange(W) < W // 2)[None].repeat(H, 0)
        # camera 1 has no body mask: foreground where the image is not black
        alpha = image.max(-1) > 0 if i == 1 else body
        expected = (
            torch.from_numpy(image).permute(2, 0, 1).float()
            / 255.0
            * torch.from_numpy(alpha).float()
        )
        torch.testing.assert_close(data.image, expected)
        assert torch.equal(data.alpha_mask[0], torch.from_numpy(alpha).float())
        assert torch.equal(data.hair_mask[0], torch.from_numpy(hair / 255.0 > 0.01).float())
        assert data.image.shape == (3, H, W) and data.int_map.shape == (3, H, W)
        if i == 2:  # missing int map: no measurement, zero confidence
            assert (data.int_map[0] == -1).all() and (data.int_map[1:] == 0).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_missing_int_map_gives_zero_geometry_loss(tmp_path):
    from strand_surfels.losses import (  # pylint: disable=import-outside-toplevel
        direction_to_structure_tensor,
        geometry_loss,
    )

    scene = make_scene(str(tmp_path))
    data = HairDataset(scene["data"], scene["output"])[2].load("cuda")
    depth = torch.full((H, W), 2.0, device="cuda")
    dirs = torch.nn.functional.normalize(torch.randn(3, H, W, device="cuda"), dim=0)
    tensors = direction_to_structure_tensor(dirs)
    assert (geometry_loss(depth, tensors, data.int_map, data.line_camera) == 0).all()


def test_foreground_from_colors(tmp_path):
    scene = make_scene(str(tmp_path))
    path = os.path.join(scene["data"], "image", "img_000001.png")
    image = np.array(Image.open(path))
    image[:5] = 0  # black background rows
    image[5, :, 0] = 3  # dark but not black
    Image.fromarray(image).save(path)
    view = HairDataset(scene["data"], scene["output"], foreground_threshold=0)[1].load("cpu")
    assert (view.alpha_mask[0, :5] == 0).all() and (view.alpha_mask[0, 5] == 1).all()
    view = HairDataset(scene["data"], scene["output"], foreground_threshold=3)[1].load("cpu")
    assert (view.alpha_mask[0, 5][image[5].max(-1) <= 3] == 0).all()
    view = HairDataset(scene["data"], scene["output"], foreground_threshold=None)[1].load("cpu")
    assert (view.alpha_mask == 1).all()


def test_downscaled_views(tmp_path):
    scene = make_scene(str(tmp_path))
    dataset = HairDataset(scene["data"], scene["output"], downscale=2.0)
    cam = dataset.cameras[0]
    assert (cam.width, cam.height) == (W // 2, H // 2)
    assert cam.K[0, 0] == pytest.approx(20.0) and cam.K[0, 2] == pytest.approx(W / 4)
    data = dataset[0].load("cpu")
    assert data.image.shape == (3, H // 2, W // 2)
    assert data.alpha_mask.shape == data.hair_mask.shape == (1, H // 2, W // 2)
    assert 0.0 <= data.image.min() and data.image.max() <= 1.0


def test_initial_points(tmp_path):
    scene = make_scene(str(tmp_path))
    points = load_initial_points(scene["output"], stride=7)
    np.testing.assert_array_equal(points.xyz, scene["positions"][::7, :3])
    assert points.dirs.shape == (len(points), 3) and points.gray.shape == (len(points), 1)


def test_neighbor_distances_match_brute_force():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(300, 3)).astype(np.float32)
    pts[10] = pts[11]  # coincident points
    pts[12] = pts[11]
    d2 = ((pts[:, None].astype(np.float64) - pts[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    expected = np.sort(d2, axis=1)[:, :3].mean(1)
    np.testing.assert_allclose(mean_squared_neighbor_distance(pts), expected, rtol=1e-5, atol=1e-12)
