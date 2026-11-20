"""Initial points from the line-based multi-view stereo stage (SMHC).

``output/<scene>/filtered/`` holds the filtered 3D line samples:

* ``filtered_positions.npy`` ``[M, 4]``: x, y, z and depth,
* ``filtered_directions.npy`` ``[M, 4]``: unit strand direction and a gray value.

A regular subsample of them initialises the surfels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class InitialPoints:
    """Strand samples used to initialise the surfels."""

    xyz: np.ndarray  # [N, 3] positions
    dirs: np.ndarray  # [N, 3] unit strand directions
    gray: np.ndarray  # [N, 1] gray values in [0, 1]

    def __len__(self) -> int:
        return self.xyz.shape[0]


def load_initial_points(output_dir: str, stride: int = 100) -> InitialPoints:
    """Every ``stride``-th filtered line sample of ``output_dir/filtered``."""
    folder = os.path.join(output_dir, "filtered")
    positions = np.load(os.path.join(folder, "filtered_positions.npy"))
    directions = np.load(os.path.join(folder, "filtered_directions.npy"))
    return InitialPoints(
        xyz=positions[::stride, :3].astype(np.float32).copy(),
        dirs=directions[::stride, :3].astype(np.float32).copy(),
        gray=directions[::stride, 3:4].astype(np.float32).copy(),
    )


def mean_squared_neighbor_distance(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean squared distance ``[N]`` of each point to its ``k`` nearest neighbours.

    Exact k-nearest neighbours with a k-d tree; coincident points count as
    neighbours at distance zero.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=np.float32)
    k = min(k, points.shape[0] - 1)
    tree = cKDTree(points)
    # the nearest "neighbour" of a point is the point itself
    dist, idx = tree.query(points, k=k + 1, workers=-1)
    own = idx == np.arange(points.shape[0])[:, None]
    # drop the point itself (or, with duplicates, one of the zero distances)
    keep = ~own
    keep[keep.sum(axis=1) > k, -1] = False
    sq = np.where(keep, dist.astype(np.float32) ** 2, 0.0).sum(axis=1)
    return (sq / k).astype(np.float32)
