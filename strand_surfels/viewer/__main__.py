"""View an optimised surfel point cloud.

Example::

    python -m strand_surfels.viewer output/jewfro/point_cloud/iteration_15000/point_cloud.ply \\
        --calibration data/jewfro/calibration.json
"""

from __future__ import annotations

import argparse
import logging

import torch

from strand_surfels.io import load_surfels_ply
from strand_surfels.rendering import Surfels
from strand_surfels.viewer.cameras import load_calibration
from strand_surfels.viewer.viewer import SurfelViewer


def main(argv=None) -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("ply", help="surfel point cloud (.ply)")
    parser.add_argument(
        "--calibration", help="calibration.json of the dataset, to show its cameras"
    )
    parser.add_argument(
        "--downscale", type=float, default=1.0, help="image downscale of the cameras"
    )
    parser.add_argument(
        "--sh-degree", type=int, default=None, help="SH degree (default: all stored)"
    )
    parser.add_argument(
        "--no-interop", action="store_true", help="upload images through host memory"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    params = {k: torch.from_numpy(v).cuda() for k, v in load_surfels_ply(args.ply).items()}
    surfels = Surfels.from_raw(**params, sh_degree=args.sh_degree)
    cameras = load_calibration(args.calibration, args.downscale) if args.calibration else []
    print(f"{len(surfels):,} surfels (SH degree {surfels.sh_degree}), {len(cameras)} cameras")

    viewer = SurfelViewer(surfels, cameras, interop=not args.no_interop)
    if not viewer.enabled:
        raise SystemExit("The viewer could not be started (no display?)")
    viewer.run()


if __name__ == "__main__":
    main()
