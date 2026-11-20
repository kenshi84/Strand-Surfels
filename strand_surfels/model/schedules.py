"""Learning rate and densification budget schedules."""

from __future__ import annotations

from typing import Callable, List

import numpy as np


def exponential_lr(
    lr_init: float,
    lr_final: float,
    max_steps: int,
    delay_steps: int = 0,
    delay_mult: float = 1.0,
) -> Callable[[int], float]:
    """Log-linear decay from ``lr_init`` (step 0) to ``lr_final`` (``max_steps``).

    With ``delay_steps > 0`` the rate starts at ``lr_init * delay_mult`` and is
    eased back with a quarter sine over the first ``delay_steps`` steps (the
    schedule of Plenoxels / 3D Gaussian Splatting).
    """

    def lr(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if delay_steps > 0:
            delay = delay_mult + (1 - delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / delay_steps, 0, 1)
            )
        else:
            delay = 1.0
        t = np.clip(step / max_steps, 0, 1)
        return float(delay * np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t))

    return lr


def densification_budgets(
    start_count: int, final_count: int, densify_from: int, densify_until: int, interval: int
) -> List[int]:
    """Surfel budget after each densification step (Taming 3DGS, Eq. 2).

    The count grows along a concave parabola from ``start_count`` towards
    ``final_count``: fast at first, levelling off at the end of the
    densification period. Entry ``i`` is the budget of the ``i``-th step.
    """
    num_steps = (densify_until - densify_from) // interval
    if num_steps <= 0:
        return []
    slope = (final_count - start_count) / num_steps
    b = 2 * slope
    a = (final_count - start_count - b * num_steps) / (num_steps * num_steps)
    return [int(a * x**2 + b * x + start_count) for x in range(num_steps)]
