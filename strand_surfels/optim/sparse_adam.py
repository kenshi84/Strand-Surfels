"""Adam that only updates the surfels visible in the current view.

With one view per iteration most surfels receive no gradient, and a dense Adam
step would still decay their moments and move them. :class:`SparseAdam`
updates parameters and moments of the visible surfels only, which is both
faster and closer to the per-surfel statistics one wants (Taming 3DGS). As in
that reference implementation, no bias correction is applied.

Every parameter group must hold exactly one tensor whose first dimension is
the number of surfels; the optimizer state uses the keys of
:class:`torch.optim.Adam` (``exp_avg``, ``exp_avg_sq``), so code that prunes
or extends the state (densification) works for both.
"""

from __future__ import annotations

import torch
from torch import Tensor

from strand_surfels._backend import _C


class SparseAdam(torch.optim.Adam):
    """Adam restricted to visible surfels; call ``step(visibility)``."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-15) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps)

    @torch.no_grad()
    def step(self, visibility: Tensor) -> None:  # type: ignore[override]  # pylint: disable=arguments-differ,arguments-renamed
        """Update the surfels with ``visibility[i] == True`` (bool tensor [N])."""
        visibility = visibility.contiguous()
        for group in self.param_groups:
            assert len(group["params"]) == 1, "SparseAdam expects one tensor per group"
            param = group["params"][0]
            if param.grad is None:
                continue
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            beta1, beta2 = group["betas"]
            _C.sparse_adam_update(
                param,
                param.grad.contiguous(),
                state["exp_avg"],
                state["exp_avg_sq"],
                visibility,
                group["lr"],
                beta1,
                beta2,
                group["eps"],
            )
