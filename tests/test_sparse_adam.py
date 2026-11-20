"""SparseAdam against a dense PyTorch implementation of the same update."""

# pylint: disable=missing-function-docstring

import pytest
import torch

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from strand_surfels.optim import SparseAdam  # noqa: E402  # pylint: disable=wrong-import-position


def test_sparse_adam_updates_visible_surfels_only():
    torch.manual_seed(0)
    N = 1000
    p = torch.randn(N, 3, 2, device="cuda", requires_grad=True)
    ref = p.detach().clone()
    m, v = torch.zeros_like(ref), torch.zeros_like(ref)
    opt = SparseAdam([{"params": [p], "lr": 0.01, "name": "x"}], eps=1e-15)
    b1, b2, lr, eps = 0.9, 0.999, 0.01, 1e-15
    for _ in range(5):
        grad = torch.randn_like(ref)
        visible = torch.rand(N, device="cuda") > 0.5
        p.grad = grad.clone()
        opt.step(visible)
        # reference: masked Adam without bias correction
        vis = visible[:, None, None]
        m = torch.where(vis, b1 * m + (1 - b1) * grad, m)
        v = torch.where(vis, b2 * v + (1 - b2) * grad * grad, v)
        ref = torch.where(vis, ref - lr * m / (v.sqrt() + eps), ref)
    torch.testing.assert_close(p.detach(), ref)
    torch.testing.assert_close(opt.state[p]["exp_avg"], m)
    torch.testing.assert_close(opt.state[p]["exp_avg_sq"], v)
