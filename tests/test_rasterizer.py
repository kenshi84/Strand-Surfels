"""Forward and backward pass of the CUDA rasterizer against the PyTorch reference."""

# pylint: disable=missing-function-docstring

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

# pylint: disable=wrong-import-position
from reference_rasterizer import project, rasterize  # noqa: E402

from strand_surfels.rendering import rasterize_surfels  # noqa: E402

WIDTH, HEIGHT = 40, 36  # several tiles, including partial ones
BACKGROUND = torch.tensor([0.1, 0.2, 0.3])


def random_scene(seed: int, n: int = 80):
    """Surfels in front of a camera looking down +z (OpenCV)."""
    g = torch.Generator().manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-0.6, 0.6, generator=g)
    means[:, 2] = torch.empty(n).uniform_(2.0, 4.0, generator=g)
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    scales = torch.empty(n, 2).uniform_(0.03, 0.25, generator=g)
    opacities = torch.empty(n).uniform_(0.2, 0.95, generator=g)
    colors = torch.rand(n, 3, generator=g)
    viewmat = torch.eye(4)
    viewmat[:3, 3] = torch.tensor([0.05, -0.02, 0.1])
    K = torch.tensor([[50.0, 0.0, WIDTH / 2 + 0.3], [0.0, 52.0, HEIGHT / 2 - 0.4], [0.0, 0.0, 1.0]])
    tensors = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    return {k: v.cuda() for k, v in tensors.items()}, viewmat.cuda(), K.cuda()


def leaves(scene):
    return {k: v.clone().requires_grad_(True) for k, v in scene.items()}


def upstream(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(3, HEIGHT, WIDTH, device="cuda", generator=g),
        torch.randn(10, HEIGHT, WIDTH, device="cuda", generator=g),
        torch.randn(1, HEIGHT, WIDTH, device="cuda", generator=g),
    )


def run_cuda(scene, viewmat, K, up):
    p = leaves(scene)
    pse = torch.zeros(len(scene["means"]), device="cuda", requires_grad=True)
    out = rasterize_surfels(
        p["means"], p["quats"], p["scales"], p["opacities"], p["colors"], viewmat, K, WIDTH, HEIGHT,
        background=BACKGROUND.cuda(), pse=pse,
    )  # fmt: skip
    loss = (out.render * up[0]).sum() + (out.aux * up[1]).sum() + (out.error_map * up[2]).sum()
    loss.backward()
    grads = {k: v.grad for k, v in p.items()} | {"pse": pse.grad}
    return out, grads


def run_reference(scene, viewmat, K, up):
    p = leaves(scene)
    pse = torch.zeros(len(scene["means"]), device="cuda", requires_grad=True)
    radii, means2d, depths, ray_transforms, dirs = project(
        p["means"], p["quats"], p["scales"], p["opacities"], viewmat, K, WIDTH, HEIGHT
    )
    render, aux, error = rasterize(
        radii, means2d, depths, ray_transforms, dirs, p["colors"], p["opacities"], pse,
        BACKGROUND.cuda(), WIDTH, HEIGHT,
    )  # fmt: skip
    loss = (render * up[0]).sum() + (aux * up[1]).sum() + (error * up[2]).sum()
    loss.backward()
    grads = {k: v.grad for k, v in p.items()} | {"pse": pse.grad}
    return (radii, render, aux), grads


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_and_backward_match_reference(seed):
    scene, viewmat, K = random_scene(seed)
    up = upstream(seed)
    out, grads = run_cuda(scene, viewmat, K, up)
    (radii, render, aux), ref_grads = run_reference(scene, viewmat, K, up)

    assert torch.equal(out.radii, radii)
    torch.testing.assert_close(out.render, render, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out.aux, aux, atol=1e-3, rtol=1e-4)
    # Float32 rounding of the two implementations differs; against a float64
    # reference the kernel and the float32 reference show the same error
    # (up to ~2e-3 relative for badly conditioned surfels).
    for name, ref in ref_grads.items():
        rel = ((grads[name] - ref).norm() / ref.norm()).item()
        assert rel < 5e-3, f"gradient of {name}: relative error {rel:.2e}"


def test_no_grad_rendering_matches():
    scene, viewmat, K = random_scene(5)
    with torch.no_grad():
        a = rasterize_surfels(
            **{k: scene[k] for k in ("means", "quats", "scales", "opacities", "colors")},
            viewmat=viewmat,
            K=K,
            width=WIDTH,
            height=HEIGHT,
            background=BACKGROUND.cuda(),
        )
    b, _ = run_cuda(scene, viewmat, K, upstream(5))
    torch.testing.assert_close(a.render, b.render.detach())
    torch.testing.assert_close(a.aux, b.aux.detach())


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_kernel_matches_reference(degree):
    from strand_surfels.rendering import sh_to_color  # pylint: disable=import-outside-toplevel

    scene, viewmat, K = random_scene(7)
    g = torch.Generator(device="cuda").manual_seed(degree)
    coeffs = torch.randn(len(scene["means"]), 16, 3, device="cuda", generator=g) * 0.3
    up = upstream(degree)

    def run(use_kernel):
        c = coeffs.clone().requires_grad_(True)
        m = scene["means"].clone().requires_grad_(True)
        kwargs = {"colors": None, "sh_coeffs": c, "sh_degree": degree}
        if not use_kernel:
            center = -viewmat[:3, :3].T @ viewmat[:3, 3]
            kwargs = {"colors": sh_to_color(degree, c, m, center)}
        out = rasterize_surfels(
            m,
            scene["quats"],
            scene["scales"],
            scene["opacities"],
            viewmat=viewmat,
            K=K,
            width=WIDTH,
            height=HEIGHT,
            **kwargs,
        )
        ((out.render * up[0]).sum()).backward()
        return out.render.detach(), c.grad, m.grad

    for a, b in zip(run(True), run(False)):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
