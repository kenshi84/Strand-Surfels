"""Real spherical harmonics up to degree 3 for view dependent surfel colors.

The basis, its normalisation constants and sign convention are the ones used
by 3D Gaussian Splatting, so colors are ``0.5 + sum_k c_k Y_k(d)``, clamped at
zero, where ``d`` is the unit direction from the camera to the surfel.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def num_sh_bases(degree: int) -> int:
    """Number of SH coefficients per color channel for a given degree."""
    return (degree + 1) ** 2


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Degree-0 coefficient that reproduces a constant color."""
    return (rgb - 0.5) / SH_C0


def eval_sh(
    degree: int,
    coeffs: Float[Tensor, "N K 3"],
    dirs: Float[Tensor, "N 3"],
) -> Float[Tensor, "N 3"]:
    """Evaluate SH coefficients ``coeffs`` (the first ``(degree+1)^2`` are used)
    in the unit directions ``dirs``."""
    assert 0 <= degree <= 3, "SH degree must be in [0, 3]"
    assert coeffs.shape[-2] >= num_sh_bases(degree)
    result = SH_C0 * coeffs[:, 0]
    if degree < 1:
        return result

    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    result = result - SH_C1 * y * coeffs[:, 1] + SH_C1 * z * coeffs[:, 2] - SH_C1 * x * coeffs[:, 3]
    if degree < 2:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + SH_C2[0] * xy * coeffs[:, 4]
        + SH_C2[1] * yz * coeffs[:, 5]
        + SH_C2[2] * (2.0 * zz - xx - yy) * coeffs[:, 6]
        + SH_C2[3] * xz * coeffs[:, 7]
        + SH_C2[4] * (xx - yy) * coeffs[:, 8]
    )
    if degree < 3:
        return result

    return (
        result
        + SH_C3[0] * y * (3.0 * xx - yy) * coeffs[:, 9]
        + SH_C3[1] * xy * z * coeffs[:, 10]
        + SH_C3[2] * y * (4.0 * zz - xx - yy) * coeffs[:, 11]
        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffs[:, 12]
        + SH_C3[4] * x * (4.0 * zz - xx - yy) * coeffs[:, 13]
        + SH_C3[5] * z * (xx - yy) * coeffs[:, 14]
        + SH_C3[6] * x * (xx - 3.0 * yy) * coeffs[:, 15]
    )


def sh_to_color(
    degree: int,
    coeffs: Float[Tensor, "N K 3"],
    means: Float[Tensor, "N 3"],
    camera_center: Float[Tensor, "3"],
) -> Float[Tensor, "N 3"]:
    """View dependent RGB colors of surfels seen from ``camera_center``."""
    dirs = means - camera_center[None]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return torch.clamp_min(eval_sh(degree, coeffs, dirs) + 0.5, 0.0)
