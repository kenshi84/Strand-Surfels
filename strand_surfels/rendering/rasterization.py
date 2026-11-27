"""Differentiable rasterization of 2D Gaussian surfels with strand directions.

The pipeline has three stages, each backed by CUDA kernels in
``strand_surfels/csrc``:

1. **projection** (:class:`_ProjectSurfels`): per surfel, the ray transform that
   maps surfel coordinates to pixels, the screen space bounding box and the
   world space strand direction;
2. **tile binning** (``intersect_tiles``): surfel/tile pairs sorted by tile and
   depth, plus the number of backward *buckets* (groups of 32 surfels) per tile;
3. **rasterization** (:class:`_RasterizeSurfels`): front-to-back alpha blending
   per pixel. The forward pass samples the blending state at the start of
   every bucket so that the backward pass can process each bucket in parallel
   (one warp per bucket) instead of each pixel sequentially (Taming 3DGS).

Besides the color image, the rasterizer renders an auxiliary image
(:data:`AUX_CHANNELS`) with the alpha-weighted depth, the accumulated opacity,
the alpha-weighted world-space direction structure tensor, the depth distortion
and the median depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from strand_surfels._backend import _C

# torch.autograd.Function subclasses override forward/backward with explicit
# signatures by design
# pylint: disable=abstract-method,arguments-differ

#: Channels of the auxiliary image ``RasterOutput.aux`` (index -> name).
AUX_CHANNELS = (
    "depth",
    "alpha",
    "tensor_xx",
    "tensor_xy",
    "tensor_xz",
    "tensor_yy",
    "tensor_yz",
    "tensor_zz",
    "distortion",
    "median_depth",
)


class SampleBuffer:
    """Reusable GPU storage for the per-bucket blending state samples.

    The number of buckets changes every iteration, and allocating the samples
    anew each time fragments the CUDA memory pool. A ``SampleBuffer`` keeps one
    allocation and grows it (by 25 % head room) only when needed.

    The samples of a forward pass are consumed by its backward pass, so a
    buffer must not be shared by two forward passes whose backward passes are
    still pending. This is checked: a backward pass whose samples have been
    overwritten raises an error.
    """

    def __init__(self, capacity: int = 0, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.capacity = 0
        self.generation = 0
        self._tensors: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        if capacity > 0:
            self._allocate(capacity)

    def _allocate(self, capacity: int) -> None:
        self._tensors = None  # release before allocating the larger buffer
        tp = _C.TILE_PIXELS
        self._tensors = (
            torch.empty(capacity, dtype=torch.int32, device=self.device),
            torch.empty(capacity, _C.SAMPLED_STATE_DIM, tp, device=self.device),
            torch.empty(capacity, 3, tp, device=self.device),
            torch.empty(capacity, _C.SAMPLED_AUX_DIM, tp, device=self.device),
        )
        self.capacity = capacity

    def acquire(self, n_buckets: int) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], int]:
        """Views for ``n_buckets`` buckets and the generation stamp of this use."""
        if n_buckets > self.capacity:
            self._allocate(max(int(n_buckets * 1.25), 1))
        self.generation += 1
        assert self._tensors is not None
        views = tuple(t[: max(n_buckets, 1)] for t in self._tensors)
        return views, self.generation  # type: ignore[return-value]

    def release(self) -> None:
        """Free the GPU memory (the buffer is re-allocated on its next use)."""
        self._tensors = None
        self.capacity = 0


@dataclass
class RasterOutput:
    """Result of :func:`rasterize_surfels`.

    Image tensors are channel-first. ``depth`` and the direction tensor of
    ``aux`` are alpha-weighted sums; divide by ``alpha`` for expected values.
    """

    render: Float[Tensor, "3 H W"]  # color composited over the background
    aux: Float[Tensor, "10 H W"]  # see AUX_CHANNELS
    error_map: Float[Tensor, "1 H W"]  # zeros; its gradient drives the per-surfel error
    radii: Int[Tensor, " N"]  # screen space radius, 0 for culled surfels
    means2d: Float[Tensor, "N 2"]  # screen space centers
    depths: Float[Tensor, " N"]  # camera depth of the centers
    n_isects: int  # number of surfel/tile pairs
    n_buckets: int  # number of backward buckets

    @property
    def alpha(self) -> Float[Tensor, "1 H W"]:
        """Accumulated opacity."""
        return self.aux[1:2]

    @property
    def visibility(self) -> Tensor:
        """Mask of the surfels that were rasterized."""
        return self.radii > 0


class _ProjectSurfels(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Surfel projection; differentiable w.r.t. means, quats and scales."""

    @staticmethod
    def forward(ctx, means, quats, scales, opacities, viewmat, K, width, height, scale_modifier):
        radii, means2d, depths, ray_transforms, dirs = _C.project_surfels_fwd(
            means, quats, scales, opacities, viewmat, K, width, height, scale_modifier
        )
        ctx.save_for_backward(quats, scales, opacities, viewmat, K, radii, ray_transforms)
        ctx.scale_modifier = scale_modifier
        ctx.mark_non_differentiable(radii, depths)
        return radii, means2d, depths, ray_transforms, dirs

    @staticmethod
    def backward(ctx, _v_radii, v_means2d, _v_depths, v_ray_transforms, v_dirs):
        quats, scales, opacities, viewmat, K, radii, ray_transforms = ctx.saved_tensors
        v_means, v_quats, v_scales = _C.project_surfels_bwd(
            quats,
            scales,
            opacities,
            viewmat,
            K,
            ctx.scale_modifier,
            radii,
            ray_transforms,
            v_means2d.contiguous(),
            v_ray_transforms.contiguous(),
            v_dirs.contiguous(),
        )
        # the opacity only enters through the bounding box cut-off, which is
        # treated as constant (as in the legacy rasterizer)
        return v_means, v_quats, v_scales, None, None, None, None, None, None


class _RasterizeSurfels(torch.autograd.Function):
    """Alpha blending of the binned surfels."""

    @staticmethod
    def forward(
        ctx,
        means2d,
        ray_transforms,
        colors,
        dirs,
        opacities,
        pse,
        densify,
        background,
        width,
        height,
        tile_ranges,
        flatten_ids,
        bucket_offsets,
        n_buckets,
        sample_buffer,
    ):
        del pse, densify  # inputs only to receive gradients
        save_samples = any(ctx.needs_input_grad[:7])
        if save_samples:
            buffer = sample_buffer if sample_buffer is not None else SampleBuffer(0, means2d.device)
            samples, generation = buffer.acquire(n_buckets)
        else:
            buffer, generation = None, 0
            empty = torch.empty(0, device=means2d.device)
            samples = (empty.int(), empty, empty, empty)

        render, aux, final_state, contributors, max_contrib = _C.rasterize_fwd(
            means2d,
            ray_transforms,
            colors,
            dirs,
            opacities,
            background,
            width,
            height,
            tile_ranges,
            flatten_ids,
            bucket_offsets,
            save_samples,
            *samples,
        )
        error_map = torch.zeros(1, height, width, device=means2d.device)

        if save_samples:
            ctx.save_for_backward(
                means2d,
                ray_transforms,
                colors,
                dirs,
                opacities,
                background,
                tile_ranges,
                flatten_ids,
                bucket_offsets,
                render,
                aux,
                final_state,
                contributors,
                max_contrib,
                *samples,
            )
            ctx.buffer = buffer
            ctx.generation = generation
            ctx.n_buckets = n_buckets
            ctx.size = (width, height)
        return render, aux, error_map

    @staticmethod
    def backward(ctx, v_render, v_aux, v_error_map):
        if ctx.buffer.generation != ctx.generation:
            raise RuntimeError(
                "The SampleBuffer of this rasterization was reused by another forward pass "
                "before the backward pass. Use separate buffers, or run the other pass "
                "under torch.no_grad()."
            )
        (
            means2d,
            ray_transforms,
            colors,
            dirs,
            opacities,
            background,
            tile_ranges,
            flatten_ids,
            bucket_offsets,
            render,
            aux,
            final_state,
            contributors,
            max_contrib,
            bucket_to_tile,
            sampled_state,
            sampled_color,
            sampled_aux,
        ) = ctx.saved_tensors
        width, height = ctx.size
        v_render = torch.zeros_like(render) if v_render is None else v_render.contiguous()
        v_aux = torch.zeros_like(aux) if v_aux is None else v_aux.contiguous()
        if v_error_map is None:
            v_error_map = torch.zeros(1, height, width, device=render.device)

        v_means2d, v_ray_transforms, v_colors, v_dirs, v_opacities, v_pse = _C.rasterize_bwd(
            means2d,
            ray_transforms,
            colors,
            dirs,
            opacities,
            background,
            width,
            height,
            tile_ranges,
            flatten_ids,
            bucket_offsets,
            bucket_to_tile,
            max_contrib,
            sampled_state,
            sampled_color,
            sampled_aux,
            ctx.n_buckets,
            render,
            aux,
            final_state,
            contributors,
            v_render,
            v_aux,
            v_error_map.contiguous(),
        )

        v_densify = None
        if ctx.needs_input_grad[6]:
            # screen space positional gradient of 2DGS used for densification:
            # gradient of the translation part of the ray transform scaled to NDC
            ndc_scale = torch.tensor([0.5 * width, 0.5 * height], device=means2d.device)
            v_densify = v_ray_transforms[:, [2, 5]] * ray_transforms[:, 8:9] * ndc_scale

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_dirs,
            v_opacities,
            v_pse if ctx.needs_input_grad[5] else None,
            v_densify,
        ) + (None,) * 8


class _SHToColor(torch.autograd.Function):
    """View dependent colors of the visible surfels from SH coefficients."""

    @staticmethod
    def forward(ctx, degree, coeffs, means, camera_center, radii):
        colors = _C.sh_to_color_fwd(degree, coeffs, means, camera_center, radii)
        ctx.save_for_backward(coeffs, means, camera_center, radii, colors)
        ctx.degree = degree
        return colors

    @staticmethod
    def backward(ctx, v_colors):
        coeffs, means, camera_center, radii, colors = ctx.saved_tensors
        v_coeffs, v_means = _C.sh_to_color_bwd(
            ctx.degree, coeffs, means, camera_center, radii, colors, v_colors.contiguous()
        )
        return None, v_coeffs, v_means, None, None


def rasterize_surfels(
    means: Float[Tensor, "N 3"],
    quats: Float[Tensor, "N 4"],
    scales: Float[Tensor, "N 2"],
    opacities: Float[Tensor, " N"],
    colors: Optional[Float[Tensor, "N 3"]],
    viewmat: Float[Tensor, "4 4"],
    K: Float[Tensor, "3 3"],
    width: int,
    height: int,
    background: Optional[Float[Tensor, "3"]] = None,
    pse: Optional[Float[Tensor, " N"]] = None,
    densify: Optional[Float[Tensor, "N 2"]] = None,
    scale_modifier: float = 1.0,
    sample_buffer: Optional[SampleBuffer] = None,
    sh_coeffs: Optional[Float[Tensor, "N K 3"]] = None,
    sh_degree: int = 0,
) -> RasterOutput:
    """Render surfels into one pinhole camera.

    Args:
        means: surfel centers (world space).
        quats: surfel rotations as quaternions (w, x, y, z), not necessarily
            normalised. The first axis of the rotation is the strand direction,
            the third axis the surfel normal.
        scales: extents of the surfel along its two tangent axes (activated).
        opacities: surfel opacities in (0, 1) (activated).
        colors: RGB color of every surfel, or ``None`` to evaluate
            ``sh_coeffs`` (faster: fused kernel, visible surfels only).
        viewmat: world-to-camera matrix (OpenCV convention).
        K: camera intrinsics (OpenCV convention, pixel centers at +0.5).
        width, height: image size in pixels.
        background: RGB background color, black by default.
        pse: optional zero tensor with ``requires_grad``; after the backward
            pass its gradient is the per-surfel error score of Taming 3DGS, i.e.
            the sum over pixels of the blending weight times the gradient
            arriving at ``error_map``.
        densify: optional zero tensor with ``requires_grad``; its gradient is
            the screen space positional gradient of 2DGS used to decide which
            surfels to densify.
        scale_modifier: global factor on all scales (visualisation only).
        sample_buffer: storage reused across calls for the backward samples.
        sh_coeffs: spherical harmonics coefficients of the colors (used when
            ``colors`` is ``None``); see :mod:`strand_surfels.rendering.sh`.
        sh_degree: active SH degree, ``(sh_degree + 1)^2 <= K``.

    Returns:
        A :class:`RasterOutput`.
    """
    device = means.device
    N = means.shape[0]
    if background is None:
        background = torch.zeros(3, device=device)
    if pse is None:
        pse = torch.zeros(N, device=device)
    if densify is None:
        densify = torch.zeros(N, 2, device=device)

    radii, means2d, depths, ray_transforms, dirs = _ProjectSurfels.apply(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        opacities.contiguous(),
        viewmat.contiguous(),
        K.contiguous(),
        width,
        height,
        scale_modifier,
    )
    if colors is None:
        if sh_coeffs is None:
            raise ValueError("either colors or sh_coeffs must be given")
        R, t = viewmat[:3, :3], viewmat[:3, 3]
        camera_center = (-R.T @ t).contiguous()
        colors = _SHToColor.apply(
            sh_degree, sh_coeffs.contiguous(), means.contiguous(), camera_center, radii
        )
    tile_ranges, flatten_ids, bucket_offsets, n_buckets = _C.intersect_tiles(
        means2d.detach(), radii, depths, width, height
    )
    render, aux, error_map = _RasterizeSurfels.apply(
        means2d,
        ray_transforms,
        colors.contiguous(),
        dirs,
        opacities.contiguous(),
        pse,
        densify,
        background.contiguous(),
        width,
        height,
        tile_ranges,
        flatten_ids,
        bucket_offsets,
        n_buckets,
        sample_buffer,
    )
    return RasterOutput(
        render=render,
        aux=aux,
        error_map=error_map,
        radii=radii,
        means2d=means2d,
        depths=depths,
        n_isects=int(flatten_ids.shape[0]),
        n_buckets=int(n_buckets),
    )
