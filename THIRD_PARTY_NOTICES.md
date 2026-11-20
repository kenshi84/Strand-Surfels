# Third-party notices

Strand-Surfels is released under the MIT License (see `LICENSE`). Some files in
this repository are derived from third-party projects that are distributed
under other permissive licenses. Those files keep their original copyright
headers and state the changes made to them. This document lists every such
project together with its license.

## Code included in this repository

| Project | License | Used in |
|---|---|---|
| [GLM](https://github.com/g-truc/glm) 1.0.2 (vendored, unmodified) | MIT | `third_party/glm/` |
| [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3 | Apache-2.0 | `strand_surfels/csrc/`: surfel projection and tile intersection / sorting (modified, see file headers) |
| [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs) (modifications to the 3DGS rasterizer) | MIT | `strand_surfels/csrc/`: bucket sampling and per-bucket backward pass, sparse Adam (adapted, see file headers) |
| [Viridis colormap](https://github.com/BIDS/colormap) (matplotlib) | CC0 1.0 | `strand_surfels/viewer/display.py`: 17 color samples |

## Images

The images in `docs/images/` show a Strand-Surfels reconstruction of a hair
model of the [CT2Hair](https://github.com/facebookresearch/CT2Hair) dataset
(Shen et al., *CT2Hair: High-Fidelity 3D Hair Modeling using Computed
Tomography*, ACM Transactions on Graphics 2023). CT2Hair is licensed under
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0), and these images are shared under the same license. They are
not covered by the MIT License of the code. The dataset itself is not
included in this repository.

| Source | License | Used in |
|---|---|---|
| [CT2Hair](https://github.com/facebookresearch/CT2Hair) hair model, reconstructed with Strand-Surfels | CC BY-NC 4.0 | `docs/images/` |

## Runtime dependencies (not included, installed separately)

| Project | License | Purpose |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | tensors, autograd, CUDA extensions |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) | MIT | fused differentiable SSIM loss |
| [Polyscope](https://github.com/nmwsharp/polyscope) | MIT | interactive viewer |
| [NumPy](https://github.com/numpy/numpy) / [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause | numerics, k-nearest-neighbour queries |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND | image I/O |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | configuration files |
| [tqdm](https://github.com/tqdm/tqdm) | MPL-2.0 / MIT | progress bars |
| [jaxtyping](https://github.com/patrick-kidger/jaxtyping) | MIT | tensor shape annotations |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | Apache-2.0 | training logs |
