"""Loader of the compiled CUDA extension.

The extension is normally compiled when the package is installed
(``pip install --no-build-isolation -e .``, see ``setup.py``). If the compiled
module is not available, it is built just-in-time with
:func:`torch.utils.cpp_extension.load` and cached in the PyTorch extension
directory (``~/.cache/torch_extensions`` by default), so only the first import
is slow.
"""

from __future__ import annotations

import glob
import os
from types import ModuleType

import torch  # noqa: F401  # pylint: disable=unused-import  # loads libtorch for the compiled extension

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = os.path.join(_PACKAGE_DIR, "csrc")
GLM_DIR = os.path.join(os.path.dirname(_PACKAGE_DIR), "third_party", "glm")

# compiler flags shared by setup.py and the JIT build
CXX_FLAGS = ["-O3", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17"]


def source_files() -> list[str]:
    """All C++/CUDA sources of the extension."""
    return sorted(
        glob.glob(os.path.join(CSRC_DIR, "*.cpp")) + glob.glob(os.path.join(CSRC_DIR, "*.cu"))
    )


def _load() -> ModuleType:
    try:
        # pylint: disable-next=import-outside-toplevel
        from strand_surfels import _C  # type: ignore[attr-defined]

        return _C
    except ImportError:
        pass

    from torch.utils.cpp_extension import load  # pylint: disable=import-outside-toplevel

    return load(
        name="strand_surfels_cuda",
        sources=source_files(),
        extra_include_paths=[CSRC_DIR, GLM_DIR],
        extra_cflags=CXX_FLAGS,
        extra_cuda_cflags=NVCC_FLAGS,
        verbose=bool(os.environ.get("STRAND_SURFELS_VERBOSE_BUILD")),
    )


_C = _load()
