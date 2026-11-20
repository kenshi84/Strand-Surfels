"""Build script of the CUDA extension ``strand_surfels._C``.

Package metadata lives in ``pyproject.toml``. Install with

    pip install --no-build-isolation -e .

Set ``STRAND_SURFELS_NO_EXTENSION=1`` to skip the compilation; the extension
is then built just-in-time on first import (see ``strand_surfels/_backend.py``).
"""

import glob
import os

from setuptools import setup

ROOT = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join("strand_surfels", "csrc")
GLM = os.path.join(ROOT, "third_party", "glm")
FLAGS = ["-O3", "-std=c++17"]  # keep in sync with strand_surfels/_backend.py


def extensions():
    """CUDA extension modules and the build command (empty when skipped)."""
    if os.environ.get("STRAND_SURFELS_NO_EXTENSION"):
        return [], {}
    from torch.utils.cpp_extension import (  # pylint: disable=import-outside-toplevel
        BuildExtension,
        CUDAExtension,
    )

    sources = sorted(glob.glob(os.path.join(CSRC, "*.cpp")) + glob.glob(os.path.join(CSRC, "*.cu")))
    ext = CUDAExtension(
        name="strand_surfels._C",
        sources=sources,
        include_dirs=[os.path.join(ROOT, CSRC), GLM],
        extra_compile_args={"cxx": FLAGS, "nvcc": FLAGS},
    )
    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = extensions()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
