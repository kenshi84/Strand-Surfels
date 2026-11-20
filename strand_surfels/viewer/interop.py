"""CUDA-OpenGL interop functions for Polyscope.

Polyscope can upload images directly from GPU memory
(``ManagedBuffer.update_data_from_device``) once it is given a dictionary of
interop functions. Its default implementation requires the ``cuda-python`` and
``cupy`` packages; here the functions come from the compiled extension
(``strand_surfels/csrc/gl_interop.cpp``), which uses the CUDA runtime already
loaded by PyTorch. Source arrays are contiguous CUDA tensors.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _tensor_pointer(tensor: torch.Tensor):
    """``get_array_ptr`` of Polyscope: (pointer, shape, dtype, n_bytes); only the
    pointer is required."""
    if not (tensor.is_cuda and tensor.is_contiguous()):
        raise ValueError("device uploads need a contiguous CUDA tensor")
    return tensor.data_ptr(), None, None, None


def interop_functions() -> dict:
    """Polyscope device interop functions backed by the strand surfel extension."""
    from strand_surfels._backend import _C  # pylint: disable=import-outside-toplevel

    gl = _C.gl_interop
    return {
        "register_gl_buffer": gl.register_gl_buffer,
        "register_gl_image_2d": gl.register_gl_image_2d,
        "unregister_resource": gl.unregister_resource,
        "map_resource_and_get_array": gl.map_resource_and_get_array,
        "map_resource_and_get_pointer": gl.map_resource_and_get_pointer,
        "unmap_resource": gl.unmap_resource,
        "get_array_ptr": _tensor_pointer,
        "memcpy": gl.memcpy,
        "memcpy_2d": gl.memcpy_2d,
    }


def enable_cuda_gl_interop() -> bool:
    """Install the interop functions in Polyscope; returns whether it succeeded.

    Must be called before the first device upload. On failure the viewer falls
    back to uploads through host memory.
    """
    try:
        import polyscope as ps  # pylint: disable=import-outside-toplevel

        ps.set_device_interop_funcs(interop_functions())
        return True
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.warning("CUDA-OpenGL interop unavailable (%s); using host uploads", err)
        return False
