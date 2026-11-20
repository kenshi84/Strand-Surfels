// SPDX-License-Identifier: MIT
//
// Minimal CUDA / OpenGL interoperability for the Polyscope viewer.
//
// Polyscope (>= 2.4) can update its GPU buffers directly from device memory
// when given a set of interop functions (`polyscope.set_device_interop_funcs`).
// Its default implementation needs the `cuda-python` and `cupy` packages; the
// functions below provide the same with the CUDA runtime that PyTorch already
// loads. Rendered images thus go from the rasterizer to the screen without a
// round trip through host memory.
//
// Handles (graphics resources, mapped arrays) are passed to Python as integers.
// All operations are issued on PyTorch's current CUDA stream, so copies are
// ordered after the rendering kernels that produced the data.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "ops.h"

// The two OpenGL registration entry points of the CUDA runtime, declared here
// (with GLuint = GLenum = unsigned int) so that no OpenGL headers are needed
// at build time. They are defined in libcudart (see cuda_gl_interop.h).
extern "C" {
cudaError_t CUDARTAPI cudaGraphicsGLRegisterBuffer(
    struct cudaGraphicsResource **resource, unsigned int buffer, unsigned int flags
);
cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(
    struct cudaGraphicsResource **resource, unsigned int image, unsigned int target, unsigned int flags
);
}

namespace strand_surfels {
namespace gl_interop {

namespace {

constexpr unsigned int GL_TEXTURE_2D_TARGET = 0x0DE1; // GL_TEXTURE_2D

void check(const cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA-OpenGL interop: ") + what + " failed: " +
                                 cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")");
    }
}

cudaGraphicsResource_t as_resource(const uint64_t handle) {
    return reinterpret_cast<cudaGraphicsResource_t>(static_cast<uintptr_t>(handle));
}

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

} // namespace

uint64_t register_gl_buffer(const uint64_t gl_buffer_id) {
    cudaGraphicsResource_t resource = nullptr;
    check(cudaGraphicsGLRegisterBuffer(&resource, static_cast<unsigned int>(gl_buffer_id),
                                       cudaGraphicsRegisterFlagsWriteDiscard),
          "registering an OpenGL buffer");
    return reinterpret_cast<uintptr_t>(resource);
}

uint64_t register_gl_image_2d(const uint64_t gl_texture_id) {
    cudaGraphicsResource_t resource = nullptr;
    check(cudaGraphicsGLRegisterImage(&resource, static_cast<unsigned int>(gl_texture_id), GL_TEXTURE_2D_TARGET,
                                      cudaGraphicsRegisterFlagsWriteDiscard),
          "registering an OpenGL texture");
    return reinterpret_cast<uintptr_t>(resource);
}

void unregister_resource(const uint64_t handle) {
    check(cudaGraphicsUnregisterResource(as_resource(handle)), "unregistering a resource");
}

uint64_t map_resource_and_get_array(const uint64_t handle) {
    cudaGraphicsResource_t resource = as_resource(handle);
    check(cudaGraphicsMapResources(1, &resource, stream()), "mapping a resource");
    cudaArray_t array = nullptr;
    check(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0), "getting a mapped array");
    return reinterpret_cast<uintptr_t>(array);
}

std::tuple<uint64_t, uint64_t> map_resource_and_get_pointer(const uint64_t handle) {
    cudaGraphicsResource_t resource = as_resource(handle);
    check(cudaGraphicsMapResources(1, &resource, stream()), "mapping a resource");
    void *ptr = nullptr;
    size_t size = 0;
    check(cudaGraphicsResourceGetMappedPointer(&ptr, &size, resource), "getting a mapped pointer");
    return {reinterpret_cast<uintptr_t>(ptr), static_cast<uint64_t>(size)};
}

void unmap_resource(const uint64_t handle) {
    cudaGraphicsResource_t resource = as_resource(handle);
    check(cudaGraphicsUnmapResources(1, &resource, stream()), "unmapping a resource");
}

void memcpy_device(const uint64_t dst, const uint64_t src, const uint64_t n_bytes) {
    check(cudaMemcpyAsync(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), n_bytes,
                          cudaMemcpyDeviceToDevice, stream()),
          "copying to a buffer");
}

void memcpy_2d_to_array(const uint64_t dst_array, const uint64_t src, const uint64_t width_bytes,
                        const uint64_t height) {
    check(cudaMemcpy2DToArrayAsync(reinterpret_cast<cudaArray_t>(static_cast<uintptr_t>(dst_array)), 0, 0,
                                   reinterpret_cast<const void *>(src), width_bytes, width_bytes, height,
                                   cudaMemcpyDeviceToDevice, stream()),
          "copying to a texture");
}

} // namespace gl_interop
} // namespace strand_surfels
