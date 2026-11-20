// SPDX-License-Identifier: MIT
//
// Python bindings of the strand surfel CUDA extension.

#include <torch/extension.h>

#include "config.h"
#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels of the strand surfel rasterizer and losses";
    m.def("project_surfels_fwd", &strand_surfels::project_surfels_fwd);
    m.def("project_surfels_bwd", &strand_surfels::project_surfels_bwd);
    m.def("intersect_tiles", &strand_surfels::intersect_tiles);
    m.def("rasterize_fwd", &strand_surfels::rasterize_fwd);
    m.def("rasterize_bwd", &strand_surfels::rasterize_bwd);
    m.def("sh_to_color_fwd", &strand_surfels::sh_to_color_fwd);
    m.def("sh_to_color_bwd", &strand_surfels::sh_to_color_bwd);
    m.def("geometry_loss_fwd", &strand_surfels::geometry_loss_fwd);
    m.def("geometry_loss_bwd", &strand_surfels::geometry_loss_bwd);
    m.def("orientation_2d", &strand_surfels::orientation_2d);
    m.def("sparse_adam_update", &strand_surfels::sparse_adam_update);

    // CUDA-OpenGL interop for the viewer (see strand_surfels/viewer/interop.py)
    auto gl = m.def_submodule("gl_interop", "CUDA-OpenGL interop functions for Polyscope");
    gl.def("register_gl_buffer", &strand_surfels::gl_interop::register_gl_buffer);
    gl.def("register_gl_image_2d", &strand_surfels::gl_interop::register_gl_image_2d);
    gl.def("unregister_resource", &strand_surfels::gl_interop::unregister_resource);
    gl.def("map_resource_and_get_array", &strand_surfels::gl_interop::map_resource_and_get_array);
    gl.def("map_resource_and_get_pointer", &strand_surfels::gl_interop::map_resource_and_get_pointer);
    gl.def("unmap_resource", &strand_surfels::gl_interop::unmap_resource);
    gl.def("memcpy", &strand_surfels::gl_interop::memcpy_device);
    gl.def("memcpy_2d", &strand_surfels::gl_interop::memcpy_2d_to_array);

    // layout constants shared with the Python side
    m.attr("TILE_SIZE") = strand_surfels::TILE_SIZE;
    m.attr("TILE_PIXELS") = strand_surfels::TILE_PIXELS;
    m.attr("BUCKET_SIZE") = strand_surfels::BUCKET_SIZE;
    m.attr("AUX_DIM") = strand_surfels::AUX_DIM;
    m.attr("SAMPLED_STATE_DIM") = strand_surfels::SAMPLED_STATE_DIM;
    m.attr("SAMPLED_AUX_DIM") = strand_surfels::SAMPLED_AUX_DIM;
}
