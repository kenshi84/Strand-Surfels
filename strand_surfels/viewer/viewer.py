"""Interactive Polyscope viewer of strand surfels.

The surfels are rendered with the CUDA rasterizer from Polyscope's view camera
and shown as a full-window *render image* that is depth composited with the
rest of the Polyscope scene (e.g. the training cameras). Rendered images are
copied into the OpenGL textures directly on the GPU (CUDA-OpenGL interop, see
:mod:`strand_surfels.viewer.interop`), with a fallback through host memory.

Usage, final scene::

    viewer = SurfelViewer(surfels, cameras)
    viewer.run()  # until the window is closed

Usage, during training::

    viewer = SurfelViewer(lambda: model.surfels(), cameras)
    for iteration in ...:
        ...  # optimisation step
        viewer.tick(iteration)  # redraws now and then; blocks while paused
    viewer.run()

If no display is available (or ``enabled=False``) the viewer does nothing, so
training scripts can call it unconditionally.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch

from strand_surfels.losses.labels import label_colors
from strand_surfels.rendering import Surfels
from strand_surfels.viewer.cameras import ViewerCamera, opencv_from_polyscope, up_direction
from strand_surfels.viewer.display import (
    LABEL_FILTERS,
    DisplayMode,
    DisplaySettings,
    display_image,
)
from strand_surfels.viewer.interop import enable_cuda_gl_interop

logger = logging.getLogger(__name__)

_IMAGE_NAME = "surfels"
_MODES = list(DisplayMode)


def display_available() -> bool:
    """Whether a window can be opened (a display server on Linux)."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class SurfelViewer:  # pylint: disable=too-many-instance-attributes
    """Polyscope viewer of surfels (see the module documentation).

    Args:
        surfels: the surfels, or a function returning the current surfels
            (called whenever a new image is rendered, e.g. during training).
        cameras: cameras shown in the scene (training views); the viewer
            starts from the first one.
        enabled: ``False`` for a no-op viewer (headless runs).
        interop: copy images on the GPU (CUDA-OpenGL interop) if possible.
        refresh_interval: minimum time in seconds between two redraws in
            :meth:`tick` (rendering costs training time).
        title: window title.
    """

    def __init__(
        self,
        surfels: Union[Surfels, Callable[[], Surfels]],
        cameras: Sequence[ViewerCamera] = (),
        *,
        enabled: bool = True,
        interop: bool = True,
        refresh_interval: float = 0.1,
        title: str = "Strand Surfels",
    ):
        self._get_surfels = surfels if callable(surfels) else (lambda: surfels)
        self.cameras = list(cameras)
        self.settings = DisplaySettings()
        self.scale_modifier = 1.0
        self.render_scale = 1.0
        self.sh_degree: Optional[int] = None  # None: the surfels' active degree
        self.refresh_interval = refresh_interval

        self._ps = None
        self._interop = False
        self._buffers: Dict[str, object] = {}
        self._image_size = (0, 0)
        self._last_key = None
        self._scene_version = 0
        self._last_refresh = 0.0
        self._camera_views = []
        self._show_cameras = True
        self._camera_index = 0

        # training state
        self._training = False
        self._paused = False
        self._step_requested = False
        self._iteration = 0
        self._stats: Dict[str, float] = {}
        self._render_ms = 0.0
        self._n_surfels = 0
        self._n_shown = 0
        self._has_labels = False

        self.enabled = enabled and torch.cuda.is_available() and display_available()
        if enabled and not self.enabled:
            logger.warning("No display or CUDA device available, the viewer is disabled")
        if self.enabled:
            try:
                self._init_polyscope(title, interop)
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.warning("Could not start the viewer (%s), it is disabled", err)
                self.enabled = False

    # ------------------------------------------------------------------ setup

    def _init_polyscope(self, title: str, interop: bool) -> None:
        import polyscope as ps  # pylint: disable=import-outside-toplevel

        self._ps = ps
        self._interop = interop and enable_cuda_gl_interop()
        ps.set_program_name(title)
        ps.set_enable_vsync(False)
        ps.set_max_fps(-1)
        ps.set_ground_plane_mode("none")
        ps.set_give_focus_on_show(True)
        ps.init()
        ps.set_open_imgui_window_for_user_callback(True)
        ps.set_user_callback(self._callback)

        surfels = self._get_surfels()
        self._set_scene_extents(surfels.means.detach())
        if self.cameras:
            ps.set_up_dir(up_direction(self.cameras))
            ps.set_front_dir("y_front" if ps.get_up_dir().endswith("z_up") else "z_front")
            self._register_cameras()
            self.go_to_camera(0)

    def _set_scene_extents(self, means: torch.Tensor) -> None:
        """Bounding box of the bulk of the surfels (robust to outliers)."""
        ps = self._ps
        if means.shape[0] == 0:
            return
        sample = means[
            torch.randint(means.shape[0], (min(means.shape[0], 1 << 16),), device=means.device)
        ]
        q = torch.quantile(sample.float(), torch.tensor([0.01, 0.99], device=means.device), dim=0)
        low, high = q[0].cpu().numpy(), q[1].cpu().numpy()
        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box(low, high)
        ps.set_length_scale(float(np.linalg.norm(high - low)))

    def _register_cameras(self) -> None:
        ps = self._ps
        length = 0.05 * ps.get_length_scale()
        for cam in self.cameras:
            view = ps.register_camera_view(
                f"camera {cam.name}",
                cam.polyscope_parameters(),
                widget_focal_length=length,
                widget_color=(0.95, 0.6, 0.2),
            )
            self._camera_views.append(view)

    # ------------------------------------------------------------- public API

    def tick(self, iteration: int, stats: Optional[Dict[str, float]] = None) -> None:
        """Call once per training iteration (after the optimiser step).

        Redraws at most every ``refresh_interval`` seconds; while the viewer is
        paused, keeps drawing (and blocks training) until it is resumed or a
        single step is requested. Closing the window disables the viewer.
        """
        if not self.enabled:
            return
        self._training = True
        self._iteration = iteration
        if stats:
            self._stats = dict(stats)
        self._scene_version += 1
        self._step_requested = False

        now = time.perf_counter()
        if not self._paused and now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now
        self._frame()
        while self.enabled and self._paused and not self._step_requested:
            self._frame()

    def go_to_camera(self, index: int, flight: bool = False) -> None:
        """Move the view to camera ``index`` (whose widget is hidden meanwhile)."""
        if not self.enabled or not self._camera_views:
            return
        for i, view in enumerate(self._camera_views):
            view.set_enabled(self._show_cameras and i != index)
        self._camera_index = index
        self._camera_views[index].set_view_to_this_camera(with_flight=flight)

    def notify_changed(self) -> None:
        """Force a new rendering at the next frame (the surfels changed)."""
        self._scene_version += 1

    def run(self) -> None:
        """Show the viewer until its window is closed."""
        self._training = False
        self._paused = False
        while self.enabled:
            self._frame()
        self.close()

    def close(self) -> None:
        """Close the window and release the GPU resources."""
        if self._ps is None:
            return
        self._release_image()
        self.enabled = False
        try:
            self._ps.shutdown()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        self._ps = None

    # -------------------------------------------------------------- internals

    def _frame(self) -> None:
        if self._ps is None:
            return
        if self._ps.window_requests_close():
            self.close()
            return
        self._ps.frame_tick()

    def _callback(self) -> None:
        """Polyscope user callback: UI, then rendering if anything changed."""
        self._build_ui()
        try:
            self._update_image()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Rendering in the viewer failed")
            self._paused = False
            self.enabled = False

    def _view(self):
        """OpenCV camera and resolution of the current Polyscope view."""
        ps = self._ps
        params = ps.get_view_camera_parameters()
        buf_w, buf_h = ps.get_buffer_size()
        width = max(1, int(buf_w * self.render_scale))
        height = max(1, int(buf_h * self.render_scale))
        K, viewmat = opencv_from_polyscope(
            params.get_view_mat(), params.get_fov_vertical_deg(), width, height
        )
        return K, viewmat, width, height

    def _update_image(self) -> None:
        K, viewmat, width, height = self._view()
        key = (
            viewmat.tobytes(), K.tobytes(), self._scene_version, repr(self.settings),
            self.scale_modifier, self.sh_degree,
        )  # fmt: skip
        if key == self._last_key:
            return
        self._last_key = key
        self._ps.set_background_color(self.settings.background)

        start = time.perf_counter()
        with torch.no_grad():
            surfels = self._get_surfels()
            self._n_surfels = len(surfels)
            surfels = self._filter_by_label(surfels)
            self._n_shown = len(surfels)
            device = surfels.means.device
            colors = None
            if self.settings.mode == DisplayMode.LABEL:
                labels = surfels.labels
                if labels is None:  # no labels: everything "other"
                    labels = torch.zeros(len(surfels), device=device)
                colors = label_colors(labels)
            out = surfels.render(
                torch.from_numpy(viewmat).to(device),
                torch.from_numpy(K).to(device),
                width,
                height,
                scale_modifier=self.scale_modifier,
                sh_degree=self.sh_degree,
                colors=colors,
            )
            rgba, depth = display_image(out, self.settings, K, viewmat)
        self._upload(rgba, depth)
        torch.cuda.synchronize()
        self._render_ms = 1000.0 * (time.perf_counter() - start)

    def _filter_by_label(self, surfels: Surfels) -> Surfels:
        """The surfels selected by the label filter of the settings."""
        self._has_labels = surfels.labels is not None
        s = self.settings
        if not self._has_labels or s.label_filter == LABEL_FILTERS[0]:
            return surfels
        hair = surfels.labels > s.label_threshold
        return surfels.select(hair if s.label_filter == LABEL_FILTERS[1] else ~hair)

    def _upload(self, rgba: torch.Tensor, depth: torch.Tensor) -> None:
        ps = self._ps
        height, width = depth.shape
        if self._interop:
            try:
                if self._image_size != (width, height):
                    self._release_image()
                    self._add_image(
                        np.zeros((height, width), np.float32),
                        np.zeros((height, width, 4), np.float32),
                    )
                    self._buffers = {
                        "colors": ps.get_quantity_buffer(_IMAGE_NAME, "colors"),
                        "depths": ps.get_quantity_buffer(_IMAGE_NAME, "depths"),
                    }
                    self._image_size = (width, height)
                self._buffers["colors"].update_data_from_device(rgba)
                self._buffers["depths"].update_data_from_device(depth)
                return
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.warning("GPU upload failed (%s); using host uploads", err)
                self._release_image()
                self._interop = False
        self._add_image(depth.cpu().numpy(), rgba.cpu().numpy())
        self._image_size = (width, height)

    def _add_image(self, depth: np.ndarray, rgba: np.ndarray) -> None:
        self._ps.add_raw_color_alpha_render_image_quantity(
            _IMAGE_NAME,
            depth,
            rgba,
            enabled=True,
            allow_fullscreen_compositing=True,
            is_premultiplied=True,
        )

    def _release_image(self) -> None:
        # dropping the buffer objects unregisters their CUDA resources, which
        # must happen while the OpenGL context is alive
        for buf in self._buffers.values():
            buf.release_mapped_buffer_CUDAOpenGL()
        self._buffers = {}
        self._image_size = (0, 0)

    # --------------------------------------------------------------------- UI

    def _build_ui(self) -> None:  # pylint: disable=too-many-branches,too-many-statements
        import polyscope.imgui as psim  # pylint: disable=import-outside-toplevel

        s = self.settings
        shown = "" if self._n_shown == self._n_surfels else f" ({self._n_shown:,} shown)"
        psim.TextUnformatted(
            f"{self._n_surfels:,} surfels{shown} | {self._image_size[0]}x{self._image_size[1]} | "
            f"{self._render_ms:.1f} ms | {'GPU' if self._interop else 'host'} upload"
        )

        if self._training:
            psim.Separator()
            line = f"iteration {self._iteration:,}"
            for name, value in self._stats.items():
                line += f" | {name} {value:.4g}"
            psim.TextUnformatted(line)
            _, self._paused = psim.Checkbox("Pause training", self._paused)
            if self._paused:
                psim.SameLine()
                if psim.Button("Step"):
                    self._step_requested = True
            _, self.refresh_interval = psim.SliderFloat(
                "Refresh interval [s]", self.refresh_interval, 0.0, 2.0
            )

        psim.Separator()
        changed, index = psim.Combo("Mode", _MODES.index(s.mode), [m.value for m in _MODES])
        if changed:
            s.mode = _MODES[index]
        if s.mode == DisplayMode.DEPTH:
            _, s.median_depth = psim.Checkbox("Median depth", s.median_depth)
            auto = s.depth_range is None
            changed, auto = psim.Checkbox("Automatic range", auto)
            if changed:
                s.depth_range = None if auto else s.last_depth_range
            if s.depth_range is None:
                lo, hi = s.last_depth_range
                psim.TextUnformatted(f"range {lo:.3f} .. {hi:.3f}")
            else:
                lo, hi = s.depth_range
                _, lo = psim.InputFloat("Near", lo)
                _, hi = psim.InputFloat("Far", hi)
                s.depth_range = (lo, max(hi, lo + 1e-6))
        elif s.mode == DisplayMode.DIRECTION_3D:
            _, s.direction_abs = psim.Checkbox("|d| (sign agnostic)", s.direction_abs)
        elif s.mode == DisplayMode.ORIENTATION_2D:
            _, s.orientation_hue = psim.Checkbox("Periodic hue colors", s.orientation_hue)
            _, s.median_depth = psim.Checkbox("Median depth", s.median_depth)
        elif s.mode == DisplayMode.LABEL and not self._has_labels:
            psim.TextUnformatted("These surfels have no hair labels")

        if self._has_labels:
            index = LABEL_FILTERS.index(s.label_filter)
            changed, index = psim.Combo("Surfels", index, list(LABEL_FILTERS))
            if changed:
                s.label_filter = LABEL_FILTERS[index]
            if s.label_filter != LABEL_FILTERS[0]:
                _, s.label_threshold = psim.SliderFloat(
                    "Hair threshold", s.label_threshold, 0.0, 1.0
                )

        _, s.composite = psim.Checkbox("Composite with scene", s.composite)
        _, bg = psim.ColorEdit3("Background", s.background)
        s.background = tuple(float(c) for c in bg)
        _, s.alpha_threshold = psim.SliderFloat("Alpha threshold", s.alpha_threshold, 0.0, 1.0)

        if psim.CollapsingHeader("Rendering"):
            _, self.scale_modifier = psim.SliderFloat(
                "Scale modifier", self.scale_modifier, 0.05, 1.0
            )
            _, self.render_scale = psim.SliderFloat("Resolution scale", self.render_scale, 0.1, 1.0)
            max_degree = 3
            degree = -1 if self.sh_degree is None else self.sh_degree
            changed, degree = psim.SliderInt("SH degree (-1: active)", degree, -1, max_degree)
            if changed:
                self.sh_degree = None if degree < 0 else degree

        if self.cameras and psim.CollapsingHeader("Cameras"):
            changed, self._show_cameras = psim.Checkbox("Show cameras", self._show_cameras)
            if changed:
                for view in self._camera_views:
                    view.set_enabled(self._show_cameras)
            _, self._camera_index = psim.SliderInt(
                "Camera", self._camera_index, 0, len(self.cameras) - 1
            )
            psim.SameLine()
            if psim.Button("Go"):
                self.go_to_camera(self._camera_index, flight=True)
            psim.TextUnformatted(self.cameras[self._camera_index].name)
