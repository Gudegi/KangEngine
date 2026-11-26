"""Standalone KangEngine window for Newton simulations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from ..._core import _ke
from ...app import App
from ._dependency import load_newton
from .buffers import to_numpy
from .visualizer import NewtonVisualizer


class _NewtonApp(App):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

    def render(self):
        viewer = self.viewer
        if viewer is None:
            return
        imgui = _ke.imgui
        imgui.begin("Newton Viewer")
        model = getattr(viewer, "model", None)
        if model is not None:
            imgui.text(
                f"Bodies: {model.body_count}  Shapes: {model.shape_count}  "
                f"Worlds: {model.world_count}"
            )
        imgui.text("Physics: Newton  |  Rendering: KangEngine PBR")
        if self.get_interaction_mode() == _ke.InteractionMode.EDIT:
            imgui.text("Edit: select a SceneGraph object to use its gizmo")
        else:
            imgui.text("Force drag: Shift + Left drag")
        imgui.separator()
        _, viewer.show_visual = imgui.checkbox(
            "Visual geometry", bool(viewer.show_visual)
        )
        _, viewer.show_ground = imgui.checkbox("Ground", bool(viewer.show_ground))
        _, viewer.show_static = imgui.checkbox(
            "Static geometry", bool(viewer.show_static)
        )
        _, viewer.show_inertia_boxes = imgui.checkbox(
            "Inertia boxes", bool(viewer.show_inertia_boxes)
        )
        _, viewer.show_particles = imgui.checkbox(
            "Particles", bool(viewer.show_particles)
        )
        _, viewer.show_triangles = imgui.checkbox(
            "Deformable surface", bool(viewer.show_triangles)
        )
        _, viewer.show_springs = imgui.checkbox(
            "Particle constraints", bool(viewer.show_springs)
        )
        if imgui.button("Screenshot"):
            viewer.save_screenshot()
        imgui.same_line()
        recording = viewer.is_recording()
        if imgui.button("Stop recording" if recording else "Start recording"):
            if recording:
                viewer.stop_recording()
            else:
                viewer.start_recording()
        imgui.separator()
        imgui.text("Selection")
        selection = viewer._selection_info
        if selection is None:
            imgui.text("No Newton body selected")
        else:
            imgui.text(f"Body: {selection['body_label']} [{selection['body_idx']}]")
            imgui.text(f"Shape: {selection['shape_label']} [{selection['shape_idx']}]")
            imgui.text(
                f"World: {selection['world_idx']}  Mass: {selection['mass']:.4g}"
            )
        imgui.end()

    def on_ray_picked(self, result):
        self.viewer._select_from_ray_pick(result)

    def on_force_drag_begin(self, result, target):
        self.viewer._begin_force_drag(result, target)

    def on_force_drag_update(self, result, target):
        self.viewer._update_force_drag(result, target)

    def on_force_drag_end(self):
        self.viewer._end_force_drag()


class NewtonViewer(NewtonVisualizer):
    """Own a KangEngine window and render caller-owned Newton simulation state."""

    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
        headless: bool = False,
        renderer: str | None = None,
        allow_cuda_readback: bool = False,
        # TODO: Remove the legacy render_backend keyword.
        render_backend: str | None = None,
    ):
        load_newton()
        app = _NewtonApp(None)
        app.initialize(
            int(width),
            int(height),
            False,
            _ke.UpAxis.Z,
            headless=bool(headless),
            renderer=renderer,
            render_backend=render_backend,
        )
        super().__init__(app, allow_cuda_readback=allow_cuda_readback)
        app.viewer = self
        self._enable_picking = True
        app.set_simulation_hotkeys_enabled(True)
        app.set_external_force_drag_enabled(False)
        app.set_interaction_mode(_ke.InteractionMode.FORCE)

    def set_model(self, model, max_worlds=None):
        result = super().set_model(model, max_worlds=max_worlds)
        self.app.set_external_force_drag_enabled(self._picking is not None)
        return result

    def is_running(self) -> bool:
        return not self._closed and not self.app.should_close()

    def is_paused(self) -> bool:
        return bool(self.app.is_simulation_paused())

    def should_step(self) -> bool:
        if not self.is_paused():
            return True
        return self.app.consume_simulation_step()

    def is_key_down(self, key: str | int) -> bool:
        if isinstance(key, int):
            return bool(self.app.is_key_down(key))
        aliases = {
            "space": _ke.keys.SPACE,
            "enter": _ke.keys.ENTER,
            "escape": _ke.keys.ESCAPE,
            "left": _ke.keys.LEFT,
            "right": _ke.keys.RIGHT,
        }
        native_key = aliases.get(str(key).strip().lower())
        return False if native_key is None else bool(self.app.is_key_down(native_key))

    def _model_focus_point(self) -> np.ndarray | None:
        model = getattr(self, "model", None)
        if model is None:
            return None

        source = getattr(model, "body_q", None)
        source_is_body = source is not None and len(source) > 0
        if source is None or len(source) == 0:
            source = getattr(model, "shape_transform", None)
        if source is None or len(source) == 0:
            return None

        # This is a one-time camera setup read, not the per-frame render path.
        transforms = np.asarray(
            to_numpy(source, name="camera focus transforms", allow_cuda_readback=True),
            dtype=np.float32,
        ).reshape(-1, 7)
        positions = transforms[:, :3].copy()

        body_world = getattr(model, "body_world", None)
        if source_is_body and body_world is not None:
            world_ids = np.asarray(
                to_numpy(
                    body_world,
                    name="camera focus world indices",
                    allow_cuda_readback=True,
                ),
                dtype=np.int32,
            ).reshape(-1)
            offsets = getattr(self, "world_offsets", None)
            if offsets is not None and len(world_ids) == len(positions):
                offset_values = np.asarray(
                    to_numpy(
                        offsets,
                        name="camera focus world offsets",
                        allow_cuda_readback=True,
                    ),
                    dtype=np.float32,
                ).reshape(-1, 3)
                valid_worlds = (world_ids >= 0) & (world_ids < len(offset_values))
                positions[valid_worlds] += offset_values[world_ids[valid_worlds]]

        positions = positions[np.all(np.isfinite(positions), axis=1)]
        if len(positions) == 0:
            return None
        return 0.5 * (positions.min(axis=0) + positions.max(axis=0))

    def set_camera(self, pos, pitch: float, yaw: float):
        position = np.asarray([float(pos[i]) for i in range(3)], dtype=np.float32)
        pitch_radians = np.deg2rad(np.clip(float(pitch), -89.0, 89.0))
        yaw_radians = np.deg2rad((float(yaw) + 180.0) % 360.0 - 180.0)
        front = np.array(
            [
                np.cos(yaw_radians) * np.cos(pitch_radians),
                np.sin(yaw_radians) * np.cos(pitch_radians),
                np.sin(pitch_radians),
            ],
            dtype=np.float32,
        )
        focus = self._model_focus_point()
        if focus is None:
            pivot_distance = max(float(getattr(self, "scene_scale", 1.0)) * 10.0, 1.0)
        else:
            pivot_distance = float(np.dot(focus - position, front))
            if pivot_distance <= 1.0e-3:
                pivot_distance = max(float(np.linalg.norm(focus - position)), 1.0)
        target = position + front * pivot_distance

        camera = self.app.get_camera()
        camera.set_camera_pos(_ke.Vec3(*map(float, position)))
        camera.set_target_pos(_ke.Vec3(*map(float, target)))

    def end_frame(self):
        if not self._closed:
            self.app.process_events()
            self.app.draw_frame()
            super().end_frame()

    def save_screenshot(self, output_path=None) -> Path:
        """Save the last presented KangEngine frame as a PNG."""

        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_path = Path("tmp/newton/screenshots") / f"newton_{stamp}.png"
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.app.write_pixels_png(str(path), True):
            raise RuntimeError(f"failed to save Newton screenshot: {path}")
        print(f"[INFO]: Saved Newton screenshot: {path}")
        return path

    def start_recording(self, output_path=None, fps: float | None = None):
        """Start recording KangEngine frames rendered by this viewer."""

        return self.app.start_video_recording(output_path, fps)

    def stop_recording(self):
        """Stop recording and return the finalized output path."""

        return self.app.stop_video_recording()

    def is_recording(self) -> bool:
        return bool(self.app.is_video_recording())

    def close(self):
        if self._closed:
            return
        app = self.app
        if app.is_video_recording():
            app.stop_video_recording()
        super().release()
        app.viewer = None
        app.close()
