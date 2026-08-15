"""KangEngine backend for Newton's public ViewerBase contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import inspect
from pathlib import Path
import re

import numpy as np

from ..._core import _ke
from ...app import App
from ... import render as render_api
from ._dependency import (
    NewtonUnavailableError,
    load_newton,
    load_picking,
    load_viewer_base,
)
from .buffers import rgba_array, to_numpy, transform_array_to_warp_matrices
from .conventions import transform_array_to_glm_matrices
from .geometry import compute_vertex_normals, mesh_data_from_arrays

try:
    _newton, _warp = load_newton()
    _ViewerBase = load_viewer_base()
except NewtonUnavailableError:
    _newton = None
    _warp = None
    _ViewerBase = object


def _path(name: str, category: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "item"
    return f"/Newton/{category}/{token}"


@dataclass(slots=True)
class _InstanceBatch:
    view: object
    mesh_name: str
    count: int
    transform_buffer: object | None = None


@dataclass(slots=True)
class _DynamicMeshBatch:
    view: object
    indices: np.ndarray


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
        _, viewer.show_collision = imgui.checkbox(
            "Collision geometry", bool(viewer.show_collision)
        )
        _, viewer.show_ground = imgui.checkbox("Ground", bool(viewer.show_ground))
        _, viewer.show_static = imgui.checkbox(
            "Static geometry", bool(viewer.show_static)
        )
        _, viewer.show_joints = imgui.checkbox("Joint axes", bool(viewer.show_joints))
        _, viewer.show_com = imgui.checkbox("Centers of mass", bool(viewer.show_com))
        _, viewer.show_inertia_boxes = imgui.checkbox(
            "Inertia boxes", bool(viewer.show_inertia_boxes)
        )
        _, viewer.show_contacts = imgui.checkbox("Contacts", bool(viewer.show_contacts))
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
            imgui.text(f"Body: {selection['body_label']} [{selection['body_index']}]")
            imgui.text(
                f"Shape: {selection['shape_label']} [{selection['shape_index']}]"
            )
            imgui.text(
                f"World: {selection['world_index']}  Mass: {selection['mass']:.4g}"
            )
        imgui.end()

    def fixed_update(self, fixed_dt):
        del fixed_dt
        viewer = self.viewer
        if viewer is not None and self.is_simulation_paused():
            viewer._pending_single_steps += 1

    def on_ray_picked(self, result):
        self.viewer._select_from_ray_pick(result)

    def on_force_drag_begin(self, result, target):
        self.viewer._begin_force_drag(result, target)

    def on_force_drag_update(self, result, target):
        self.viewer._update_force_drag(result, target)

    def on_force_drag_end(self):
        self.viewer._end_force_drag()


class NewtonViewer(_ViewerBase):
    """Render Newton models through KangEngine without owning simulation state.

    CPU models use host transform uploads. CUDA model transforms use a
    zero-copy Warp-to-Torch view and KangEngine's ExternalBuffer path; features
    that still require host inspection remain disabled unless explicit CUDA
    readback is enabled. All Newton-owned render geometry uses KangEngine's
    standard PBR material by default, including collision shapes and point/line
    debug meshes.
    """

    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
        headless: bool = False,
        allow_cuda_readback: bool = False,
    ):
        if _newton is None:
            load_newton()
        super().__init__()
        self.allow_cuda_readback = bool(allow_cuda_readback)
        self._closed = False
        self._frame_version = 0
        self._pending_single_steps = 0
        self._meshes: dict[str, object] = {}
        self._instances: dict[str, _InstanceBatch] = {}
        self._lines: dict[str, object] = {}
        self._points: dict[str, object] = {}
        self._dynamic_meshes: dict[str, _DynamicMeshBatch] = {}
        self._last_state = None
        self._picking = None
        self._mujoco_contacts = None
        self._mujoco_contacts_solver_id = None
        self._selection_info = None
        self._particle_constraint_indices = np.empty((0, 2), dtype=np.int32)
        self._particle_world_indices = np.empty(0, dtype=np.int32)

        self.app = _NewtonApp(self)
        self.app.initialize(
            int(width),
            int(height),
            False,
            _ke.UpAxis.Z,
            headless=bool(headless),
        )
        self.app.set_simulation_hotkeys_enabled(True)
        # App's fixed clock only transports paused single-step requests here;
        # Newton still owns the actual simulation dt and substep count.
        self.app.set_fixed_update_hz(60.0)
        self.app.set_external_force_drag_enabled(False)
        self.app.set_interaction_mode(_ke.InteractionMode.FORCE)
        self.materials = self.app.create_standard_materials()
        self.newton_material = self.materials.pbr

    def set_model(self, model, max_worlds=None):
        for batch in self._instances.values():
            batch.view.remove()
        for view in self._lines.values():
            view.remove()
        for view in self._points.values():
            view.remove()
        for batch in self._dynamic_meshes.values():
            batch.view.remove()
        self._instances.clear()
        self._lines.clear()
        self._points.clear()
        self._dynamic_meshes.clear()
        self._meshes.clear()
        self._mujoco_contacts = None
        self._mujoco_contacts_solver_id = None
        self._selection_info = None
        base_set_model = super().set_model
        supports_max_worlds = (
            "max_worlds" in inspect.signature(base_set_model).parameters
        )
        if supports_max_worlds:
            result = base_set_model(model, max_worlds=max_worlds)
        else:
            result = base_set_model(model)
        if (
            not supports_max_worlds
            and max_worlds is not None
            and hasattr(self, "set_visible_worlds")
            and model is not None
        ):
            count = min(int(max_worlds), int(model.world_count))
            self.set_visible_worlds(range(count))
        model_device = getattr(model, "device", None)
        device_is_cuda = bool(getattr(model_device, "is_cuda", False))
        picking_enabled = model is not None and (
            not device_is_cuda or self.allow_cuda_readback
        )
        self._picking = (
            load_picking()(model, world_offsets=self.world_offsets)
            if picking_enabled
            else None
        )
        if self._picking is not None:
            self._picking.visible_worlds_mask = self._visible_worlds_mask
        self._cache_particle_constraints(model)
        self.app.set_external_force_drag_enabled(self._picking is not None)
        return result

    def set_world_offsets(self, offset):
        result = super().set_world_offsets(offset)
        if self._picking is not None:
            self._picking.world_offsets = self.world_offsets
            self._picking.visible_worlds_mask = self._visible_worlds_mask
        return result

    def is_running(self) -> bool:
        return not self._closed and not self.app.should_close()

    def is_paused(self) -> bool:
        return bool(self.app.is_simulation_paused())

    def should_step(self) -> bool:
        if not self.is_paused():
            self._pending_single_steps = 0
            return True
        if self._pending_single_steps == 0:
            return False
        self._pending_single_steps -= 1
        return True

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

    def begin_frame(self, time: float):
        super().begin_frame(time)

    def log_state(self, state):
        self._last_state = state
        result = super().log_state(state)
        self._log_particle_constraints(state)
        self._log_picking_line()
        return result

    def _cache_particle_constraints(self, model):
        segments = []
        if model is not None:
            spring_indices = getattr(model, "spring_indices", None)
            if spring_indices is not None and len(spring_indices) > 0:
                segments.append(
                    np.asarray(spring_indices.numpy(), dtype=np.int32).reshape(-1, 2)
                )
            edge_indices = getattr(model, "edge_indices", None)
            if edge_indices is not None and len(edge_indices) > 0:
                edges = np.asarray(edge_indices.numpy(), dtype=np.int32).reshape(-1, 4)
                segments.append(edges[:, 2:4])
        self._particle_constraint_indices = (
            np.concatenate(segments, axis=0)
            if segments
            else np.empty((0, 2), dtype=np.int32)
        )
        particle_world = (
            None if model is None else getattr(model, "particle_world", None)
        )
        self._particle_world_indices = (
            np.asarray(particle_world.numpy(), dtype=np.int32).reshape(-1)
            if particle_world is not None
            else np.empty(0, dtype=np.int32)
        )

    def _log_particle_constraints(self, state):
        name = "/model/particle_constraints"
        indices = self._particle_constraint_indices
        if not self.show_springs or len(indices) == 0:
            self.log_lines(name, None, None, None)
            return

        positions = np.asarray(
            to_numpy(
                state.particle_q,
                name="particle constraint positions",
                allow_cuda_readback=self.allow_cuda_readback,
            ),
            dtype=np.float32,
        ).reshape(-1, 3)
        if len(self._particle_world_indices) == len(positions):
            worlds = self._particle_world_indices[indices[:, 0]]
            if self._visible_worlds is not None:
                visible = np.fromiter(
                    (world < 0 or world in self._visible_worlds for world in worlds),
                    dtype=bool,
                    count=len(worlds),
                )
                indices = indices[visible]
                worlds = worlds[visible]
            offsets = getattr(self, "world_offsets", None)
            if offsets is not None and len(indices) > 0:
                offset_values = np.asarray(offsets.numpy(), dtype=np.float32)
                positions = positions.copy()
                particle_worlds = self._particle_world_indices
                valid = (particle_worlds >= 0) & (particle_worlds < len(offset_values))
                positions[valid] += offset_values[particle_worlds[valid]]

        if len(indices) == 0:
            self.log_lines(name, None, None, None)
            return
        self.log_lines(
            name,
            positions[indices[:, 0]],
            positions[indices[:, 1]],
            np.array([[1.0, 0.55, 0.1]], dtype=np.float32),
            width=0.01,
        )

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
            world_indices = np.asarray(
                to_numpy(
                    body_world,
                    name="camera focus world indices",
                    allow_cuda_readback=True,
                ),
                dtype=np.int32,
            ).reshape(-1)
            offsets = getattr(self, "world_offsets", None)
            if offsets is not None and len(world_indices) == len(positions):
                offset_values = np.asarray(
                    to_numpy(
                        offsets,
                        name="camera focus world offsets",
                        allow_cuda_readback=True,
                    ),
                    dtype=np.float32,
                ).reshape(-1, 3)
                valid_worlds = (world_indices >= 0) & (
                    world_indices < len(offset_values)
                )
                positions[valid_worlds] += offset_values[world_indices[valid_worlds]]

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
            self.app.render_frame_once()
            self._frame_version += 1

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

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        roughness=None,
        metallic=None,
    ):
        del texture, color, roughness, metallic
        key = str(name)
        dynamic_surface = key == "/model/triangles"
        point_values = to_numpy(
            points,
            name=f"{name}.points",
            # Prototype topology is immutable setup data. Deformable vertices
            # are frame state and must not silently read back from CUDA.
            allow_cuda_readback=(self.allow_cuda_readback if dynamic_surface else True),
        )
        index_values = to_numpy(
            indices,
            name=f"{name}.indices",
            allow_cuda_readback=True,
        )
        normal_values = (
            None
            if normals is None
            else to_numpy(
                normals,
                name=f"{name}.normals",
                allow_cuda_readback=True,
            )
        )
        uv_values = (
            None
            if uvs is None
            else to_numpy(
                uvs,
                name=f"{name}.uvs",
                allow_cuda_readback=True,
            )
        )
        if not dynamic_surface:
            self._meshes[key] = mesh_data_from_arrays(
                point_values, index_values, normal_values, uv_values
            )
            return
        dynamic = self._dynamic_meshes.get(key)
        if hidden:
            if dynamic is not None:
                dynamic.view.set_visible(False)
            return

        flat_indices = np.asarray(index_values, dtype=np.uint32).reshape(-1)
        topology_changed = dynamic is None or not np.array_equal(
            dynamic.indices, flat_indices
        )
        if topology_changed:
            if dynamic is not None:
                dynamic.view.remove()
            mesh_data = mesh_data_from_arrays(
                point_values, flat_indices, normal_values, uv_values
            )
            view = self.app.scene.add_mesh(
                _path(name, "DynamicMeshes"), mesh_data, self.newton_material
            )
            view.set_double_sided(not bool(backface_culling))
            dynamic = _DynamicMeshBatch(view=view, indices=flat_indices.copy())
            self._dynamic_meshes[key] = dynamic
        else:
            normals = (
                compute_vertex_normals(point_values, flat_indices)
                if normal_values is None
                else np.asarray(normal_values, dtype=np.float32).reshape(-1, 3)
            )
            dynamic.view.update_geometry(point_values, normals)
        dynamic.view.set_visible(True)

    def log_instances(
        self,
        name,
        mesh,
        xforms,
        scales,
        colors,
        materials,
        hidden=False,
    ):
        del materials
        key = str(name)
        if hidden:
            batch = self._instances.get(key)
            if batch is not None:
                batch.view.set_visible(False)
            return
        if xforms is None:
            batch = self._instances.get(key)
            if batch is not None:
                batch.view.set_visible(True)
            return
        try:
            mesh_data = self._meshes[str(mesh)]
        except KeyError as exc:
            raise KeyError(f"Newton mesh {mesh!r} was not registered") from exc

        device = getattr(xforms, "device", None)
        if bool(getattr(device, "is_cuda", False)):
            self._log_cuda_instances(key, name, mesh, mesh_data, xforms, scales, colors)
            return

        transform_values = to_numpy(
            xforms,
            name=f"{name}.xforms",
            allow_cuda_readback=self.allow_cuda_readback,
        )
        scale_values = (
            None
            if scales is None
            else to_numpy(
                scales,
                name=f"{name}.scales",
                allow_cuda_readback=self.allow_cuda_readback,
            )
        )
        batch = self._instances.get(key)
        try:
            matrices = transform_array_to_glm_matrices(
                transform_values,
                scale_values,
                out=None if batch is None else batch.transform_buffer,
            )
        except ValueError as exc:
            raise ValueError(f"Newton instance batch {name!r}: {exc}") from exc
        if (
            batch is None
            or batch.mesh_name != str(mesh)
            or batch.count != len(matrices)
        ):
            if batch is not None:
                batch.view.remove()
            view = self.app.scene.add_mesh(
                _path(name, "Instances"),
                mesh_data,
                self.newton_material,
                transform_source=render_api.TransformSource.EXTERNAL_BUFFER,
            )
            color_values = (
                None
                if colors is None
                else rgba_array(
                    colors,
                    len(matrices),
                    allow_cuda_readback=self.allow_cuda_readback,
                )
            )
            batch = _InstanceBatch(view=view, mesh_name=str(mesh), count=len(matrices))
            self._instances[key] = batch
            batch.view._render_system.update_instances(
                batch.view.component, matrices, color_values
            )
        batch.transform_buffer = matrices
        batch.view.set_transform_buffer(
            matrices,
            sim_device="cpu",
            sync_policy=render_api.ExternalSyncPolicy.VERSIONED,
            version=self._frame_version,
        )
        batch.view.set_visible(True)

    def _log_cuda_instances(self, key, name, mesh, mesh_data, xforms, scales, colors):
        """Update one Newton instance batch without CUDA-to-host state copies."""

        from ...utils.sim_buffer import SimBuffer

        count = len(xforms)
        batch = self._instances.get(key)
        if batch is None or batch.mesh_name != str(mesh) or batch.count != count:
            if batch is not None:
                batch.view.remove()
            view = self.app.scene.add_mesh(
                _path(name, "Instances"),
                mesh_data,
                self.newton_material,
                transform_source=render_api.TransformSource.EXTERNAL_BUFFER,
            )
            # Colors are model metadata. Read them once while registering the
            # batch; dynamic state transforms stay on the CUDA device.
            color_values = rgba_array(colors, count, allow_cuda_readback=True)
            identities = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
            view._render_system.update_instances(
                view.component, identities, color_values
            )
            batch = _InstanceBatch(view=view, mesh_name=str(mesh), count=count)
            self._instances[key] = batch

        matrices, torch_stream = transform_array_to_warp_matrices(
            xforms, scales, out=batch.transform_buffer
        )
        # Keep the reused output tensor alive on the batch. The SimBuffer owner
        # is also retained by GpuArrayView while the renderer holds its
        # descriptor. Conversion and CUDA-to-OpenGL copy use this same Warp
        # stream, so no cross-stream event is needed for this path.
        batch.transform_buffer = matrices
        stream_handle = 0 if torch_stream is None else int(torch_stream.cuda_stream)
        buffer = SimBuffer(
            matrices,
            sim_device=str(matrices.device),
            memory_type="cuda_device",
            device_id=(
                matrices.device.index if matrices.device.index is not None else 0
            ),
            version=self._frame_version,
            stream_handle=stream_handle,
            owner=matrices,
        )
        batch.view.set_transform_buffer(
            buffer, sync_policy=render_api.ExternalSyncPolicy.VERSIONED
        )
        batch.view.set_visible(True)

    def log_lines(
        self,
        name,
        starts,
        ends,
        colors,
        width=0.01,
        hidden=False,
    ):
        key = str(name)
        if hidden:
            view = self._lines.get(key)
            if view is not None:
                view.set_visible(False)
            return
        if starts is None or ends is None:
            view = self._lines.get(key)
            if view is not None:
                view.set_visible(False)
            return
        start_values = to_numpy(
            starts,
            name=f"{name}.starts",
            allow_cuda_readback=self.allow_cuda_readback,
        ).reshape(-1, 3)
        end_values = to_numpy(
            ends,
            name=f"{name}.ends",
            allow_cuda_readback=self.allow_cuda_readback,
        ).reshape(-1, 3)
        if len(start_values) == 0:
            view = self._lines.get(key)
            if view is not None:
                view.set_visible(False)
            return
        color_values = rgba_array(
            colors,
            len(start_values),
            allow_cuda_readback=self.allow_cuda_readback,
        )
        view = self._lines.get(key)
        if view is None:
            view = self.app.scene.log_lines(
                _path(name, "Lines"),
                self.newton_material,
                start_values,
                end_values,
                color_values,
                radius=max(0.001, float(width) * 0.5),
            )
            self._lines[key] = view
        else:
            view.update_lines(start_values, end_values, color_values)
        view.set_visible(True)

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        key = str(name)
        if hidden:
            view = self._points.get(key)
            if view is not None:
                view.set_visible(False)
            return
        if points is None:
            view = self._points.get(key)
            if view is not None:
                view.set_visible(False)
            return
        point_values = to_numpy(
            points,
            name=f"{name}.points",
            allow_cuda_readback=self.allow_cuda_readback,
        ).reshape(-1, 3)
        radius_values = (
            0.01
            if radii is None
            else to_numpy(
                radii,
                name=f"{name}.radii",
                allow_cuda_readback=self.allow_cuda_readback,
            )
        )
        color_values = (
            None
            if colors is None
            else rgba_array(
                colors,
                len(point_values),
                allow_cuda_readback=self.allow_cuda_readback,
            )
        )
        view = self._points.get(key)
        if view is None:
            view = self.app.scene.debug_geometry.add_spheres(
                _path(name, "Points"),
                point_values,
                radius_values,
                color_values,
                material=self.newton_material,
            )
            self._points[key] = view
        else:
            view.update_spheres(point_values, radius_values, color_values)
        view.set_visible(True)

    def log_array(self, name, array):
        """Accept the ViewerBase signal API; array plots are not rendered yet."""

        del name, array

    def log_scalar(self, name, value, *, clear=False, smoothing=1):
        """Accept the ViewerBase signal API; scalar plots are not rendered yet."""

        del name, value, clear, smoothing

    def log_mujoco_contacts(self, solver, state):
        """Render contact normals from a Newton ``SolverMuJoCo`` instance."""

        name = "/contacts"
        if not self.show_contacts:
            self.log_arrows(name, None, None, None)
            return

        if bool(getattr(solver, "use_mujoco_cpu", False)):
            data = solver.mj_data
            count = int(data.ncon)
            if count == 0:
                self.log_arrows(name, None, None, None)
                return
            starts = np.asarray(data.contact.pos[:count], dtype=np.float32).copy()
            normals = np.asarray(
                data.contact.frame[:count, :3], dtype=np.float32
            ).copy()

            model = getattr(self, "model", None)
            if model is not None and self.world_offsets is not None:
                geom_to_shape = np.asarray(
                    solver.mjc_geom_to_newton_shape.numpy(), dtype=np.int32
                ).reshape(-1)
                shape_world = np.asarray(
                    model.shape_world.numpy(), dtype=np.int32
                ).reshape(-1)
                geom_ids = np.asarray(data.contact.geom[:count, 0], dtype=np.int32)
                valid_geom = (geom_ids >= 0) & (geom_ids < len(geom_to_shape))
                shape_ids = np.full(count, -1, dtype=np.int32)
                shape_ids[valid_geom] = geom_to_shape[geom_ids[valid_geom]]
                valid_shape = (shape_ids >= 0) & (shape_ids < len(shape_world))
                world_ids = np.full(count, -1, dtype=np.int32)
                world_ids[valid_shape] = shape_world[shape_ids[valid_shape]]
                offsets = np.asarray(self.world_offsets.numpy(), dtype=np.float32)
                valid_world = (world_ids >= 0) & (world_ids < len(offsets))
                starts[valid_world] += offsets[world_ids[valid_world]]

            ends = starts + normals * float(self.scene_scale * self._arrow_scale())
            self.log_arrows(name, starts, ends, (0.0, 1.0, 0.0))
            return

        solver_id = id(solver)
        if self._mujoco_contacts_solver_id != solver_id:
            self._mujoco_contacts = _newton.Contacts(
                solver.get_max_contact_count(),
                0,
                device=self.model.device,
                requested_attributes={"force"},
            )
            self._mujoco_contacts_solver_id = solver_id
        solver.update_contacts(self._mujoco_contacts, state)
        self.log_contacts(self._mujoco_contacts, state)

    def apply_forces(self, state):
        if self._picking is not None:
            self._picking._apply_picking_force(state)

    @staticmethod
    def _vec3_array(value) -> np.ndarray:
        if all(hasattr(value, axis) for axis in ("x", "y", "z")):
            return np.asarray(
                [float(value.x), float(value.y), float(value.z)],
                dtype=np.float32,
            )
        return np.asarray([float(value[i]) for i in range(3)], dtype=np.float32)

    def _camera_ray_to(self, point):
        camera_pos = self._vec3_array(self.app.get_camera().get_camera_pos())
        target = self._vec3_array(point)
        direction = target - camera_pos
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-8:
            return None
        direction /= length
        return (
            _warp.vec3(*map(float, camera_pos)),
            _warp.vec3(*map(float, direction)),
        )

    def _begin_force_drag(self, result, target):
        del target
        if self._picking is None or self._last_state is None or not result.hit:
            return
        ray = self._camera_ray_to(result.position)
        if ray is not None:
            self._picking.pick(self._last_state, *ray)

    def _select_from_ray_pick(self, result):
        """Resolve a KangEngine render pick to Newton body and shape metadata."""

        if self._picking is None or self._last_state is None or not result.hit:
            self._clear_selection()
            return

        already_picking = self._picking.is_picking()
        if not already_picking:
            ray = self._camera_ray_to(result.position)
            if ray is None:
                self._clear_selection()
                return
            self._picking.pick(self._last_state, *ray)

        body_index = int(self._picking.pick_body.numpy()[0])
        shape_index = (
            -1
            if self._picking.min_index is None
            else int(self._picking.min_index.numpy()[0])
        )
        if body_index < 0:
            self._clear_selection()
        else:
            model = self.model
            body_labels = getattr(model, "body_label", ())
            shape_labels = getattr(model, "shape_label", ())
            body_label = (
                str(body_labels[body_index])
                if body_index < len(body_labels)
                else f"body_{body_index}"
            )
            shape_label = (
                str(shape_labels[shape_index])
                if 0 <= shape_index < len(shape_labels)
                else f"shape_{shape_index}"
            )
            body_world = getattr(model, "body_world", None)
            world_index = (
                int(body_world.numpy()[body_index]) if body_world is not None else 0
            )
            if world_index < 0 and int(model.world_count) == 1:
                world_index = 0
            mass = float(model.body_mass.numpy()[body_index])
            self._selection_info = {
                "body_index": body_index,
                "body_label": body_label,
                "shape_index": shape_index,
                "shape_label": shape_label,
                "world_index": world_index,
                "mass": mass,
            }

        if not already_picking:
            self._picking.release()

    def _clear_selection(self):
        self._selection_info = None

    def _update_force_drag(self, result, target):
        del result
        if self._picking is None or not self._picking.is_picking():
            return
        ray = self._camera_ray_to(target)
        if ray is not None:
            self._picking.update(*ray)

    def _end_force_drag(self):
        if self._picking is not None:
            self._picking.release()

    def _log_picking_line(self):
        name = "/model/picking_line"
        if self._picking is None or not self._picking.is_picking():
            self.log_lines(name, None, None, None)
            return

        body_index = int(self._picking.pick_body.numpy()[0])
        if body_index < 0:
            self.log_lines(name, None, None, None)
            return

        pick_state = self._picking.pick_state.numpy()[0]
        start = np.asarray(pick_state["picked_point_world"], dtype=np.float32)
        end = np.asarray(pick_state["picking_target_world"], dtype=np.float32)

        model = getattr(self, "model", None)
        if (
            model is not None
            and self.world_offsets is not None
            and model.body_world is not None
        ):
            body_world = int(model.body_world.numpy()[body_index])
            if 0 <= body_world < len(self.world_offsets):
                offset = np.asarray(
                    self.world_offsets.numpy()[body_world], dtype=np.float32
                )
                start += offset
                end += offset

        self.log_lines(
            name,
            start.reshape(1, 3),
            end.reshape(1, 3),
            np.array([[0.0, 1.0, 1.0]], dtype=np.float32),
            width=0.025,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._end_force_drag()
        app = self.app
        if app.is_video_recording():
            app.stop_video_recording()
        app.request_close()
        app.viewer = None
        self._instances.clear()
        self._lines.clear()
        self._points.clear()
        self._dynamic_meshes.clear()
        self._meshes.clear()
        self.app = None
