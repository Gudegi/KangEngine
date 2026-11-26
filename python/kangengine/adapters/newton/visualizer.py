"""Render Newton models in an existing KangEngine App."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
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
from .buffers import (
    instance_rgba,
    rgba_array,
    to_numpy,
    transform_array_to_warp_matrices,
)
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
    colors: np.ndarray | None = None
    cuda: bool = False


@dataclass(slots=True)
class _DynamicMeshBatch:
    view: object
    vertex_ids: np.ndarray
    colors: np.ndarray | None = None


class NewtonVisualizer(_ViewerBase):
    """Register Newton geometry and update poses in a caller-owned App.

    CPU transforms use host uploads; CUDA transforms use Warp-to-Torch views
    and ExternalBuffer rendering. Keep the model and state alive through the
    App draw. One visualizer uses the /Newton scene subtree.
    """

    def __init__(self, app: App, *, allow_cuda_readback: bool = False):
        if _newton is None:
            load_newton()
        super().__init__()
        self.allow_cuda_readback = bool(allow_cuda_readback)
        self._closed = False
        self._frame_version = 0
        self._meshes: dict[str, object] = {}
        self._mesh_colors: dict[str, np.ndarray] = {}
        self._instances: dict[str, _InstanceBatch] = {}
        self._lines: dict[str, object] = {}
        self._points: dict[str, object] = {}
        self._dynamic_meshes: dict[str, _DynamicMeshBatch] = {}
        self._last_state = None
        self._inspection_source = None
        self._inspection_stream = None
        self._legacy_contacts_visible = False
        self._legacy_joints_visible = False
        self._picking = None
        self._selection_info = None
        self._particle_constraint_ids = np.empty((0, 2), dtype=np.int32)
        self._particle_world_ids = np.empty(0, dtype=np.int32)

        self.app = app
        self._enable_picking = False
        self.materials = app.create_standard_materials()
        self.newton_material = self.materials.pbr

    def set_model(self, model, max_worlds=None):
        inspector = self.app.physics_inspector
        if (
            self._inspection_source is not None
            and inspector.source is self._inspection_source
        ):
            inspector.set_source(None)
        self._inspection_source = None
        self._legacy_contacts_visible = False
        self._last_state = None
        self._inspection_stream = (
            _warp.get_stream(model.device)
            if model is not None and model.device.is_cuda
            else None
        )
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
        self._mesh_colors.clear()
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
        picking_enabled = (
            self._enable_picking
            and model is not None
            and (not device_is_cuda or self.allow_cuda_readback)
        )
        self._picking = (
            load_picking()(model, world_offsets=self.world_offsets)
            if picking_enabled
            else None
        )
        if self._picking is not None:
            self._picking.visible_worlds_mask = self._visible_worlds_mask
        self._cache_particle_constraints(model)
        if model is not None:
            from .inspection import NewtonInspectionSource

            self._inspection_source = NewtonInspectionSource(self)
            if inspector.source is None:
                inspector.set_source(self._inspection_source)
        return result

    def set_world_offsets(self, offset):
        result = super().set_world_offsets(offset)
        if self._picking is not None:
            self._picking.world_offsets = self.world_offsets
            self._picking.visible_worlds_mask = self._visible_worlds_mask
        if self._inspection_source is not None:
            self._inspection_source.clear_pose_cache()
            self._inspection_source.clear_com_cache()
            self._inspection_source.clear_frame_cache()
            self._inspection_source.clear_contact_cache()
            self._frame_version += 1
            inspector = self.app.physics_inspector
            if inspector.source is self._inspection_source:
                inspector.refresh()
        return result

    def log_state(self, state):
        self._frame_version += 1
        self._last_state = state
        if self.model.device.is_cuda:
            self._inspection_stream = _warp.get_stream(self.model.device)
        result = super().log_state(state)
        self._log_particle_constraints(state)
        self._log_picking_line()
        return result

    def _cache_particle_constraints(self, model):
        segments = []
        if model is not None:
            spring_ids = getattr(model, "spring_indices", None)
            if spring_ids is not None and len(spring_ids) > 0:
                segments.append(
                    np.asarray(spring_ids.numpy(), dtype=np.int32).reshape(-1, 2)
                )
            edge_ids = getattr(model, "edge_indices", None)
            if edge_ids is not None and len(edge_ids) > 0:
                edges = np.asarray(edge_ids.numpy(), dtype=np.int32).reshape(-1, 4)
                segments.append(edges[:, 2:4])
        self._particle_constraint_ids = (
            np.concatenate(segments, axis=0)
            if segments
            else np.empty((0, 2), dtype=np.int32)
        )
        particle_world = (
            None if model is None else getattr(model, "particle_world", None)
        )
        self._particle_world_ids = (
            np.asarray(particle_world.numpy(), dtype=np.int32).reshape(-1)
            if particle_world is not None
            else np.empty(0, dtype=np.int32)
        )

    def _log_particle_constraints(self, state):
        name = "/model/particle_constraints"
        particle_ids = self._particle_constraint_ids
        if not self.show_springs or len(particle_ids) == 0:
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
        if len(self._particle_world_ids) == len(positions):
            worlds = self._particle_world_ids[particle_ids[:, 0]]
            if self._visible_worlds is not None:
                visible = np.fromiter(
                    (world < 0 or world in self._visible_worlds for world in worlds),
                    dtype=bool,
                    count=len(worlds),
                )
                particle_ids = particle_ids[visible]
                worlds = worlds[visible]
            offsets = getattr(self, "world_offsets", None)
            if offsets is not None and len(particle_ids) > 0:
                offset_values = np.asarray(offsets.numpy(), dtype=np.float32)
                positions = positions.copy()
                particle_worlds = self._particle_world_ids
                valid = (particle_worlds >= 0) & (particle_worlds < len(offset_values))
                positions[valid] += offset_values[particle_worlds[valid]]

        if len(particle_ids) == 0:
            self.log_lines(name, None, None, None)
            return
        self.log_lines(
            name,
            positions[particle_ids[:, 0]],
            positions[particle_ids[:, 1]],
            np.array([[1.0, 0.55, 0.1]], dtype=np.float32),
            width=0.01,
        )

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
        dynamic=False,
        opacity=None,
    ):
        del texture, roughness, metallic
        key = str(name)
        triangle_path = self._qualify("/model/triangles")
        dynamic_surface = key == triangle_path or key.startswith(triangle_path + "/")
        if dynamic and not dynamic_surface:
            raise NotImplementedError(
                "Dynamic mesh prototypes are not supported by NewtonVisualizer"
            )
        if dynamic_surface and hidden:
            if key in self._dynamic_meshes:
                self._dynamic_meshes[key].view.set_visible(False)
            return
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
            self._mesh_colors[key] = instance_rgba(color, opacity, 1)
            return
        dynamic = self._dynamic_meshes.get(key)

        vertex_ids = np.asarray(index_values, dtype=np.uint32).reshape(-1)
        topology_changed = dynamic is None or not np.array_equal(
            dynamic.vertex_ids, vertex_ids
        )
        if topology_changed:
            if dynamic is not None:
                dynamic.view.remove()
            mesh_data = mesh_data_from_arrays(
                point_values, vertex_ids, normal_values, uv_values
            )
            view = self.app.scene.add_mesh(
                _path(name, "DynamicMeshes"), mesh_data, self.newton_material
            )
            view.set_double_sided(not bool(backface_culling))
            dynamic = _DynamicMeshBatch(view=view, vertex_ids=vertex_ids.copy())
            self._dynamic_meshes[key] = dynamic
        else:
            normals = (
                compute_vertex_normals(point_values, vertex_ids)
                if normal_values is None
                else np.asarray(normal_values, dtype=np.float32).reshape(-1, 3)
            )
            dynamic.view.update_geometry(point_values, normals)
        dynamic.view.set_visible(True)
        values = instance_rgba(color, opacity, 1, previous=dynamic.colors)
        if values is not dynamic.colors:
            rgba = values[0]
            dynamic.view.set_base_color(_ke.Vec4(*map(float, rgba)))
            dynamic.view.set_alpha_mode(
                render_api.AlphaMode.BLEND
                if rgba[3] < 1.0
                else render_api.AlphaMode.OPAQUE
            )
            dynamic.colors = values

    def log_instances(
        self,
        name,
        mesh,
        xforms,
        scales,
        colors,
        materials,
        hidden=False,
        opacities=None,
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
                self._update_instance_appearance(batch, colors, opacities)
                batch.view.set_visible(True)
            return
        try:
            mesh_data = self._meshes[str(mesh)]
        except KeyError as exc:
            raise KeyError(f"Newton mesh {mesh!r} was not registered") from exc

        device = getattr(xforms, "device", None)
        if bool(getattr(device, "is_cuda", False)):
            self._log_cuda_instances(
                key, name, mesh, mesh_data, xforms, scales, colors, opacities
            )
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
            batch = _InstanceBatch(view=view, mesh_name=str(mesh), count=len(matrices))
            self._instances[key] = batch
            batch.view._render_system.update_instances(batch.view.component, matrices)
        self._update_instance_appearance(batch, colors, opacities)
        batch.transform_buffer = matrices
        batch.view.set_transform_buffer(
            matrices,
            sim_device="cpu",
            sync_policy=render_api.ExternalSyncPolicy.VERSIONED,
            version=self._frame_version,
        )
        batch.view.set_visible(True)

    def _update_instance_appearance(self, batch, colors, opacities):
        initial = batch.colors is None
        # CUDA appearance is setup metadata unless the caller opts into readback.
        if batch.cuda and not initial and not self.allow_cuda_readback:
            return
        if not initial and colors is None and opacities is None:
            return
        values = instance_rgba(
            colors,
            opacities,
            batch.count,
            previous=self._mesh_colors[batch.mesh_name] if initial else batch.colors,
            allow_cuda_readback=initial or self.allow_cuda_readback,
        )
        if values is batch.colors:
            return
        batch.view._render_system.set_instance_colors(batch.view.component, values)
        batch.view.set_alpha_mode(
            render_api.AlphaMode.BLEND
            if np.any(values[:, 3] < 1.0)
            else render_api.AlphaMode.OPAQUE
        )
        batch.colors = values

    def _log_cuda_instances(
        self, key, name, mesh, mesh_data, xforms, scales, colors, opacities
    ):
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
            identities = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
            view._render_system.update_instances(view.component, identities)
            batch = _InstanceBatch(
                view=view, mesh_name=str(mesh), count=count, cuda=True
            )
            self._instances[key] = batch

        self._update_instance_appearance(batch, colors, opacities)

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
            view = self.app.debug.geometry.add_spheres(
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

    def log_contacts(self, contacts, state):
        """Supply rigid contacts to the eye menu; no device queries while hidden."""
        self._inspection_source.set_contacts(contacts, state)
        self._sync_contact_visibility()

    def _sync_contact_visibility(self):
        # Honor explicit ViewerBase toggles without overriding eye-menu changes.
        if self.show_contacts != self._legacy_contacts_visible:
            self._legacy_contacts_visible = bool(self.show_contacts)
            inspector = self.app.physics_inspector
            if inspector.source is self._inspection_source:
                inspector.set_contacts_visible(self._legacy_contacts_visible)

    def _log_joints(self, state):
        # Honor explicit ViewerBase toggles; the shared inspector draws child frames.
        if self.show_joints != self._legacy_joints_visible:
            self._legacy_joints_visible = bool(self.show_joints)
            inspector = self.app.physics_inspector
            if inspector.source is self._inspection_source:
                inspector.set_frames_visible(
                    bodies=inspector.body_frames_visible,
                    joints=self._legacy_joints_visible,
                )

    def log_mujoco_contacts(self, solver, state):
        """Supply MuJoCo contacts lazily; call after log_state() each displayed frame."""
        self._inspection_source.set_contacts(None, state, solver=solver)
        self._sync_contact_visibility()

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

        body_idx = int(self._picking.pick_body.numpy()[0])
        shape_idx = (
            -1
            if self._picking.min_index is None
            else int(self._picking.min_index.numpy()[0])
        )
        if body_idx < 0:
            self._clear_selection()
        else:
            model = self.model
            body_labels = getattr(model, "body_label", ())
            shape_labels = getattr(model, "shape_label", ())
            body_label = (
                str(body_labels[body_idx])
                if body_idx < len(body_labels)
                else f"body_{body_idx}"
            )
            shape_label = (
                str(shape_labels[shape_idx])
                if 0 <= shape_idx < len(shape_labels)
                else f"shape_{shape_idx}"
            )
            body_world = getattr(model, "body_world", None)
            world_idx = (
                int(body_world.numpy()[body_idx]) if body_world is not None else 0
            )
            if world_idx < 0 and int(model.world_count) == 1:
                world_idx = 0
            mass = float(model.body_mass.numpy()[body_idx])
            self._selection_info = {
                "body_idx": body_idx,
                "body_label": body_label,
                "shape_idx": shape_idx,
                "shape_label": shape_label,
                "world_idx": world_idx,
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

        body_idx = int(self._picking.pick_body.numpy()[0])
        if body_idx < 0:
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
            body_world = int(model.body_world.numpy()[body_idx])
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

    def end_frame(self):
        """Leave event processing and drawing to the App owner."""
        self._frame_version += 1

    def release(self):
        """Remove Newton visuals while keeping the caller's App open."""
        if self._closed:
            return
        self._end_force_drag()
        self.set_model(None)
        self._last_state = None
        self._closed = True
        self.app = None

    def close(self):
        self.release()
