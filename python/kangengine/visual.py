"""Viewer-side wiring helpers for KangSimWorld.

KangSimWorld intentionally stays headless and owns canonical runtime state.
These helpers are owned by App or examples that need scene Prim/render visuals
synced from PhysX. They should not become a second simulation state source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re

import numpy as np

from ._core import _ke
from .rigid import expand_rigid_body_state, rigid_shape_specs


@dataclass(slots=True)
class ArticulationVisualView:
    """Viewer-side visual wrapper for one articulation.

    This does not own simulation state. It keeps the scene prims and hidden
    render handles needed to draw one robot/reference articulation, then exposes
    small visual controls such as color, visibility, collision visibility, and
    render-pick handle to body-id lookup.
    """

    env_id: int
    obj_id: int
    skeleton_bridge: object
    body_prims: list[object]
    collision_prims: list[object]
    body_handles: list[int]
    app: object | None = None

    @property
    def key(self):
        return (self.env_id, self.obj_id)

    @property
    def prims(self):
        return tuple(self.body_prims)

    @property
    def num_bodies(self) -> int:
        return len(self.body_prims)

    @property
    def collision_visuals(self):
        return tuple(self.collision_prims)

    def body_id_from_render_handle(self, handle) -> int | None:
        handle = int(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if int(body_handle) == handle:
                return body_id
        return None

    def set_color(self, color):
        rgba_vec = _normalize_color(color)
        if rgba_vec is not None:
            for prim in self.body_prims:
                prim.set_display_color_alpha(rgba_vec)

        rgba = _normalize_color_array(color)
        if rgba is None or self.app is None or not self.body_handles:
            return self
        colors = rgba.reshape(1, 4)
        for handle in self.body_handles:
            self.app.set_renderable_colors(handle, colors)
        return self

    def set_visible(self, visible: bool):
        for prim in self.body_prims:
            prim.set_visible(bool(visible))
        return self

    def set_collision_visible(self, visible: bool):
        for prim in self.collision_prims:
            prim.set_visible(bool(visible))
        return self


@dataclass(slots=True)
class RigidVisualView:
    """Viewer-side visual wrapper for one rigid object.

    This does not own simulation state. It keeps the scene prims used to draw
    one rigid or compound rigid object and exposes lightweight visual controls
    such as color and visibility.
    """

    env_id: int
    obj_id: int
    rigid: object
    rigid_bridge: object
    body_prims: list[object]
    body_handles: list[int]
    app: object | None = None

    @property
    def key(self):
        return (self.env_id, self.obj_id)

    @property
    def prims(self):
        return tuple(self.body_prims)

    @property
    def num_bodies(self) -> int:
        return len(self.body_prims)

    def body_id_from_render_handle(self, handle) -> int | None:
        handle = int(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if int(body_handle) == handle:
                return body_id
        return None

    def set_color(self, color):
        rgba = _normalize_color(color)
        if rgba is not None:
            for prim in self.body_prims:
                prim.set_display_color_alpha(rgba)
        rgba_arr = _normalize_color_array(color)
        if rgba_arr is None or self.app is None or not self.body_handles:
            return self
        colors = rgba_arr.reshape(1, 4)
        for handle in self.body_handles:
            self.app.set_renderable_colors(handle, colors)
        return self

    def set_visible(self, visible: bool):
        for prim in self.body_prims:
            prim.set_visible(bool(visible))
        return self


class _VisualLifetime:
    def _mark_released(self):
        self._released = True

    @property
    def is_valid(self) -> bool:
        return not getattr(self, "_released", False)

    def _require_valid(self):
        if not self.is_valid:
            raise RuntimeError(f"{type(self).__name__} has been released")


class _SceneBatchBackend:
    """Per-env scene/PhysicsBridge visual records behind a visual batch."""

    def __init__(self, records):
        self.records = tuple(records)
        self._released = False

    @property
    def is_valid(self) -> bool:
        return not self._released

    def _require_valid(self):
        if self._released:
            raise RuntimeError(f"{type(self).__name__} has been released")

    @property
    def num_bodies(self) -> int:
        self._require_valid()
        return self.records[0].num_bodies if self.records else 0

    @property
    def prims(self):
        self._require_valid()
        return tuple(prim for record in self.records for prim in record.prims)

    @property
    def body_handles(self):
        self._require_valid()
        return tuple(
            int(handle)
            for record in self.records
            for handle in getattr(record, "body_handles", [])
        )

    def __len__(self) -> int:
        self._require_valid()
        return len(self.records)

    def __iter__(self):
        self._require_valid()
        return iter(self.records)

    def __getitem__(self, index):
        self._require_valid()
        return self.records[index]

    def get_env_view(self, env_id: int):
        self._require_valid()
        env_id = int(env_id)
        for record in self.records:
            if int(record.env_id) == env_id:
                return record
        raise KeyError(f"env_id {env_id} is outside this visual batch")

    def body_id_from_render_handle(self, handle) -> tuple[int, int] | None:
        self._require_valid()
        for record in self.records:
            if hasattr(record, "body_id_from_render_handle"):
                body_id = record.body_id_from_render_handle(handle)
                if body_id is not None:
                    return int(record.env_id), int(body_id)
        return None

    def set_visible(self, visible: bool):
        self._require_valid()
        for record in self.records:
            record.set_visible(visible)
        return self

    def set_color(self, color):
        self._require_valid()
        for index, record in enumerate(self.records):
            record.set_color(
                _select_env_visual_value(color, index, len(self.records), record.env_id)
            )
        return self

    def set_collision_visible(self, visible: bool):
        self._require_valid()
        for record in self.records:
            if hasattr(record, "set_collision_visible"):
                record.set_collision_visible(visible)
        return self

    def release(self):
        if self._released:
            return self
        self.records = ()
        self._released = True
        return self


class RigidSceneGraphBackend(_SceneBatchBackend):
    """Rigid visual records synced through the existing scene/bridge path."""


class ArticulationSceneGraphBackend(_SceneBatchBackend):
    """Articulation visual records synced through the existing scene/bridge path."""


class RigidCPUExternalBackend(_VisualLifetime):
    """CPU rigid root poses -> native SimVisualBatch -> ExternalBuffer."""

    def __init__(self, app, world, obj_id, env_ids, prim, handle, local_pos, local_rot):
        from . import physics

        self._released = False
        self.app = app
        self.world = world
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self.body_prims = (prim,)
        self.body_handles = (int(handle),)
        self.num_bodies = 1
        self.num_envs = len(self.env_ids)
        self._version = 0

        self._model = physics.SimModel()
        self._model.add_shape(
            0,
            int(handle),
            tuple(float(v) for v in np.asarray(local_pos, dtype=np.float32).reshape(3)),
            tuple(float(v) for v in np.asarray(local_rot, dtype=np.float32).reshape(4)),
            "rigid_shape",
        )
        self._model.add_object_boundary(0, 1, f"rigid_{self.obj_id}")
        self._state = physics.SimState()
        self._state.resize(self.num_envs, 1)
        self._batch = physics.SimVisualBatch()
        self._batch.set_model(self._model)

    @property
    def prims(self):
        return self.body_prims

    @property
    def body_handles(self):
        return self._body_handles

    @body_handles.setter
    def body_handles(self, value):
        self._body_handles = tuple(int(handle) for handle in value)

    def __len__(self) -> int:
        self._require_valid()
        return self.num_envs

    def sync(self):
        self._require_valid()
        for row, env_id in enumerate(self.env_ids):
            rigid = self.world.rigid(env_id, self.obj_id)
            self._state.set_body_transform(
                row,
                0,
                rigid.get_root_position(),
                rigid.get_root_rotation(),
            )
        self._batch.prepare_from_state(self._state)
        self._version += 1
        desc = self._batch.external_transform_desc(
            0, self._version, f"rigid_cpu_external_{self.obj_id}"
        )
        self.app.get_renderer().set_renderable_external_buffer(
            self.body_handles[0], desc
        )
        return self

    def set_visible(self, visible: bool):
        self._require_valid()
        self.body_prims[0].set_visible(bool(visible))
        return self

    def set_color(self, color):
        self._require_valid()
        self.app.set_renderable_colors(
            self.body_handles[0], _batch_colors(color, self.env_ids)
        )
        return self

    def body_id_from_render_handle(self, handle) -> int | None:
        self._require_valid()
        return 0 if int(handle) == self.body_handles[0] else None

    def release(self):
        if not self.is_valid:
            return self
        self._batch = None
        self._state = None
        self._model = None
        self.body_prims = ()
        self.body_handles = ()
        self.app = None
        self.world = None
        self._mark_released()
        return self


class ArticulationCPUExternalBackend(_VisualLifetime):
    """CPU articulation link poses -> native SimVisualBatch -> ExternalBuffer."""

    def __init__(self, app, world, obj_id, env_ids, body_prims, body_handles):
        from . import physics

        self._released = False
        self.app = app
        self.world = world
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self.body_prims = tuple(body_prims)
        self.body_handles = tuple(int(handle) for handle in body_handles)
        self.num_bodies = len(self.body_handles)
        self.num_envs = len(self.env_ids)
        self._version = 0

        self._model = physics.SimModel()
        self._model.set_body_renderables(list(self.body_handles))
        self._state = physics.SimState()
        self._state.resize(self.num_envs, self.num_bodies)
        self._batch = physics.SimVisualBatch()
        self._batch.set_model(self._model)

    @property
    def prims(self):
        return self.body_prims

    def __len__(self) -> int:
        self._require_valid()
        return self.num_envs

    def sync(self):
        self._require_valid()
        for row, env_id in enumerate(self.env_ids):
            articulation = self.world.articulation(env_id, self.obj_id)
            pos = np.asarray(
                articulation.get_link_positions(), dtype=np.float32
            ).reshape(self.num_bodies, 3)
            rot = np.asarray(
                articulation.get_link_rotations(), dtype=np.float32
            ).reshape(self.num_bodies, 4)
            for body_id in range(self.num_bodies):
                self._state.set_body_transform(
                    row,
                    body_id,
                    pos[body_id],
                    rot[body_id],
                )
        self._batch.prepare_from_state(self._state)
        self._version += 1
        renderer = self.app.get_renderer()
        for shape_id, handle in enumerate(self.body_handles):
            desc = self._batch.external_transform_desc(
                shape_id,
                self._version,
                f"articulation_cpu_external_{self.obj_id}_{shape_id}",
            )
            renderer.set_renderable_external_buffer(handle, desc)
        return self

    def set_visible(self, visible: bool):
        self._require_valid()
        for prim in self.body_prims:
            prim.set_visible(bool(visible))
        return self

    def set_color(self, color):
        self._require_valid()
        colors = _batch_colors(color, self.env_ids)
        for handle in self.body_handles:
            self.app.set_renderable_colors(handle, colors)
        return self

    def body_id_from_render_handle(self, handle) -> int | None:
        self._require_valid()
        handle = int(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if body_handle == handle:
                return body_id
        return None

    def release(self):
        if not self.is_valid:
            return self
        self._batch = None
        self._state = None
        self._model = None
        self.body_prims = ()
        self.body_handles = ()
        self.app = None
        self.world = None
        self._mark_released()
        return self


class SimVisualBatch:
    """Public visual batch handle returned when one sim view spans many envs.

    The handle owns no simulation state. It delegates visual operations to a
    backend such as scene/PhysicsBridge records or GPU ExternalBuffer batches.
    The low-level C++ transform batch remains exposed as
    ``kangengine.physics.SimVisualBatch``.
    """

    def __init__(self, obj_id, env_ids, records=(), *, backend=None):
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self._backend = (
            backend if backend is not None else _SceneBatchBackend(records)
        )
        self._released = False

    @property
    def key(self):
        self._require_valid()
        return (self.env_ids, self.obj_id)

    @property
    def num_envs(self) -> int:
        self._require_valid()
        return len(self.env_ids)

    @property
    def num_bodies(self) -> int:
        self._require_valid()
        return self._backend.num_bodies

    @property
    def prims(self):
        self._require_valid()
        return self._backend.prims

    @property
    def body_handles(self):
        self._require_valid()
        return self._backend.body_handles

    def __len__(self) -> int:
        self._require_valid()
        return len(self._backend)

    def __iter__(self):
        self._require_valid()
        return iter(self._backend)

    def __getitem__(self, index):
        self._require_valid()
        return self._backend[index]

    def sync(self):
        self._require_valid()
        if hasattr(self._backend, "sync"):
            self._backend.sync()
        return self

    def get_env_view(self, env_id: int):
        self._require_valid()
        if not hasattr(self._backend, "get_env_view"):
            raise TypeError(f"{type(self._backend).__name__} has no per-env views")
        return self._backend.get_env_view(env_id)

    def body_id_from_render_handle(self, handle):
        self._require_valid()
        return self._backend.body_id_from_render_handle(handle)

    def set_visible(self, visible: bool):
        self._require_valid()
        self._backend.set_visible(visible)
        return self

    def set_color(self, color):
        self._require_valid()
        self._backend.set_color(color)
        return self

    def set_collision_visible(self, visible: bool):
        self._require_valid()
        if hasattr(self._backend, "set_collision_visible"):
            self._backend.set_collision_visible(visible)
        return self

    @property
    def is_valid(self) -> bool:
        return not self._released

    def _require_valid(self):
        if self._released:
            raise RuntimeError("SimVisualBatch has been released")

    def release(self):
        if self._released:
            return self
        if hasattr(self._backend, "release"):
            self._backend.release()
        self._backend = None
        self._released = True
        return self


class ArticulationGPUExternalBackend(_VisualLifetime):
    """CUDA articulation link poses consumed as renderer instance buffers."""

    def __init__(self, app, world, obj_id, env_ids, body_prims, body_handles):
        import torch

        from .utils import to_gpu_array_view

        self._released = False
        self.app = app
        self.world = world
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self.body_prims = tuple(body_prims)
        self.body_handles = tuple(int(handle) for handle in body_handles)
        self.num_bodies = len(self.body_handles)
        self.num_envs = len(self.env_ids)
        self._rows = world.articulation_gpu_index_view(
            self.env_ids, self.obj_id
        )
        link_view = world.gpu_system.articulation_link_data()
        device = torch.device(f"cuda:{int(link_view.device_id)}")
        link_indices = world.articulations[
            (self.env_ids[0], self.obj_id)
        ].articulation.get_link_indices()
        for env_id in self.env_ids[1:]:
            other = world.articulations[
                (env_id, self.obj_id)
            ].articulation.get_link_indices()
            if other != link_indices:
                raise RuntimeError(
                    "GPU articulation instances have different link index maps"
                )
        self._link_indices_tensor = torch.tensor(
            link_indices, dtype=torch.int32, device=device
        )
        self._link_indices = to_gpu_array_view(
            self._link_indices_tensor,
            dtype=torch.int32,
            name=f"gpu_articulation_{self.obj_id}_link_indices",
        )
        self.device_id = int(link_view.device_id)
        self.stream_handle = int(
            torch.cuda.current_stream(device).cuda_stream
        )

    @property
    def prims(self):
        return self.body_prims

    def __len__(self) -> int:
        self._require_valid()
        return self.num_envs

    def sync(self):
        self._require_valid()
        gpu_system = self.world.gpu_system
        gpu_system.fetch_articulation_link_pose()
        link_view = gpu_system.articulation_link_data()
        renderer = self.app.get_renderer()
        mapped = renderer.map_renderable_cuda_transform_buffers(
            self.body_handles,
            self.num_envs,
            self.device_id,
            self.stream_handle,
        )
        try:
            _ke.articulation_link_state_to_mapped_mat4_cuda(
                link_view,
                self._rows,
                self._link_indices,
                mapped,
            )
        finally:
            renderer.unmap_renderable_cuda_transform_buffers(
                self.body_handles,
                self.device_id,
                self.stream_handle,
            )
        return self

    def set_visible(self, visible: bool):
        self._require_valid()
        for prim in self.body_prims:
            prim.set_visible(bool(visible))
        return self

    def set_color(self, color):
        self._require_valid()
        colors = _batch_colors(color, self.env_ids)
        for handle in self.body_handles:
            self.app.set_renderable_colors(handle, colors)
        return self

    def body_id_from_render_handle(self, handle) -> int | None:
        self._require_valid()
        handle = int(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if body_handle == handle:
                return body_id
        return None

    def release(self):
        if not self.is_valid:
            return self
        self._rows = None
        self._link_indices = None
        self._link_indices_tensor = None
        self.body_prims = ()
        self.body_handles = ()
        self.app = None
        self.world = None
        self._mark_released()
        return self


class RigidGPUExternalBackend(_VisualLifetime):
    """CUDA rigid root poses consumed as one renderer instance buffer."""

    def __init__(self, app, world, obj_id, env_ids, prim, handle):
        import torch

        self._released = False
        self.app = app
        self.world = world
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self.body_prims = (prim,)
        self.body_handles = (int(handle),)
        self.num_bodies = 1
        self.num_envs = len(self.env_ids)
        self._rows = world.rigid_gpu_index_view(self.env_ids, self.obj_id)
        rigid_view = world.gpu_system.rigid_data()
        device = torch.device(f"cuda:{int(rigid_view.device_id)}")
        self.device_id = int(rigid_view.device_id)
        self.stream_handle = int(
            torch.cuda.current_stream(device).cuda_stream
        )

    @property
    def prims(self):
        return self.body_prims

    def __len__(self) -> int:
        self._require_valid()
        return self.num_envs

    def sync(self):
        self._require_valid()
        gpu_system = self.world.gpu_system
        gpu_system.fetch_rigid_data()
        renderer = self.app.get_renderer()
        mapped = renderer.map_renderable_cuda_transform_buffers(
            self.body_handles,
            self.num_envs,
            self.device_id,
            self.stream_handle,
        )
        try:
            _ke.indexed_rigid_state_to_mat4_cuda(
                gpu_system.rigid_data(), self._rows, mapped[0]
            )
        finally:
            renderer.unmap_renderable_cuda_transform_buffers(
                self.body_handles,
                self.device_id,
                self.stream_handle,
            )
        return self

    def set_visible(self, visible: bool):
        self._require_valid()
        self.body_prims[0].set_visible(bool(visible))
        return self

    def set_color(self, color):
        self._require_valid()
        self.app.set_renderable_colors(
            self.body_handles[0], _batch_colors(color, self.env_ids)
        )
        return self

    def body_id_from_render_handle(self, handle) -> int | None:
        self._require_valid()
        return 0 if int(handle) == self.body_handles[0] else None

    def release(self):
        if not self.is_valid:
            return self
        self._rows = None
        self.body_prims = ()
        self.body_handles = ()
        self.app = None
        self.world = None
        self._mark_released()
        return self


GpuArticulationVisualBatch = ArticulationGPUExternalBackend
GpuRigidVisualBatch = RigidGPUExternalBackend


class RigidVisualBridge:
    """Viewer-side visualizer for one compound rigid actor."""

    def __init__(
        self,
        app,
        scene,
        rigid,
        data,
        prim_base_path: str,
        shader=None,
        add_shapes: bool = True,
        color=None,
    ):
        self.app = app
        self.scene = scene
        self.rigid = rigid
        self.specs = rigid_shape_specs(data)
        self.local_pos = np.stack([spec.local_pos for spec in self.specs], axis=0)
        self.local_rot = np.stack([spec.local_rot for spec in self.specs], axis=0)
        self.body_prims = []
        self.body_handles = []
        for idx, spec in enumerate(self.specs):
            prim = self._define_shape_prim(prim_base_path, idx, spec)
            _apply_prim_color([prim], color)
            if add_shapes and shader is not None:
                self.body_handles.append(app.add_renderable(shader, prim))
            self.body_prims.append(prim)

    def sync(self):
        root_pos = np.asarray(self.rigid.get_root_position(), dtype=np.float32)
        root_rot = np.asarray(self.rigid.get_root_rotation(), dtype=np.float32)
        body_pos, body_rot = expand_rigid_body_state(
            root_pos, root_rot, self.local_pos, self.local_rot
        )
        for prim, pos, rot in zip(self.body_prims, body_pos, body_rot):
            prim.set_world_translation(
                _ke.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            )
            prim.set_world_rotation(
                _ke.quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2]))
            )

    def _define_shape_prim(self, base_path, idx, spec):
        path = f"{base_path}/{_safe_prim_name(spec.name)}_{idx}"
        prim = self.scene.define_prim(path, _ke.scene.PrimType.Mesh)
        geom_type = spec.geom_type
        size = spec.size
        if geom_type == "Sphere":
            mesh = _ke.scene.Prim.create_sphere_data(float(size[0]), 24, 12)
        elif geom_type == "Box":
            mesh = _ke.scene.Prim.create_rectangle_data(
                float(size[0] * 2.0), float(size[1] * 2.0), float(size[2] * 2.0)
            )
        elif geom_type == "Cylinder":
            mesh = _ke.scene.Prim.create_cylinder_data(
                float(size[0]), float(size[1] * 2.0), _ke.UpAxis.X, 24
            )
        else:
            mesh = _ke.scene.Prim.create_capsule_data(
                float(size[0]), float(size[1] * 2.0), _ke.UpAxis.X, 24
            )
        prim.set_mesh_data(mesh)
        return prim


class KangWorldVisualBridge:
    """Viewer-side wiring for one KangSimWorld.

    This class syncs render/scene visuals from simulation objects. It is not the
    canonical state owner; use ``world.state`` / ``world.refresh()`` for runtime
    state consumed by policy or training code.
    """

    def __init__(self, app, world):
        if not hasattr(_ke, "PhysicsBridge"):
            raise RuntimeError("KangWorldVisualBridge requires PhysicsBridge bindings")
        self.app = app
        self.world = world
        self.scene = app.get_scene()
        self.physics_bridge = _ke.PhysicsBridge(app)
        self.records: dict[tuple[int, int], ArticulationVisualView] = {}
        self.rigid_records: dict[tuple[int, int], RigidVisualView] = {}
        self.visual_batches: dict[int, SimVisualBatch] = {}
        self.cpu_external_visual_batches: dict[int, SimVisualBatch] = {}
        self.gpu_visual_batches: dict[int, SimVisualBatch] = {}
        self._skeleton_assets = {}
        self._instanced_colors: dict[tuple[int, ...], np.ndarray] = {}
        self._released = False

    @property
    def is_valid(self) -> bool:
        return not self._released

    def _require_valid(self):
        if self._released:
            raise RuntimeError("KangWorldVisualBridge has been released")

    def release(self):
        if self._released:
            return self
        batches = (
            list(self.visual_batches.values())
            + list(self.cpu_external_visual_batches.values())
            + list(self.gpu_visual_batches.values())
        )
        seen = set()
        for batch in batches:
            marker = id(batch)
            if marker in seen:
                continue
            seen.add(marker)
            if hasattr(batch, "release"):
                batch.release()
        self.gpu_visual_batches.clear()
        self.cpu_external_visual_batches.clear()
        self.visual_batches.clear()
        self.records.clear()
        self.rigid_records.clear()
        self._skeleton_assets.clear()
        self._instanced_colors.clear()
        self._released = True
        return self

    cleanup = release

    def add(
        self,
        sim_view,
        mjcf_path: str,
        prim_base_path: str | None = None,
        **kwargs,
    ):
        """Create viewer visuals from a simulation object/view.

        This is the preferred high-level path for user code: the simulation
        object owns env/object identity, while this bridge only creates and
        syncs viewer-side prims.
        """
        self._require_valid()
        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(eid) for eid in sim_view.env_ids)
        name = _safe_prim_name(getattr(sim_view, "name", "") or f"obj_{obj_id}")
        base_path = prim_base_path or f"/{name}"

        if all((eid, obj_id) in self.world.articulations for eid in env_ids):
            add_one = self.add_articulation
            backend_type = ArticulationSceneGraphBackend
        elif all((eid, obj_id) in self.world.rigids for eid in env_ids):
            add_one = self.add_rigid
            backend_type = RigidSceneGraphBackend
        else:
            raise KeyError(
                f"simulation view obj_id={obj_id} does not match registered objects"
            )

        records = []
        for index, env_id in enumerate(env_ids):
            path = base_path if len(env_ids) == 1 else f"{base_path}/env_{env_id}"
            env_kwargs = dict(kwargs)
            if "color" in env_kwargs:
                env_kwargs["color"] = _select_env_visual_value(
                    env_kwargs["color"], index, len(env_ids), env_id
                )
            env_kwargs["_debug_registration"] = False
            records.append(
                add_one(
                    env_id,
                    obj_id,
                    mjcf_path,
                    prim_base_path=path,
                    **env_kwargs,
                )
            )
        if len(records) == 1:
            return records[0]
        batch = SimVisualBatch(obj_id, env_ids, backend=backend_type(records))
        self.visual_batches[obj_id] = batch
        _debug_visual_batch(batch)
        return batch

    def add_articulation(
        self,
        env_id: int,
        obj_id: int,
        mjcf_path: str,
        prim_base_path: str = "/robot",
        scale: float = 1.0,
        order: str = "DFS",
        shader=None,
        add_shapes: bool = True,
        collision_base_path: str | None = None,
        collision_shader=None,
        show_collision: bool = False,
        color=None,
        _debug_registration: bool = True,
    ) -> ArticulationVisualView:
        self._require_valid()
        key = (int(env_id), int(obj_id))
        if key in self.records:
            raise ValueError(f"visual already registered for env={key[0]}, obj={key[1]}")

        articulation = self.world.articulation(key[0], key[1])
        asset, mesh_asset_base_path = self._skeleton_asset(mjcf_path, scale, order)
        skeleton_bridge = asset.instantiate(
            self.scene,
            prim_base_path,
            mesh_asset_base_path,
        )

        body_prims = list(skeleton_bridge.body_prims())
        _apply_prim_color(body_prims, color)
        body_handles = []
        if add_shapes and shader is not None:
            for prim in body_prims:
                body_handles.append(
                    self.app.add_renderable(
                        shader, prim, _ke.TransformSource.ExternalBuffer
                    )
                )
            self.physics_bridge.add_instanced(articulation, body_handles)
            self._append_instanced_color(body_handles, color)
            if _debug_registration:
                _debug_instancing(
                    kind="sim",
                    env_id=key[0],
                    obj_id=key[1],
                    num_bodies=len(body_prims),
                    handles=body_handles,
                    mesh_asset_base_path=mesh_asset_base_path,
                )
        else:
            self.physics_bridge.add(articulation, skeleton_bridge)
            if _debug_registration:
                _debug_instancing(
                    kind="sim-scenegraph",
                    env_id=key[0],
                    obj_id=key[1],
                    num_bodies=len(body_prims),
                    handles=[],
                    mesh_asset_base_path=mesh_asset_base_path,
                )

        collision_prims = []
        if collision_base_path is not None:
            collision_prims = list(
                self.physics_bridge.add_collision_visuals(
                    articulation,
                    self.scene,
                    collision_base_path,
                    bool(show_collision),
                )
            )
            shape_shader = collision_shader if collision_shader is not None else shader
            if add_shapes and shape_shader is not None:
                for prim in collision_prims:
                    self.app.add_renderable(shape_shader, prim)

        record = ArticulationVisualView(
            key[0],
            key[1],
            skeleton_bridge,
            body_prims,
            collision_prims,
            body_handles,
            self.app,
        )
        self.records[key] = record
        return record

    def add_gpu_articulation(
        self,
        sim_view,
        mjcf_path: str,
        prim_base_path: str = "/gpu_robot",
        scale: float = 1.0,
        order: str = "DFS",
        shader=None,
        color=None,
    ) -> SimVisualBatch:
        """Create one renderable per link backed by CUDA instance transforms."""
        self._require_valid()
        if shader is None:
            raise ValueError("add_gpu_articulation requires a shader")
        if not hasattr(_ke, "articulation_link_state_to_mat4_cuda"):
            raise RuntimeError("KangEngine was built without CUDA transform kernels")

        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(env_id) for env_id in sim_view.env_ids)
        if obj_id in self.gpu_visual_batches:
            raise ValueError(f"GPU visual batch already registered for obj={obj_id}")
        if any((env_id, obj_id) not in self.world.articulations for env_id in env_ids):
            raise KeyError(f"articulation obj={obj_id} is not registered in every env")

        asset, mesh_asset_base_path = self._skeleton_asset(mjcf_path, scale, order)
        skeleton_bridge = asset.instantiate(
            self.scene, prim_base_path, mesh_asset_base_path
        )
        body_prims = list(skeleton_bridge.body_prims())
        if len(body_prims) != sim_view.num_bodies:
            raise RuntimeError(
                "GPU articulation visual body count does not match PhysX links"
            )
        body_handles = [
            self.app.add_renderable(
                shader, prim, _ke.TransformSource.ExternalBuffer
            )
            for prim in body_prims
        ]
        backend = ArticulationGPUExternalBackend(
            self.app, self.world, obj_id, env_ids, body_prims, body_handles
        )
        batch = SimVisualBatch(obj_id, env_ids, backend=backend)
        batch.set_color(color)
        batch.sync()
        self.gpu_visual_batches[obj_id] = batch
        self.visual_batches[obj_id] = batch
        return batch

    def add_cpu_external_articulation(
        self,
        sim_view,
        mjcf_path: str,
        prim_base_path: str = "/cpu_external_robot",
        scale: float = 1.0,
        order: str = "DFS",
        shader=None,
        color=None,
    ) -> SimVisualBatch:
        """Create one renderable per link backed by CPU ExternalBuffer."""
        self._require_valid()
        if shader is None:
            raise ValueError("add_cpu_external_articulation requires a shader")

        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(env_id) for env_id in sim_view.env_ids)
        if obj_id in self.visual_batches:
            raise ValueError(f"visual batch already registered for obj={obj_id}")
        if obj_id in self.cpu_external_visual_batches:
            raise ValueError(
                f"CPU external visual batch already registered for obj={obj_id}"
            )
        if obj_id in self.gpu_visual_batches:
            raise ValueError(f"GPU visual batch already registered for obj={obj_id}")
        if any((env_id, obj_id) not in self.world.articulations for env_id in env_ids):
            raise KeyError(f"articulation obj={obj_id} is not registered in every env")

        asset, mesh_asset_base_path = self._skeleton_asset(mjcf_path, scale, order)
        skeleton_bridge = asset.instantiate(
            self.scene, prim_base_path, mesh_asset_base_path
        )
        body_prims = list(skeleton_bridge.body_prims())
        if len(body_prims) != sim_view.num_bodies:
            raise RuntimeError(
                "CPU external articulation body count does not match PhysX links"
            )
        body_handles = [
            self.app.add_renderable(
                shader, prim, _ke.TransformSource.ExternalBuffer
            )
            for prim in body_prims
        ]
        backend = ArticulationCPUExternalBackend(
            self.app, self.world, obj_id, env_ids, body_prims, body_handles
        )
        batch = SimVisualBatch(obj_id, env_ids, backend=backend)
        batch.set_color(color)
        batch.sync()
        self.cpu_external_visual_batches[obj_id] = batch
        self.visual_batches[obj_id] = batch
        return batch

    def add_gpu_rigid(
        self,
        sim_view,
        mjcf_path: str,
        prim_base_path: str = "/gpu_rigid",
        scale: float = 1.0,
        order: str = "DFS",
        shader=None,
        color=None,
    ) -> SimVisualBatch:
        """Create a single-shape rigid batch backed by CUDA transforms."""
        self._require_valid()
        if shader is None:
            raise ValueError("add_gpu_rigid requires a shader")
        if not hasattr(_ke, "indexed_rigid_state_to_mat4_cuda"):
            raise RuntimeError("KangEngine was built without CUDA transform kernels")

        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(env_id) for env_id in sim_view.env_ids)
        if obj_id in self.gpu_visual_batches:
            raise ValueError(f"GPU visual batch already registered for obj={obj_id}")
        data = self.world.load_mjcf(mjcf_path, scale=scale, order=order)
        rigid_bridge = RigidVisualBridge(
            self.app,
            self.scene,
            sim_view.first.rigid,
            data,
            prim_base_path,
            add_shapes=False,
            color=color,
        )
        if len(rigid_bridge.body_prims) != 1:
            raise NotImplementedError(
                "GPU rigid visual batches currently require one MJCF shape"
            )
        prim = rigid_bridge.body_prims[0]
        handle = self.app.add_renderable(
            shader, prim, _ke.TransformSource.ExternalBuffer
        )
        backend = RigidGPUExternalBackend(
            self.app, self.world, obj_id, env_ids, prim, handle
        )
        batch = SimVisualBatch(obj_id, env_ids, backend=backend)
        batch.set_color(color)
        batch.sync()
        self.gpu_visual_batches[obj_id] = batch
        self.visual_batches[obj_id] = batch
        return batch

    def add_cpu_external_rigid(
        self,
        sim_view,
        mjcf_path: str,
        prim_base_path: str = "/cpu_external_rigid",
        scale: float = 1.0,
        order: str = "DFS",
        shader=None,
        color=None,
    ) -> SimVisualBatch:
        """Create a single-shape rigid batch backed by CPU ExternalBuffer."""
        self._require_valid()
        if shader is None:
            raise ValueError("add_cpu_external_rigid requires a shader")

        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(env_id) for env_id in sim_view.env_ids)
        if obj_id in self.visual_batches:
            raise ValueError(f"visual batch already registered for obj={obj_id}")
        if obj_id in self.cpu_external_visual_batches:
            raise ValueError(
                f"CPU external visual batch already registered for obj={obj_id}"
            )
        if obj_id in self.gpu_visual_batches:
            raise ValueError(f"GPU visual batch already registered for obj={obj_id}")
        data = self.world.load_mjcf(mjcf_path, scale=scale, order=order)
        rigid_bridge = RigidVisualBridge(
            self.app,
            self.scene,
            sim_view.first.rigid,
            data,
            prim_base_path,
            add_shapes=False,
            color=color,
        )
        if len(rigid_bridge.body_prims) != 1:
            raise NotImplementedError(
                "CPU external rigid visual batches currently require one MJCF shape"
            )
        spec = rigid_bridge.specs[0]
        prim = rigid_bridge.body_prims[0]
        handle = self.app.add_renderable(
            shader, prim, _ke.TransformSource.ExternalBuffer
        )
        backend = RigidCPUExternalBackend(
            self.app,
            self.world,
            obj_id,
            env_ids,
            prim,
            handle,
            spec.local_pos,
            spec.local_rot,
        )
        batch = SimVisualBatch(obj_id, env_ids, backend=backend)
        batch.set_color(color)
        batch.sync()
        self.cpu_external_visual_batches[obj_id] = batch
        self.visual_batches[obj_id] = batch
        return batch

    def add_rigid(
        self,
        env_id: int,
        obj_id: int,
        mjcf_path: str,
        prim_base_path: str = "/rigid",
        scale: float = 1.0,
        order: str = "DFS",
        shader=None,
        add_shapes: bool = True,
        color=None,
        _debug_registration: bool = True,
    ) -> RigidVisualView:
        self._require_valid()
        key = (int(env_id), int(obj_id))
        if key in self.rigid_records or key in self.records:
            raise ValueError(f"visual already registered for env={key[0]}, obj={key[1]}")

        rigid = self.world.rigid(key[0], key[1])
        data = self.world.load_mjcf(mjcf_path, scale=scale, order=order)
        rigid_bridge = RigidVisualBridge(
            self.app,
            self.scene,
            rigid,
            data,
            prim_base_path,
            shader=shader,
            add_shapes=add_shapes,
            color=color,
        )

        record = RigidVisualView(
            key[0],
            key[1],
            rigid,
            rigid_bridge,
            list(rigid_bridge.body_prims),
            list(rigid_bridge.body_handles),
            self.app,
        )
        self.rigid_records[key] = record
        if _debug_registration:
            _debug_instancing(
                kind="rigid",
                env_id=key[0],
                obj_id=key[1],
                num_bodies=len(record.body_prims),
                handles=record.body_handles,
                mesh_asset_base_path="",
            )
        return record

    def add_visual_articulation(
        self,
        env_id: int,
        obj_id: int,
        mjcf_path: str,
        prim_base_path: str = "/robot",
        scale: float = 1.0,
        order: str = "DFS",
        shader=None,
        add_shapes: bool = True,
        color=None,
    ) -> ArticulationVisualView:
        """Create a rendered skeleton that is not attached to a PhysX articulation."""
        self._require_valid()
        key = (int(env_id), int(obj_id))
        if key in self.records:
            raise ValueError(f"visual already registered for env={key[0]}, obj={key[1]}")

        asset, mesh_asset_base_path = self._skeleton_asset(mjcf_path, scale, order)
        skeleton_bridge = asset.instantiate(
            self.scene,
            prim_base_path,
            mesh_asset_base_path,
        )
        body_prims = list(skeleton_bridge.body_prims())
        _apply_prim_color(body_prims, color)
        if add_shapes and shader is not None:
            for prim in body_prims:
                self.app.add_renderable(shader, prim)
        _debug_instancing(
            kind="visual-scenegraph",
            env_id=key[0],
            obj_id=key[1],
            num_bodies=len(body_prims),
            handles=[],
            mesh_asset_base_path=mesh_asset_base_path,
        )

        record = ArticulationVisualView(
            key[0],
            key[1],
            skeleton_bridge,
            body_prims,
            [],
            [],
            self.app,
        )
        self.records[key] = record
        return record

    def sync(self):
        self._require_valid()
        self.physics_bridge.sync()
        self._sync_rigids()
        for batch in self.cpu_external_visual_batches.values():
            batch.sync()
        for batch in self.gpu_visual_batches.values():
            batch.sync()

    def get_articulation_view(
        self,
        env_id: int,
        obj_id: int,
    ) -> ArticulationVisualView | None:
        self._require_valid()
        return self.records.get((int(env_id), int(obj_id)))

    def get_rigid_view(self, env_id: int, obj_id: int) -> RigidVisualView | None:
        self._require_valid()
        return self.rigid_records.get((int(env_id), int(obj_id)))

    def get_visual_view(
        self,
        env_id: int,
        obj_id: int,
    ) -> ArticulationVisualView | RigidVisualView | None:
        self._require_valid()
        key = (int(env_id), int(obj_id))
        return self.records.get(key) or self.rigid_records.get(key)

    def get_visual_batch(self, obj_id: int) -> SimVisualBatch | None:
        self._require_valid()
        return self.visual_batches.get(int(obj_id))

    def _sync_rigids(self):
        for record in self.rigid_records.values():
            record.rigid_bridge.sync()

    def set_body_transforms(self, env_id: int, obj_id: int, body_pos=None, body_rot=None):
        """Override rendered body prim transforms with world-space FK poses.

        This is used by MimicKit view_motion, where the environment computes the
        reference pose itself and expects the engine viewer to draw that pose.
        """
        record = self.get_articulation_view(env_id, obj_id)
        if record is None:
            return

        if body_pos is not None:
            for prim, pos in zip(record.body_prims, body_pos):
                prim.set_world_translation(
                    _ke.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                )

        if body_rot is not None:
            for prim, rot in zip(record.body_prims, body_rot):
                # MimicKit stores quaternions as xyzw; KangEngine's Python quat
                # constructor takes wxyz.
                prim.set_world_rotation(
                    _ke.quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])),
                )

    def set_root_transform(self, env_id: int, obj_id: int, root_pos=None, root_rot=None):
        """Apply a root-only fallback pose to a visual articulation.

        MimicKit visual/reference objects are not backed by PhysX, but some envs
        still drive them through root setters before or instead of body setters.
        SkeletonBridge keeps a zero-pose FK model that can at least move the
        whole rendered character with that root pose.
        """
        record = self.get_articulation_view(env_id, obj_id)
        if record is None:
            return

        if root_pos is not None:
            pos = root_pos
            record.skeleton_bridge.set_root_translation(
                _ke.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            )

        if root_rot is not None:
            rot = root_rot
            record.skeleton_bridge.set_joint_rotation(
                0,
                _ke.quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])),
            )

        if root_pos is not None or root_rot is not None:
            record.skeleton_bridge.apply_pose()

    def set_collision_visible(self, visible: bool):
        self.physics_bridge.set_collision_visible(bool(visible))

    def body_id_from_render_handle(self, env_id: int, obj_id: int, handle) -> int | None:
        record = self.get_articulation_view(env_id, obj_id)
        if record is None:
            return None
        return record.body_id_from_render_handle(handle)

    def set_articulation_color(self, env_id: int, obj_id: int, color):
        record = self.get_articulation_view(env_id, obj_id)
        if record is None:
            return
        record.set_color(color)

    def _append_instanced_color(self, body_handles, color):
        if not body_handles:
            return
        key = tuple(int(h) for h in body_handles)
        colors = self._instanced_colors.get(key)
        rgba = _normalize_color_array(color)
        if rgba is None:
            rgba = np.array([0.15, 0.15, 0.15, 1.0], dtype=np.float32)
        if colors is None:
            colors = rgba.reshape(1, 4)
        else:
            colors = np.concatenate([colors, rgba.reshape(1, 4)], axis=0)
        self._instanced_colors[key] = colors
        for handle in body_handles:
            self.app.set_renderable_colors(handle, colors)

    def _skeleton_asset(self, mjcf_path: str, scale: float, order: str):
        scale = float(scale)
        key = (str(mjcf_path), scale, str(order))
        record = self._skeleton_assets.get(key)
        if record is not None:
            return record

        asset = _ke.animation.SkeletonBridgeAsset.from_mjcf(mjcf_path, scale, order)
        mesh_asset_base_path = _mesh_asset_base_path(mjcf_path, scale, order)
        asset.define_mesh_assets(self.scene, mesh_asset_base_path)
        record = (asset, mesh_asset_base_path)
        self._skeleton_assets[key] = record
        return record


def _normalize_color(color):
    arr = _normalize_color_array(color)
    if arr is None:
        return None
    return _ke.vec4(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _normalize_color_array(color):
    if color is None:
        return None
    arr = np.asarray(color, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size == 3:
        arr = np.concatenate([arr, np.ones(1, dtype=np.float32)])
    if arr.size < 4:
        raise ValueError(f"color must have 1, 3, or 4 values; got {arr.size}")
    return np.clip(arr[:4], 0.0, 1.0).astype(np.float32, copy=False)


def _batch_colors(color, env_ids):
    env_ids = tuple(env_ids)
    if color is None:
        return np.tile(
            np.asarray([0.15, 0.15, 0.15, 1.0], dtype=np.float32),
            (len(env_ids), 1),
        )
    rows = [
        _normalize_color_array(
            _select_env_visual_value(color, index, len(env_ids), env_id)
        )
        for index, env_id in enumerate(env_ids)
    ]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _apply_prim_color(prims, color):
    rgba = _normalize_color(color)
    if rgba is None:
        return
    for prim in prims:
        prim.set_display_color_alpha(rgba)


def _select_env_visual_value(value, index: int, env_count: int, env_id: int):
    if callable(value):
        return value(env_id)
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == env_count:
        return arr[index]
    return value


def _safe_prim_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return safe or "shape"


def _debug_instancing(kind, env_id, obj_id, num_bodies, handles, mesh_asset_base_path):
    if os.environ.get("KANGENGINE_DEBUG_RENDER_INSTANCING", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    unique_handles = len(set(int(h) for h in handles))
    print(
        "[kangengine render instancing] "
        f"kind={kind} env={env_id} obj={obj_id} bodies={num_bodies} "
        f"handles={len(handles)} unique_handles={unique_handles} "
        f"mesh_asset={mesh_asset_base_path}"
    )


def _debug_visual_batch(batch: SimVisualBatch):
    if os.environ.get("KANGENGINE_DEBUG_RENDER_INSTANCING", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    handles = batch.body_handles
    print(
        "[kangengine render instancing] "
        f"kind=sim-batch obj={batch.obj_id} envs={batch.num_envs} "
        f"bodies_per_env={batch.num_bodies} handles={len(handles)} "
        f"unique_handles={len(set(handles))}"
    )


def _mesh_asset_base_path(mjcf_path: str, scale: float, order: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", str(mjcf_path).split("/")[-1]).strip("_")
    if not stem:
        stem = "character"
    digest_src = f"{mjcf_path}|{float(scale):.9g}|{order}".encode("utf-8")
    digest = hashlib.sha1(digest_src).hexdigest()[:10]
    return f"/mesh_assets/skeletons/{stem}_{digest}"


# Backward-compatible aliases for older experimental code.
_VisualArticulationRecord = ArticulationVisualView
_VisualRigidRecord = RigidVisualView
