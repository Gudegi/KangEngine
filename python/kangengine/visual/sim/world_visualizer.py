"""Viewer-side visualizers for KangSimWorld.

KangSimWorld intentionally stays headless and owns canonical runtime state.
These helpers are owned by App or examples that need scene Prim/render visuals
synced from PhysX. They may use native bridge objects internally, but the public
Python API treats them as visualizers, not simulation state owners.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..._core import _ke
from ... import render as render_api
from ...utils.math import quat_xyzw_to_wxyz
from ..articulation_visual import ArticulationVisualAsset
from ..rigid_visual import RigidVisual
from ..utils import (
    apply_visual_alpha,
    normalize_color_array as _normalize_color_array,
    visual_color,
    visual_material,
)
from .batch import VisualBatch
from .utils import (
    batch_colors as _batch_colors,
    debug_instancing as _debug_instancing,
    mesh_asset_base_path as _mesh_asset_base_path,
    safe_prim_name as _safe_prim_name,
)
from .records import (
    VisualArticulationSceneGraph,
    VisualBodyPick,
    VisualRigidSceneGraph,
)

if TYPE_CHECKING:
    from ...sim import SimArticulation, SimArticulationBatch, SimRigid, SimRigidBatch


class _VisualLifetime:
    def _mark_released(self):
        self._released = True

    @property
    def is_valid(self) -> bool:
        return not getattr(self, "_released", False)

    def _require_valid(self):
        if not self.is_valid:
            raise RuntimeError(f"{type(self).__name__} has been released")


def _physical_visual_mapping(visual, articulation):
    """Resolve imported visual node indices to compact physics body indices."""
    template = articulation.template
    prims = list(visual.body_prims())
    frames = template.source_frames
    if len(prims) != len(frames):
        raise RuntimeError("Articulation visual source topology does not match template")
    if list(visual.skeleton().node_names()) != [frame.name for frame in frames]:
        raise RuntimeError("Articulation visual node order does not match template")
    return (
        [prims[i] for i in template.body_source_indices],
        [frames[i].body_index for i in visual.render_prim_body_indices()],
    )


class _RigidExternalBackend(_VisualLifetime):
    """Shared rigid part controls and environment-to-renderable ownership."""

    def __init__(self, app, world, obj_id, env_ids, path, groups):
        self._released = False
        self.app = app
        self.world = world
        self.obj_id = int(obj_id)
        self.env_ids = tuple(env_ids)
        self.path = path
        self.num_envs = len(self.env_ids)
        self.num_bodies = 1
        self._groups = groups
        self._handle_env_ids = {
            handle: tuple(self.env_ids[row] for row in rows)
            for rows, visual in groups
            for handle in visual.body_handles
        }

    @property
    def prims(self):
        return tuple(prim for _, visual in self._groups for prim in visual.body_prims)

    @property
    def body_handles(self):
        return tuple(self._handle_env_ids)

    def __len__(self):
        self._require_valid()
        return self.num_envs

    def set_visible(self, visible):
        self._require_valid()
        for _, visual in self._groups:
            for view in visual.views:
                view.set_visible(visible)
        return self

    def set_color(self, color):
        self._require_valid()
        colors = _batch_colors(color, self.env_ids)
        render_system = self.app.get_scene_render_system()
        for rows, visual in self._groups:
            group_colors = colors[list(rows)]
            for part_id, view in enumerate(visual.views):
                render_system.set_instance_colors(
                    view.component, group_colors * visual.base_colors[part_id]
                )
        return self

    def body_id_from_render_handle(self, handle):
        self._require_valid()
        return 0 if int(handle) in self._handle_env_ids else None

    def env_ids_from_render_handle(self, handle):
        self._require_valid()
        return self._handle_env_ids[int(handle)]

    def release(self):
        if not self.is_valid:
            return self
        for _, visual in self._groups:
            visual.remove()
        self.app.scene.remove_prim(self.path)
        self._groups.clear()
        self._handle_env_ids.clear()
        self.app = None
        self.world = None
        self._mark_released()
        return self


class RigidCPUExternalBackend(_RigidExternalBackend):
    """CPU actor poses -> native SimVisualBatch -> per-part ExternalBuffer."""

    def __init__(self, app, world, obj_id, env_ids, path, groups):
        from ... import physics

        super().__init__(app, world, obj_id, env_ids, path, groups)
        self._version = 0
        self._batches = []
        self._group_actors = []
        for rows, visual in groups:
            model = physics.SimModel()
            for part_id, handle in enumerate(visual.body_handles):
                model.add_shape(
                    0,
                    handle,
                    visual.local_pos[part_id].tolist(),
                    visual.local_quat_xyzw[part_id].tolist(),
                    f"rigid_shape_{part_id}",
                    local_scale=visual.local_scale[part_id].tolist(),
                )
            model.add_object_boundary(0, 1, f"rigid_{self.obj_id}")
            state = physics.SimState()
            state.resize(len(rows), 1)
            batch = physics.SimVisualBatch()
            batch.set_model(model)
            self._batches.append((model, state, batch))
            self._group_actors.append(
                tuple(world.rigid(self.env_ids[row], self.obj_id) for row in rows)
            )

    def sync(self):
        self._require_valid()
        self._version += 1
        renderer = self.app.get_renderer()
        for group_id, ((_, visual), actors, (_, state, batch)) in enumerate(
            zip(self._groups, self._group_actors, self._batches)
        ):
            for row, actor in enumerate(actors):
                state.set_body_transform(
                    row, 0, actor.get_root_position(), actor.get_root_quat_xyzw()
                )
            batch.prepare_from_state(state)
            for part_id, handle in enumerate(visual.body_handles):
                desc = batch.external_transform_desc(
                    part_id,
                    self._version,
                    f"rigid_cpu_external_{self.obj_id}_{group_id}_{part_id}",
                )
                renderer.set_renderable_external_buffer(handle, desc)
        return self

    def release(self):
        if not self.is_valid:
            return self
        super().release()
        self._batches.clear()
        self._group_actors.clear()
        return self


class RigidGPUExternalBackend(_RigidExternalBackend):
    """CUDA actor poses -> local part transforms -> mapped OpenGL buffers."""

    def __init__(self, app, world, obj_id, env_ids, path, groups):
        import torch

        from ...utils import to_gpu_array_view
        from ...utils.batched_rotations import quat_xyzw_to_matrix

        super().__init__(app, world, obj_id, env_ids, path, groups)
        state = world.gpu_system.rigid_data()
        self.device_id = int(state.device_id)
        device = torch.device("cuda", self.device_id)
        self._row_tensors = []
        self._rows = []
        self._local_transforms = []

        # Source grouping and local transforms are immutable for this visual batch.
        for group_id, (rows, visual) in enumerate(groups):
            indices = torch.tensor(
                [world.rigid_gpu_row(self.env_ids[row], self.obj_id) for row in rows],
                dtype=torch.int32,
                device=device,
            )
            self._row_tensors.append(indices)
            self._rows.append(to_gpu_array_view(
                indices, dtype=torch.int32,
                name=f"rigid_visual_{self.obj_id}_{group_id}_rows",
            ))

            transforms = np.repeat(
                np.eye(4, dtype=np.float32)[None], len(visual.body_handles), axis=0
            )
            transforms[:, :3, :3] = (
                quat_xyzw_to_matrix(torch.from_numpy(visual.local_quat_xyzw)).numpy()
                * visual.local_scale[:, None]
            )
            transforms[:, :3, 3] = visual.local_pos
            self._local_transforms.append(
                transforms.transpose(0, 2, 1).reshape(-1, 16).tolist()
            )

        self._rows_ready = torch.cuda.Event()
        self._rows_ready.record(torch.cuda.current_stream(device))
        for rows in self._rows:
            rows.ready_event_handle = int(self._rows_ready.cuda_event)

    def sync(self):
        import torch

        self._require_valid()
        gpu_system = self.world.gpu_system
        gpu_system.fetch_rigid_data()
        state = gpu_system.rigid_data()
        stream = torch.cuda.current_stream(self.device_id)
        renderer = self.app.get_renderer()
        try:
            for (_, visual), indices, rows, local in zip(
                self._groups, self._row_tensors, self._rows, self._local_transforms
            ):
                indices.record_stream(stream)
                handles = visual.body_handles
                mapped = renderer.map_renderable_cuda_transform_buffers(
                    handles, len(indices), self.device_id, stream.cuda_stream
                )
                try:
                    _ke.rigid_state_to_mapped_mat4_cuda(state, rows, local, mapped)
                finally:
                    renderer.unmap_renderable_cuda_transform_buffers(
                        handles, self.device_id, stream.cuda_stream
                    )
        finally:
            # The next physics fetch must wait until these borrowed poses are consumed.
            producer = torch.cuda.ExternalStream(state.stream_handle, device=self.device_id)
            producer.wait_stream(stream)
        return self

    def release(self):
        if not self.is_valid:
            return self
        super().release()
        self._rows.clear()
        self._row_tensors.clear()
        self._local_transforms.clear()
        self._rows_ready = None
        return self


class ArticulationCPUExternalBackend(_VisualLifetime):
    """CPU articulation link poses -> native SimVisualBatch -> ExternalBuffer."""

    def __init__(
        self,
        app,
        world,
        obj_id,
        env_ids,
        body_prims,
        body_handles,
        collision_prims=(),
        *,
        render_body_ids=None,
        base_colors=None,
        owned_paths=(),
    ):
        from ... import physics

        self._released = False
        self.app = app
        self.world = world
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self.body_prims = tuple(body_prims)
        self.body_handles = tuple(int(handle) for handle in body_handles)
        self.collision_prims = tuple(collision_prims)
        self.num_bodies = len(self.body_prims)
        self._render_body_ids = tuple(
            range(self.num_bodies) if render_body_ids is None else render_body_ids
        )
        if len(self._render_body_ids) != len(self.body_handles):
            raise ValueError("render body IDs must match renderable handles")
        self.num_envs = len(self.env_ids)
        self._version = 0
        self._base_colors = (
            np.ones((len(self.body_handles), 4), dtype=np.float32)
            if base_colors is None else np.asarray(base_colors, dtype=np.float32)
        )
        self._owned_paths = owned_paths

        self._model = physics.SimModel()
        for body_id, handle in zip(self._render_body_ids, self.body_handles):
            self._model.add_shape(body_id, handle)
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
        if not self.body_handles:
            return self
        for row, env_id in enumerate(self.env_ids):
            articulation = self.world.articulation(env_id, self.obj_id)
            pos = np.asarray(
                articulation.get_link_positions(), dtype=np.float32
            ).reshape(self.num_bodies, 3)
            rot = np.asarray(
                articulation.get_link_quats_xyzw(), dtype=np.float32
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

    def set_collision_visible(self, visible: bool):
        self._require_valid()
        for prim in self.collision_prims:
            prim.set_visible(bool(visible))
        return self

    def set_color(self, color):
        self._require_valid()
        colors = _batch_colors(color, self.env_ids)
        for handle, base in zip(self.body_handles, self._base_colors):
            self.app.set_renderable_colors(handle, colors * base)
        return self

    def body_id_from_render_handle(self, handle) -> int | None:
        self._require_valid()
        handle = int(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if body_handle == handle:
                return self._render_body_ids[body_id]
        return None

    def release(self):
        if not self.is_valid:
            return self
        for path in self._owned_paths:
            if path is not None:
                self.app.scene.remove_prim(path)
        self._batch = None
        self._state = None
        self._model = None
        self.body_prims = ()
        self.body_handles = ()
        self._render_body_ids = ()
        self.collision_prims = ()
        self.app = None
        self.world = None
        self._mark_released()
        return self


class ArticulationGPUExternalBackend(_VisualLifetime):
    """CUDA articulation link poses consumed as renderer instance buffers."""

    def __init__(
        self,
        app,
        world,
        obj_id,
        env_ids,
        body_prims,
        body_handles,
        collision_prims=(),
        *,
        render_body_ids=None,
        base_colors=None,
        owned_paths=(),
    ):
        import torch

        from ...utils import to_gpu_array_view

        self._released = False
        self.app = app
        self.world = world
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self.body_prims = tuple(body_prims)
        self.body_handles = tuple(int(handle) for handle in body_handles)
        self.collision_prims = tuple(collision_prims)
        self.num_bodies = len(self.body_prims)
        self._render_body_ids = tuple(
            range(self.num_bodies) if render_body_ids is None else render_body_ids
        )
        if len(self._render_body_ids) != len(self.body_handles):
            raise ValueError("render body IDs must match renderable handles")
        self.num_envs = len(self.env_ids)
        self._base_colors = (
            np.ones((len(self.body_handles), 4), dtype=np.float32)
            if base_colors is None else np.asarray(base_colors, dtype=np.float32)
        )
        self._owned_paths = owned_paths
        self._rows = world.articulation_gpu_index_view(self.env_ids, self.obj_id)
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
            [link_indices[i] for i in self._render_body_ids],
            dtype=torch.int32, device=device,
        )
        self._link_indices = to_gpu_array_view(
            self._link_indices_tensor,
            dtype=torch.int32,
            name=f"gpu_articulation_{self.obj_id}_link_indices",
        )
        self.device_id = int(link_view.device_id)
        self.stream_handle = int(torch.cuda.current_stream(device).cuda_stream)

    @property
    def prims(self):
        return self.body_prims

    def __len__(self) -> int:
        self._require_valid()
        return self.num_envs

    def sync(self):
        self._require_valid()
        if not self.body_handles:
            return self
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

    def set_collision_visible(self, visible: bool):
        self._require_valid()
        for prim in self.collision_prims:
            prim.set_visible(bool(visible))
        return self

    def set_color(self, color):
        self._require_valid()
        colors = _batch_colors(color, self.env_ids)
        for handle, base in zip(self.body_handles, self._base_colors):
            self.app.set_renderable_colors(handle, colors * base)
        return self

    def body_id_from_render_handle(self, handle) -> int | None:
        self._require_valid()
        handle = int(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if body_handle == handle:
                return self._render_body_ids[body_id]
        return None

    def release(self):
        if not self.is_valid:
            return self
        for path in self._owned_paths:
            if path is not None:
                self.app.scene.remove_prim(path)
        self._rows = None
        self._link_indices = None
        self._link_indices_tensor = None
        self.body_prims = ()
        self.body_handles = ()
        self._render_body_ids = ()
        self.collision_prims = ()
        self.app = None
        self.world = None
        self._mark_released()
        return self


class SimWorldVisualizer:
    """Viewer-side visualizer for one KangSimWorld.

    This class syncs render/scene visuals from simulation objects. It is not the
    canonical state owner; use ``world.state`` / ``world.refresh()`` for runtime
    state consumed by policy or training code.
    """

    def __init__(self, app, world):
        if not hasattr(getattr(_ke, "physics", None), "PhysicsBridge"):
            raise RuntimeError("SimWorldVisualizer requires PhysicsBridge bindings")
        self.app = app
        self.world = world
        self.scene = app.get_native_scene()
        self.physics_bridge = _ke.physics.PhysicsBridge()
        self.visual_articulation_scene_graphs: dict[
            tuple[int, int], VisualArticulationSceneGraph
        ] = {}
        self.visual_rigid_scene_graphs: dict[
            tuple[int, int], VisualRigidSceneGraph
        ] = {}
        self.visual_batches: dict[int, VisualBatch] = {}
        self.cpu_visual_batches: dict[int, VisualBatch] = {}
        self.gpu_visual_batches: dict[int, VisualBatch] = {}
        self._articulation_visual_assets = {}
        self._frame_axes = {}
        self._released = False

    @property
    def is_valid(self) -> bool:
        return not self._released

    def _require_valid(self):
        if self._released:
            raise RuntimeError("SimWorldVisualizer has been released")

    def release(self):
        if self._released:
            return self
        for config in self._frame_axes.values():
            for path in config["paths"]:
                self.app.debug_overlay.clear(path)
        self._frame_axes.clear()
        batches = (
            list(self.visual_batches.values())
            + list(self.cpu_visual_batches.values())
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
        self.cpu_visual_batches.clear()
        self.visual_batches.clear()
        for record in self.visual_articulation_scene_graphs.values():
            for path in record.owned_paths:
                if path is not None:
                    self.app.scene.remove_prim(path)
        self.visual_articulation_scene_graphs.clear()
        for record in self.visual_rigid_scene_graphs.values():
            record.rigid_visual.remove()
        self.visual_rigid_scene_graphs.clear()
        self._articulation_visual_assets.clear()
        self._released = True
        return self

    cleanup = release

    def add(
        self,
        sim_handle: SimRigid | SimArticulation | SimRigidBatch | SimArticulationBatch,
        mjcf_path: str | None = None,
        *,
        path: str | None = None,
        **kwargs,
    ) -> VisualBatch:
        """Create one CPU or GPU ExternalBuffer VisualBatch.

        ``sim_handle`` may be a single ``SimRigid`` / ``SimArticulation`` or a
        ``SimRigidBatch`` / ``SimArticulationBatch``. Single objects are treated
        as a one-env visual batch. Use ``add_scene_graph`` when an inspectable
        per-environment SceneGraph visual is required. Handles use their registered
        descriptor's appearance unless an explicit path overrides the source.
        """
        self._require_valid()
        obj_id = int(sim_handle.obj_id)
        env_ids = tuple(int(eid) for eid in sim_handle.env_ids)
        name = _safe_prim_name(getattr(sim_handle, "name", "") or f"obj_{obj_id}")
        base_path = path or f"/{name}"
        unexpected = set(kwargs) - {
            "scale",
            "order",
            "material",
            "color",
            "collision_path",
            "collision_material",
            "show_collision",
        }
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise TypeError(f"visual.add() received unsupported options: {names}")
        use_gpu = str(self.world.sim_device).startswith("cuda")

        if all((eid, obj_id) in self.world.articulations for eid in env_ids):
            add_batch = (
                self._add_gpu_articulation
                if use_gpu
                else self._add_cpu_external_articulation
            )
        elif all((eid, obj_id) in self.world.rigids for eid in env_ids):
            add_batch = self._add_rigid
        elif all((eid, obj_id) in self.world.static_rigids for eid in env_ids):
            # Static actors have no PhysX GPU state row.
            add_batch = self._add_rigid
        else:
            raise KeyError(
                f"simulation handle obj_id={obj_id} does not match registered objects"
            )
        if "collision_path" in kwargs:
            kwargs["collision_base_path"] = kwargs.pop("collision_path")
        return add_batch(
            sim_handle,
            mjcf_path,
            prim_base_path=base_path,
            **kwargs,
        )

    def add_scene_graph(
        self,
        sim_handle: SimRigid | SimArticulation | SimRigidBatch | SimArticulationBatch,
        mjcf_path: str | None = None,
        *,
        path: str | None = None,
        env_id: int | None = None,
        **kwargs,
    ) -> VisualArticulationSceneGraph | VisualRigidSceneGraph:
        """Create one inspectable SceneGraph visual for a simulation object.

        Unlike :meth:`add`, this path creates ordinary SceneGraph-owned prims
        and is intended for detailed playback and editor inspection. A batch
        handle requires ``env_id`` so accidentally expanding every training
        environment into prims is impossible.
        """
        self._require_valid()
        obj_id = int(sim_handle.obj_id)
        env_ids = tuple(int(value) for value in sim_handle.env_ids)
        name = _safe_prim_name(getattr(sim_handle, "name", "") or f"obj_{obj_id}")
        base_path = path or f"/{name}"
        if env_id is None:
            if len(env_ids) != 1:
                raise ValueError(
                    "add_scene_graph() requires env_id for a batched handle"
                )
            selected_env_id = env_ids[0]
        else:
            selected_env_id = int(env_id)
            if selected_env_id not in env_ids:
                raise KeyError(
                    f"env_id={selected_env_id} is outside handle env_ids={env_ids}"
                )
        key = (selected_env_id, obj_id)
        if key in self.world.articulations:
            return self.add_articulation_scene_graph(
                selected_env_id,
                obj_id,
                mjcf_path,
                path=base_path,
                **kwargs,
            )
        if key in self.world.rigids or key in self.world.static_rigids:
            return self.add_rigid_scene_graph(
                selected_env_id,
                obj_id,
                mjcf_path,
                path=base_path,
                **kwargs,
            )
        raise KeyError(
            f"simulation handle env={selected_env_id}, obj={obj_id} is not registered"
        )

    def add_articulation_scene_graph(
        self,
        env_id: int,
        obj_id: int,
        asset_path: str | None = None,
        *,
        path: str = "/robot",
        scale: float = 1.0,
        order: str = "DFS",
        material=None,
        add_shapes: bool = True,
        collision_path: str | None = None,
        collision_material=None,
        show_collision: bool = False,
        color=None,
        _debug_registration: bool = True,
    ) -> VisualArticulationSceneGraph:
        """Register one articulation through the small-scene SceneGraph path.

        Prefer ``add(sim_handle, ...)`` for normal simulation viewers. This path
        is kept for editor/debug tools, collision visual inspection, and
        MimicKit-style per-body SceneGraph control.
        """
        self._require_valid()
        key = (int(env_id), int(obj_id))
        if key in self.visual_articulation_scene_graphs:
            raise ValueError(
                f"visual already registered for env={key[0]}, obj={key[1]}"
            )

        articulation = self.world.articulation(key[0], key[1])
        asset, mesh_asset_base_path, data = self._articulation_visual_asset(
            asset_path, scale, order,
            data=self.world.articulations[key].source_data,
        )
        articulation_visual = asset.instantiate(
            self.scene,
            path,
            mesh_asset_base_path,
            True,
            hierarchical=True,
        )

        body_prims, render_body_ids = _physical_visual_mapping(
            articulation_visual, articulation
        )
        render_prims = list(articulation_visual.render_prims())
        body_handles, base_colors = self._register_articulation_parts(
            articulation_visual, data, material=material, color=color,
            add_shapes=add_shapes,
        )
        handle_body_ids = dict(zip(body_handles, render_body_ids))
        self.physics_bridge.add(articulation, articulation_visual)
        if _debug_registration:
            _debug_instancing(
                kind="sim-scenegraph",
                env_id=key[0],
                obj_id=key[1],
                num_bodies=len(body_prims),
                handles=body_handles,
                mesh_asset_base_path=mesh_asset_base_path,
            )

        collision_prims = []
        if collision_path is not None:
            collision_prims = list(
                self.physics_bridge.add_collision_visuals(
                    articulation,
                    self.scene,
                    collision_path,
                    bool(show_collision),
                )
            )
            shape_material = (
                visual_material(self.app, None, override=material)
                if collision_material is None
                else collision_material
            )
            if add_shapes and shape_material is not None:
                for prim in collision_prims:
                    self.app.scene.add_renderable(prim, shape_material)

        record = VisualArticulationSceneGraph(
            key[0],
            key[1],
            articulation_visual,
            body_prims,
            render_prims,
            collision_prims,
            body_handles,
            handle_body_ids,
            base_colors=base_colors,
            owned_paths=(path, collision_path),
        )
        self.visual_articulation_scene_graphs[key] = record
        return record

    def _add_gpu_articulation(
        self,
        sim_view,
        mjcf_path: str | None,
        prim_base_path: str = "/gpu_robot",
        scale: float = 1.0,
        order: str = "DFS",
        material=None,
        color=None,
        collision_base_path: str | None = None,
        collision_material=None,
        show_collision: bool = False,
    ) -> VisualBatch:
        """Create renderables for visual meshes backed by CUDA instance transforms."""
        self._require_valid()
        if not hasattr(_ke, "articulation_link_state_to_mat4_cuda"):
            raise RuntimeError("KangEngine was built without CUDA transform kernels")

        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(env_id) for env_id in sim_view.env_ids)
        if obj_id in self.gpu_visual_batches and self.gpu_visual_batches[obj_id].is_valid:
            raise ValueError(f"GPU visual batch already registered for obj={obj_id}")
        if any((env_id, obj_id) not in self.world.articulations for env_id in env_ids):
            raise KeyError(f"articulation obj={obj_id} is not registered in every env")

        asset, mesh_asset_base_path, data = self._articulation_visual_asset(
            mjcf_path, scale, order,
            data=self.world.articulations[(env_ids[0], obj_id)].source_data,
        )
        articulation_visual = asset.instantiate(
            self.scene, prim_base_path, mesh_asset_base_path, True
        )
        body_prims, render_body_ids = _physical_visual_mapping(
            articulation_visual, sim_view.articulation
        )
        body_handles, base_colors = self._register_articulation_parts(
            articulation_visual, data, material=material, external=True,
        )
        collision_prims = self._add_articulation_collision_visuals(
            sim_view.articulation,
            collision_base_path,
            (
                material
                if collision_material is None
                else collision_material
            ),
            show_collision,
        )
        backend = ArticulationGPUExternalBackend(
            self.app,
            self.world,
            obj_id,
            env_ids,
            body_prims,
            body_handles,
            collision_prims,
            render_body_ids=render_body_ids,
            base_colors=base_colors,
            owned_paths=(prim_base_path, collision_base_path),
        )
        batch = VisualBatch(obj_id, env_ids, backend=backend)
        batch.sync()
        batch.set_color(color)
        self.gpu_visual_batches[obj_id] = batch
        self.visual_batches[obj_id] = batch
        return batch

    def _add_cpu_external_articulation(
        self,
        sim_view,
        mjcf_path: str | None,
        prim_base_path: str = "/cpu_external_robot",
        scale: float = 1.0,
        order: str = "DFS",
        material=None,
        color=None,
        collision_base_path: str | None = None,
        collision_material=None,
        show_collision: bool = False,
    ) -> VisualBatch:
        """Create renderables for visual meshes backed by CPU ExternalBuffer."""
        self._require_valid()

        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(env_id) for env_id in sim_view.env_ids)
        if obj_id in self.visual_batches and self.visual_batches[obj_id].is_valid:
            raise ValueError(f"visual batch already registered for obj={obj_id}")
        if obj_id in self.cpu_visual_batches and self.cpu_visual_batches[obj_id].is_valid:
            raise ValueError(f"CPU visual batch already registered for obj={obj_id}")
        if obj_id in self.gpu_visual_batches and self.gpu_visual_batches[obj_id].is_valid:
            raise ValueError(f"GPU visual batch already registered for obj={obj_id}")
        if any((env_id, obj_id) not in self.world.articulations for env_id in env_ids):
            raise KeyError(f"articulation obj={obj_id} is not registered in every env")

        asset, mesh_asset_base_path, data = self._articulation_visual_asset(
            mjcf_path, scale, order,
            data=self.world.articulations[(env_ids[0], obj_id)].source_data,
        )
        articulation_visual = asset.instantiate(
            self.scene, prim_base_path, mesh_asset_base_path, True
        )
        body_prims, render_body_ids = _physical_visual_mapping(
            articulation_visual, sim_view.articulation
        )
        body_handles, base_colors = self._register_articulation_parts(
            articulation_visual, data, material=material, external=True,
        )
        collision_prims = self._add_articulation_collision_visuals(
            sim_view.articulation,
            collision_base_path,
            (
                material
                if collision_material is None
                else collision_material
            ),
            show_collision,
        )
        backend = ArticulationCPUExternalBackend(
            self.app,
            self.world,
            obj_id,
            env_ids,
            body_prims,
            body_handles,
            collision_prims,
            render_body_ids=render_body_ids,
            base_colors=base_colors,
            owned_paths=(prim_base_path, collision_base_path),
        )
        batch = VisualBatch(obj_id, env_ids, backend=backend)
        batch.sync()
        batch.set_color(color)
        self.cpu_visual_batches[obj_id] = batch
        self.visual_batches[obj_id] = batch
        return batch

    def _add_rigid(
        self,
        sim_view,
        mjcf_path: str | None = None,
        prim_base_path: str = "/rigid",
        scale: float = 1.0,
        order: str = "DFS",
        material=None,
        color=None,
    ) -> VisualBatch:
        """Group actor appearances while preserving each environment's instance row."""
        self._require_valid()
        obj_id = int(sim_view.obj_id)
        env_ids = tuple(int(env_id) for env_id in sim_view.env_ids)
        use_gpu = (
            self.world.sim_device.type == "cuda"
            and (env_ids[0], obj_id) in self.world.rigids
        )
        previous = self.visual_batches.get(obj_id)
        if previous is not None and previous.is_valid:
            raise ValueError(f"visual batch already registered for obj={obj_id}")
        if self.app.scene.get_prim_at_path(prim_base_path) is not None:
            raise ValueError(f"scene path already exists: {prim_base_path}")

        # An explicit MJCF overrides appearance; otherwise use the registered source.
        from ...asset.rigid_asset import _as_rigid_desc

        override = (
            _as_rigid_desc(self.world.load_mjcf(mjcf_path, scale=scale, order=order))
            if mjcf_path is not None else None
        )
        sources = {}
        for row, env_id in enumerate(env_ids):
            key = (env_id, obj_id)
            record = self.world.rigids.get(key) or self.world.static_rigids[key]
            data = record.source_data if override is None else override
            source_key = id(data)
            if source_key not in sources:
                sources[source_key] = ([], data)
            sources[source_key][0].append(row)

        self.app.scene.define_prim(prim_base_path, _ke.scene.PrimType.XFORM)
        groups = []
        try:
            for group_id, (rows, data) in enumerate(sources.values()):
                visual = RigidVisual(
                    self.app, f"{prim_base_path}/asset_{group_id}", data,
                    material=material,
                    transform_source=render_api.TransformSource.EXTERNAL_BUFFER,
                )
                groups.append((tuple(rows), visual))
            backend_type = RigidGPUExternalBackend if use_gpu else RigidCPUExternalBackend
            backend = backend_type(
                self.app, self.world, obj_id, env_ids, prim_base_path, groups
            )
            batch = VisualBatch(obj_id, env_ids, backend=backend)
            batch.sync()
            batch.set_color(color)
        except Exception:
            self.app.scene.remove_prim(prim_base_path)
            raise

        self.visual_batches[obj_id] = batch
        target = (
            self.gpu_visual_batches
            if use_gpu
            else self.cpu_visual_batches
        )
        target[obj_id] = batch
        return batch

    def add_rigid_scene_graph(
        self,
        env_id: int,
        obj_id: int,
        mjcf_path: str | None = None,
        *,
        path: str = "/rigid",
        scale: float = 1.0,
        order: str = "DFS",
        material=None,
        add_shapes: bool = True,
        color=None,
        _debug_registration: bool = True,
    ) -> VisualRigidSceneGraph:
        """Register one rigid object through the small-scene SceneGraph path.

        Prefer ``add(sim_handle, ...)`` for normal simulation viewers. This path
        remains for editor/debug tools and adapter code that needs direct prim
        access.
        """
        self._require_valid()
        key = (int(env_id), int(obj_id))
        if (
            key in self.visual_rigid_scene_graphs
            or key in self.visual_articulation_scene_graphs
        ):
            raise ValueError(
                f"visual already registered for env={key[0]}, obj={key[1]}"
            )

        source = self.world.rigids.get(key) or self.world.static_rigids[key]
        rigid = source.rigid
        from ...asset.rigid_asset import _as_rigid_desc

        data = (
            _as_rigid_desc(self.world.load_mjcf(mjcf_path, scale=scale, order=order))
            if mjcf_path is not None else source.source_data
        )
        rigid_visual = RigidVisual(
            self.app,
            path,
            data,
            material=self._resolve_visual_material(material),
            add_shapes=add_shapes,
            color=color,
        )

        record = VisualRigidSceneGraph(
            key[0],
            key[1],
            rigid,
            rigid_visual,
            list(rigid_visual.body_prims),
            list(rigid_visual.body_handles),
        )
        self.visual_rigid_scene_graphs[key] = record
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

    def add_articulation_skin(
        self,
        env_id: int,
        obj_id: int,
        mjcf_path: str,
        *,
        path: str = "/robot",
        scale: float = 1.0,
        order: str = "DFS",
        material=None,
        add_shapes: bool = True,
        color=None,
    ) -> VisualArticulationSceneGraph:
        """Create an adapter/debug articulation skin without PhysX ownership."""
        self._require_valid()
        key = (int(env_id), int(obj_id))
        if key in self.visual_articulation_scene_graphs:
            raise ValueError(
                f"visual already registered for env={key[0]}, obj={key[1]}"
            )

        asset, mesh_asset_base_path, data = self._articulation_visual_asset(
            mjcf_path, scale, order
        )
        articulation_visual = asset.instantiate(
            self.scene,
            path,
            mesh_asset_base_path,
            True,
        )
        body_prims = list(articulation_visual.body_prims())
        render_prims = list(articulation_visual.render_prims())
        render_body_ids = [
            int(i) for i in articulation_visual.render_prim_body_indices()
        ]
        body_handles, base_colors = self._register_articulation_parts(
            articulation_visual, data, material=material, color=color,
            add_shapes=add_shapes,
        )
        handle_body_ids = dict(zip(body_handles, render_body_ids))
        _debug_instancing(
            kind="visual-scenegraph",
            env_id=key[0],
            obj_id=key[1],
            num_bodies=len(body_prims),
            handles=body_handles,
            mesh_asset_base_path=mesh_asset_base_path,
        )

        record = VisualArticulationSceneGraph(
            key[0],
            key[1],
            articulation_visual,
            body_prims,
            render_prims,
            [],
            body_handles,
            handle_body_ids,
            base_colors=base_colors,
            owned_paths=(path,),
        )
        self.visual_articulation_scene_graphs[key] = record
        return record

    def sync(self):
        self._require_valid()
        self.physics_bridge.sync()
        self._sync_rigids()
        for batch in self.cpu_visual_batches.values():
            if batch.is_valid:
                batch.sync()
        for batch in self.gpu_visual_batches.values():
            if batch.is_valid:
                batch.sync()
        self._sync_frame_axes()

    def set_frames_visible(
        self, obj_id: int, visible: bool, *, env_id: int = 0,
        frame_names=None, length: float = 0.1,
    ):
        """Show selected attachment axes; disabled frames perform no pose queries.

        CUDA poses are copied to the host only while this debug display is on.
        ``frame_names=None`` selects every named fixed frame of one environment.
        """
        self._require_valid()
        key = (int(env_id), int(obj_id))
        if visible:
            robot = self.world.articulation(*key)
            if isinstance(frame_names, str):
                raise TypeError("frame_names must be a sequence of names")
            names = tuple(robot.frame_names if frame_names is None else frame_names)
            if not np.isfinite(length) or length <= 0:
                raise ValueError("length must be finite and positive")
            for name in names:
                if name not in robot.frame_names:
                    raise KeyError(name)
        old = self._frame_axes.pop(key, None)
        if old is not None:
            for path in old["paths"]:
                self.app.debug_overlay.clear(path)
        if visible and names:
            self._frame_axes[key] = {
                "names": names, "length": float(length),
                "paths": tuple(f"/debug/frames/env_{key[0]}/obj_{key[1]}/frame_{i}"
                               for i in range(len(names))),
            }
        return self

    def _sync_frame_axes(self):
        for (env_id, obj_id), config in self._frame_axes.items():
            names = config["names"]
            pos = self.world.state.get_frame_pos(obj_id, frame_names=names)
            rot = self.world.state.get_frame_quat_xyzw(obj_id, frame_names=names, fetch=False)
            pos = pos[env_id].detach().cpu().numpy()
            rot = rot[env_id].detach().cpu().numpy()
            for path, p, q in zip(config["paths"], pos, rot):
                self.app.debug_overlay.axes(path, p, quat_xyzw_to_wxyz(q), length=config["length"])

    def get_visual_articulation_scene_graph(
        self,
        env_id: int,
        obj_id: int,
    ) -> VisualArticulationSceneGraph | None:
        self._require_valid()
        return self.visual_articulation_scene_graphs.get((int(env_id), int(obj_id)))

    def get_visual_rigid_scene_graph(
        self, env_id: int, obj_id: int
    ) -> VisualRigidSceneGraph | None:
        self._require_valid()
        return self.visual_rigid_scene_graphs.get((int(env_id), int(obj_id)))

    def get_visual_scene_graph(
        self,
        env_id: int,
        obj_id: int,
    ) -> VisualArticulationSceneGraph | VisualRigidSceneGraph | None:
        self._require_valid()
        key = (int(env_id), int(obj_id))
        return self.visual_articulation_scene_graphs.get(
            key
        ) or self.visual_rigid_scene_graphs.get(key)

    def get_visual_batch(self, obj_id: int) -> VisualBatch | None:
        self._require_valid()
        return self.visual_batches.get(int(obj_id))

    def pick_body(self, selection) -> VisualBodyPick | None:
        """Resolve a renderer selection to visual env/object/body metadata."""
        self._require_valid()
        if selection is None or not getattr(selection, "hit", False):
            return None

        for batch in self.visual_batches.values():
            if not batch.is_valid:
                continue
            hit = batch.pick_body(selection)
            if hit is not None:
                return hit
        for record in self.visual_articulation_scene_graphs.values():
            hit = record.pick_body(selection)
            if hit is not None:
                return hit
        for record in self.visual_rigid_scene_graphs.values():
            hit = record.pick_body(selection)
            if hit is not None:
                return hit
        return None

    def _sync_rigids(self):
        for record in self.visual_rigid_scene_graphs.values():
            record.rigid_visual.apply_pose(
                position=record.rigid.get_root_position(),
                quat_xyzw=record.rigid.get_root_quat_xyzw(),
            )

    def set_body_transforms_scene_graph(
        self, env_id: int, obj_id: int, body_pos=None, body_quat_xyzw=None
    ):
        """Override rendered body prim transforms with world-space FK poses.

        This is used by MimicKit view_motion, where the environment computes the
        reference pose itself and expects the engine viewer to draw that pose.
        """
        record = self.get_visual_articulation_scene_graph(env_id, obj_id)
        if record is None:
            return

        if body_pos is not None:
            for prim, pos in zip(record.body_prims, body_pos):
                prim.set_world_translation(
                    _ke.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                )

        if body_quat_xyzw is not None:
            for prim, rot in zip(record.body_prims, body_quat_xyzw):
                prim.set_world_quat_wxyz(
                    _ke.Quat.from_xyzw(rot),
                )

    def set_root_transform_scene_graph(
        self, env_id: int, obj_id: int, root_pos=None, root_quat_xyzw=None
    ):
        """Apply a root-only fallback pose to a visual articulation.

        MimicKit visual/reference objects are not backed by PhysX, but some envs
        still drive them through root setters before or instead of body setters.
        ArticulationVisual keeps a zero-pose FK model that can at least move the
        whole rendered character with that root pose.
        """
        record = self.get_visual_articulation_scene_graph(env_id, obj_id)
        if record is None:
            return

        if root_pos is not None:
            pos = root_pos
            record.articulation_visual.set_root_translation(
                _ke.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            )

        if root_quat_xyzw is not None:
            rot = root_quat_xyzw
            record.articulation_visual.set_joint_quat_wxyz(
                0,
                _ke.Quat.from_xyzw(rot),
            )

        if root_pos is not None or root_quat_xyzw is not None:
            record.articulation_visual.apply_pose()

    def set_collision_visible(self, visible: bool):
        self.physics_bridge.set_collision_visible(bool(visible))
        for batch in self.visual_batches.values():
            if batch.is_valid:
                batch.set_collision_visible(bool(visible))

    def body_id_from_render_handle_scene_graph(
        self, env_id: int, obj_id: int, handle
    ) -> int | None:
        """Low-level scene-graph handle lookup; prefer pick_body(selection)."""
        record = self.get_visual_articulation_scene_graph(env_id, obj_id)
        if record is None:
            return None
        return record.body_id_from_render_handle(handle)

    def set_articulation_color_scene_graph(self, env_id: int, obj_id: int, color):
        record = self.get_visual_articulation_scene_graph(env_id, obj_id)
        if record is None:
            return
        record.set_color(color)

    def _register_articulation_parts(
        self, visual, data, *, material=None, color=None, external=False,
        add_shapes=True,
    ):
        handles, base_colors = [], []
        geoms = list(data.visual_geoms)
        tint = _normalize_color_array(color)
        transform_source = (
            render_api.TransformSource.EXTERNAL_BUFFER if external
            else render_api.TransformSource.SCENE_GRAPH
        )

        # Preserve every authored part's UVs and material while sharing body poses.
        for prim, visual_index in zip(
            visual.render_prims(), visual.render_prim_visual_indices()
        ):
            geom = geoms[visual_index] if visual_index >= 0 else None
            base = visual_color(geom) if geom is not None else np.ones(4, dtype=np.float32)
            base_colors.append(base)
            display = base if tint is None else tint * base
            mesh = prim.resolve_mesh_data()
            if not add_shapes:
                component = prim.get_mesh_component() or prim.add_mesh_component()
                component.mesh_data = mesh
                component.resource_handle = self.app._register_mesh_resource(mesh)
                prim.set_display_color_alpha(_ke.Vec4(*display))
                continue

            part_material = visual_material(
                self.app, geom, asset_dir=data.asset_dir,
                override=material,
            )
            view = self.app.scene.add_mesh(
                prim.get_path(), mesh, material=part_material,
                color=_ke.Vec4(*display), transform_source=transform_source,
            )
            apply_visual_alpha(view, part_material, base)
            handles.append(int(self.app.get_scene_render_system()._handle(view.component)))
        return handles, base_colors

    def _articulation_visual_asset(
        self,
        asset_path: str | None,
        scale: float,
        order: str,
        *,
        data=None,
    ):
        scale = float(scale)
        key = (("data", id(data)) if asset_path is None
               else (str(asset_path), scale, str(order)))
        record = self._articulation_visual_assets.get(key)
        if record is not None:
            return record

        if asset_path is not None:
            suffix = Path(asset_path).suffix.lower()
            if suffix == ".xml":
                data = self.world.load_mjcf(asset_path, scale=scale, order=order)
            elif suffix == ".urdf":
                data = self.world.load_urdf(asset_path, scale=scale, order=order)
            elif suffix in (".usd", ".usda", ".usdc"):
                if scale != 1.0:
                    raise ValueError("USD articulation visuals require scale=1")
                data = self.world.load_usd(asset_path, order=order)
            else:
                raise ValueError(f"unsupported articulation asset extension: {asset_path}")
            mesh_asset_base_path = _mesh_asset_base_path(asset_path, scale, order)
        else:
            mesh_asset_base_path = f"/.Resources/Meshes/Articulations/source_{id(data)}"

        asset = ArticulationVisualAsset.from_data(
            data=data, scale=1.0, asset_path=asset_path or "",
        )
        asset.define_mesh_assets(self.scene, mesh_asset_base_path, True)
        record = (asset, mesh_asset_base_path, data)
        self._articulation_visual_assets[key] = record
        return record

    def _resolve_visual_material(self, material=None, *, fallback=None):
        if material is not None:
            return material
        return fallback

    def _add_articulation_collision_visuals(
        self,
        articulation,
        collision_base_path: str | None,
        material,
        visible: bool,
    ):
        if collision_base_path is None:
            return ()
        material = visual_material(self.app, None, override=material)
        collision_prims = tuple(
            self.physics_bridge.add_collision_visuals(
                articulation,
                self.scene,
                collision_base_path,
                bool(visible),
            )
        )
        if material is not None:
            for prim in collision_prims:
                self.app.scene.add_renderable(prim, material)
        return collision_prims
