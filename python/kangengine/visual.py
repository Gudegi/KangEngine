"""Viewer-side helpers for KangSimWorld.

KangSimWorld intentionally stays headless. These helpers are owned by App or
examples that need scene Prim visuals synced from PhysX.
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
class _VisualArticulationRecord:
    env_id: int
    obj_id: int
    skeleton_bridge: object
    body_prims: list[object]
    collision_prims: list[object]
    body_handles: list[int]


@dataclass(slots=True)
class _VisualRigidRecord:
    env_id: int
    obj_id: int
    rigid: object
    rigid_bridge: object
    body_prims: list[object]


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
        for idx, spec in enumerate(self.specs):
            prim = self._define_shape_prim(prim_base_path, idx, spec)
            _apply_prim_color([prim], color)
            if add_shapes and shader is not None:
                app.addShape(shader, prim)
            self.body_prims.append(prim)

    def sync(self):
        root_pos = np.asarray(self.rigid.get_root_position(), dtype=np.float32)
        root_rot = np.asarray(self.rigid.get_root_rotation(), dtype=np.float32)
        body_pos, body_rot = expand_rigid_body_state(
            root_pos, root_rot, self.local_pos, self.local_rot
        )
        for prim, pos, rot in zip(self.body_prims, body_pos, body_rot):
            prim.set_attribute_vec3(
                "xformOp:translate",
                _ke.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
            )
            prim.set_attribute_quat(
                "xformOp:rotateQuaternion",
                _ke.quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])),
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
    """Owns PhysicsBridge/SkeletonBridge wiring for one KangSimWorld viewer."""

    def __init__(self, app, world):
        if not hasattr(_ke, "PhysicsBridge"):
            raise RuntimeError("KangWorldVisualBridge requires PhysicsBridge bindings")
        self.app = app
        self.world = world
        self.scene = app.getScene()
        self.physics_bridge = _ke.PhysicsBridge(app)
        self.records: dict[tuple[int, int], _VisualArticulationRecord] = {}
        self.rigid_records: dict[tuple[int, int], _VisualRigidRecord] = {}
        self._skeleton_assets = {}
        self._instanced_colors: dict[tuple[int, ...], np.ndarray] = {}

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
    ) -> _VisualArticulationRecord:
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
                    self.app.addShape(shader, prim, _ke.RenderTrack.Instanced)
                )
            self.physics_bridge.add_instanced(articulation, body_handles)
            self._append_instanced_color(body_handles, color)
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
                    self.app.addShape(shape_shader, prim)

        record = _VisualArticulationRecord(
            key[0],
            key[1],
            skeleton_bridge,
            body_prims,
            collision_prims,
            body_handles,
        )
        self.records[key] = record
        return record

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
    ) -> _VisualRigidRecord:
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

        record = _VisualRigidRecord(
            key[0], key[1], rigid, rigid_bridge, list(rigid_bridge.body_prims)
        )
        self.rigid_records[key] = record
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
    ) -> _VisualArticulationRecord:
        """Create a rendered skeleton that is not attached to a PhysX articulation."""
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
                self.app.addShape(shader, prim)
        _debug_instancing(
            kind="visual-scenegraph",
            env_id=key[0],
            obj_id=key[1],
            num_bodies=len(body_prims),
            handles=[],
            mesh_asset_base_path=mesh_asset_base_path,
        )

        record = _VisualArticulationRecord(
            key[0],
            key[1],
            skeleton_bridge,
            body_prims,
            [],
            [],
        )
        self.records[key] = record
        return record

    def sync(self):
        self.physics_bridge.sync()
        self._sync_rigids()

    def _sync_rigids(self):
        for record in self.rigid_records.values():
            record.rigid_bridge.sync()

    def set_body_transforms(self, env_id: int, obj_id: int, body_pos=None, body_rot=None):
        """Override rendered body prim transforms with world-space FK poses.

        This is used by MimicKit view_motion, where the environment computes the
        reference pose itself and expects the engine viewer to draw that pose.
        """
        record = self.records.get((int(env_id), int(obj_id)))
        if record is None:
            return

        if body_pos is not None:
            for prim, pos in zip(record.body_prims, body_pos):
                prim.set_attribute_vec3(
                    "xformOp:translate",
                    _ke.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                )

        if body_rot is not None:
            for prim, rot in zip(record.body_prims, body_rot):
                # MimicKit stores quaternions as xyzw; KangEngine's Python quat
                # constructor takes wxyz.
                prim.set_attribute_quat(
                    "xformOp:rotateQuaternion",
                    _ke.quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])),
                )

    def set_root_transform(self, env_id: int, obj_id: int, root_pos=None, root_rot=None):
        """Apply a root-only fallback pose to a visual articulation.

        MimicKit visual/reference objects are not backed by PhysX, but some envs
        still drive them through root setters before or instead of body setters.
        SkeletonBridge keeps a zero-pose FK model that can at least move the
        whole rendered character with that root pose.
        """
        record = self.records.get((int(env_id), int(obj_id)))
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
            self.app.setShapeColors(handle, colors)

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


def _apply_prim_color(prims, color):
    rgba = _normalize_color(color)
    if rgba is None:
        return
    for prim in prims:
        prim.set_display_color_alpha(rgba)


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


def _mesh_asset_base_path(mjcf_path: str, scale: float, order: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", str(mjcf_path).split("/")[-1]).strip("_")
    if not stem:
        stem = "character"
    digest_src = f"{mjcf_path}|{float(scale):.9g}|{order}".encode("utf-8")
    digest = hashlib.sha1(digest_src).hexdigest()[:10]
    return f"/mesh_assets/skeletons/{stem}_{digest}"
