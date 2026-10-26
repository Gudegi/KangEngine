"""Rigid-object metadata and local transforms for collision-based visuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .asset.rigid_asset import _as_rigid_desc
from .utils.math import (
    quat_xyzw_from_two_vectors,
    quat_xyzw_multiply,
    quat_xyzw_normalize,
    quat_xyzw_rotate,
)


__all__ = [
    "RigidShapeSpec",
    "rigid_shape_specs",
    "rigid_body_names",
    "expand_rigid_body_state",
]


@dataclass(slots=True)
class RigidShapeSpec:
    """One shape's pose relative to its actor: position (m) and XYZW quaternion."""

    name: str
    geom_type: str
    local_pos: np.ndarray
    local_quat_xyzw: np.ndarray
    size: np.ndarray
    mesh_data: Any = None


def rigid_shape_specs(data) -> list[RigidShapeSpec]:
    """Build shape metadata from a RigidDesc's collision geometry.

    These slots describe collision-based visual parts. Physical body state has
    one slot per actor; runtime collision queries read the actual PhysX shapes.
    """

    data = _as_rigid_desc(data)
    geoms = data.collision_geoms
    if geoms:
        return [
            _shape_spec(data.name if len(geoms) == 1 else f"{data.name}:geom_{index}", geom)
            for index, geom in enumerate(geoms)
        ]

    return [
        RigidShapeSpec(
            data.name,
            "Sphere",
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.1, 0.0, 0.0], dtype=np.float32),
        )
    ]


def rigid_body_names(data) -> list[str]:
    return [_as_rigid_desc(data).name]


def expand_rigid_body_state(root_pos, root_quat_xyzw, local_pos, local_quat_xyzw):
    """Compose actor-world and shape-local poses into world positions and XYZW quaternions."""
    root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
    root_quat_xyzw = quat_xyzw_normalize(np.asarray(root_quat_xyzw, dtype=np.float32).reshape(4))
    local_pos = np.asarray(local_pos, dtype=np.float32).reshape(-1, 3)
    local_quat_xyzw = np.asarray(local_quat_xyzw, dtype=np.float32).reshape(-1, 4)
    body_pos = np.stack(
        [root_pos + quat_xyzw_rotate(root_quat_xyzw, pos) for pos in local_pos], axis=0
    ).astype(np.float32)
    body_quat_xyzw = np.stack(
        [quat_xyzw_normalize(quat_xyzw_multiply(root_quat_xyzw, rot)) for rot in local_quat_xyzw],
        axis=0,
    ).astype(np.float32)
    return body_pos, body_quat_xyzw


def _shape_spec(name: str, geom) -> RigidShapeSpec:
    geom_type = _geom_type_name(geom)
    size = np.asarray(geom.size, dtype=np.float32).reshape(3)
    if geom_type in ("Capsule", "Cylinder"):
        if bool(geom.has_from_to):
            start = _vec3(geom.from_pos)
            end = _vec3(geom.to_pos)
            local_pos = (start + end) * 0.5
            axis = end - start
            local_quat_xyzw = quat_xyzw_from_two_vectors([1.0, 0.0, 0.0], axis)
            size = size.copy()
            size[1] = np.linalg.norm(axis) * 0.5
        else:
            local_pos = _vec3(geom.pos)
            mjcf_rot = _quat(geom.quat)
            axis = quat_xyzw_rotate(
                mjcf_rot, np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
            local_quat_xyzw = quat_xyzw_from_two_vectors([1.0, 0.0, 0.0], axis)
    else:
        local_pos = _vec3(geom.pos)
        local_quat_xyzw = (
            _quat(geom.quat)
            if geom_type in ("Box", "ConvexMesh", "TriangleMesh")
            else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )

    return RigidShapeSpec(
        name,
        geom_type,
        local_pos.astype(np.float32),
        quat_xyzw_normalize(local_quat_xyzw),
        size.astype(np.float32),
        getattr(geom, "mesh_data", None),
    )


def _geom_type_name(geom):
    value = geom.type
    name = getattr(value, "name", str(value).split(".")[-1])
    map = {
        "CAPSULE": "Capsule",
        "CYLINDER": "Cylinder",
        "SPHERE": "Sphere",
        "BOX": "Box",
        "CONVEX_MESH": "ConvexMesh",
        "TRIANGLE_MESH": "TriangleMesh",
    }
    try:
        return map[str(name).upper()]
    except KeyError as error:
        raise ValueError(
            f"unsupported rigid collision geometry type: {name}"
        ) from error


def _vec3(value):
    try:
        return np.asarray([value[0], value[1], value[2]], dtype=np.float32)
    except TypeError:
        return np.asarray([value.x, value.y, value.z], dtype=np.float32)


def _quat(value):
    try:
        arr = np.asarray([value[0], value[1], value[2], value[3]], dtype=np.float32)
    except TypeError:
        arr = np.asarray([value.x, value.y, value.z, value.w], dtype=np.float32)
    return quat_xyzw_normalize(arr)
