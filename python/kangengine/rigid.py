"""Rigid-object metadata helpers shared by state and visual paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "RigidShapeSpec",
    "rigid_shape_specs",
    "rigid_body_names",
    "expand_rigid_body_state",
]


@dataclass(slots=True)
class RigidShapeSpec:
    name: str
    geom_type: str
    local_pos: np.ndarray
    local_rot: np.ndarray
    size: np.ndarray


def rigid_shape_specs(data) -> list[RigidShapeSpec]:
    """Build render/state slots from MJCF collision geoms.

    KangEngine represents MimicKit rigid objects as one PhysX rigid actor with
    one or more attached shapes. The slots returned here are visual/body API
    slots for those shapes, not separate simulated rigid bodies.
    """

    geoms_by_body = getattr(data, "collision_geoms", {})
    names = _node_names(data)
    specs: list[RigidShapeSpec] = []
    for body_idx in range(len(names)):
        geoms = list(geoms_by_body.get(body_idx, []))
        for geom_idx, geom in enumerate(geoms):
            base_name = names[body_idx] if body_idx < len(names) else f"body_{body_idx}"
            slot_name = base_name if len(geoms) == 1 else f"{base_name}:geom_{geom_idx}"
            specs.append(_shape_spec(slot_name, geom))

    if specs:
        return specs

    root_name = names[0] if names else "rigid"
    return [
        RigidShapeSpec(
            root_name,
            "Sphere",
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.1, 0.0, 0.0], dtype=np.float32),
        )
    ]


def rigid_body_names(data) -> list[str]:
    return [spec.name for spec in rigid_shape_specs(data)]


def expand_rigid_body_state(root_pos, root_rot, local_pos, local_rot):
    root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
    root_rot = _quat_normalize(np.asarray(root_rot, dtype=np.float32).reshape(4))
    local_pos = np.asarray(local_pos, dtype=np.float32).reshape(-1, 3)
    local_rot = np.asarray(local_rot, dtype=np.float32).reshape(-1, 4)
    body_pos = np.stack(
        [root_pos + _quat_rotate(root_rot, pos) for pos in local_pos], axis=0
    ).astype(np.float32)
    body_rot = np.stack(
        [_quat_normalize(_quat_mul(root_rot, rot)) for rot in local_rot], axis=0
    ).astype(np.float32)
    return body_pos, body_rot


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_rotate(q, v):
    q = _quat_normalize(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    return _quat_mul(_quat_mul(q, vq), _quat_conj(q))[:3]


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_from_two_vectors(src, dst):
    src = _normalize_vec(np.asarray(src, dtype=np.float32).reshape(3))
    dst = _normalize_vec(np.asarray(dst, dtype=np.float32).reshape(3))
    dot = float(np.dot(src, dst))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if dot < -0.999999:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = _normalize_vec(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
    axis = np.cross(src, dst)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot], dtype=np.float32)
    return _quat_normalize(q)


def _shape_spec(name: str, geom) -> RigidShapeSpec:
    geom_type = _geom_type_name(geom)
    size = np.asarray(geom.size, dtype=np.float32).reshape(3)
    if geom_type in ("Capsule", "Cylinder"):
        if bool(geom.has_from_to):
            start = _vec3(geom.from_pos)
            end = _vec3(geom.to_pos)
            local_pos = (start + end) * 0.5
            axis = end - start
            local_rot = _quat_from_two_vectors([1.0, 0.0, 0.0], axis)
            size = size.copy()
            size[1] = np.linalg.norm(axis) * 0.5
        else:
            local_pos = _vec3(geom.pos)
            mjcf_rot = _quat(geom.quat)
            axis = _quat_rotate(
                mjcf_rot, np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
            local_rot = _quat_from_two_vectors([1.0, 0.0, 0.0], axis)
    else:
        local_pos = _vec3(geom.pos)
        local_rot = _quat(geom.quat) if geom_type == "Box" else np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )

    return RigidShapeSpec(
        name,
        geom_type,
        local_pos.astype(np.float32),
        _quat_normalize(local_rot),
        size.astype(np.float32),
    )


def _node_names(data):
    tree = getattr(data, "skeleton_tree", None)
    if tree is None:
        return []
    return list(tree.node_names())


def _geom_type_name(geom):
    value = geom.type
    return getattr(value, "name", str(value).split(".")[-1])


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
    return _quat_normalize(arr)


def _normalize_vec(v):
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _quat_normalize(q):
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)
