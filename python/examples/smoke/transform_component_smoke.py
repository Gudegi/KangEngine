"""Validate TransformComponent lifecycle and hierarchy dirty propagation."""

from __future__ import annotations

import math

import numpy as np

import kangengine as ke


def _translation(matrix):
    # GLM matrices are exposed as column vectors; the 4th column stores
    # translation for KangEngine's transform convention.
    col = matrix[3]
    return (float(col.x), float(col.y), float(col.z))


def _close_tuple(actual, expected, eps=1.0e-5):
    for i, (a, b) in enumerate(zip(actual, expected)):
        if abs(a - b) > eps:
            raise AssertionError(f"value {i} mismatch: {a} != {b}")


def _vec3_tuple(value):
    return (float(value.x), float(value.y), float(value.z))


def _quat_wxyz(value):
    return (float(value.w), float(value.x), float(value.y), float(value.z))


def main():
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)

    source = scene.define_prim("/source", ke.scene.PrimType.XFORM)
    source.set_local_scale(ke.Vec3(2, 3, 4))
    source.set_local_rotation_axis_angle(ke.Vec3(0, 0, 1), math.pi / 2)
    source.set_local_translation(ke.Vec3(1, 2, 3))
    matrix = source.compute_local_matrix()
    for use_component in (False, True):
        mixed = scene.define_prim(f"/mixed_{use_component}", ke.scene.PrimType.XFORM)
        api = mixed.get_transform_component() if use_component else mixed
        api.set_local_matrix(matrix)
        api.set_local_translation(ke.Vec3(8, 9, 10))
        _close_tuple(_translation(mixed.compute_local_matrix()), (8, 9, 10))
        for i in range(3):
            _close_tuple(_vec3_tuple(mixed.compute_local_matrix()[i]),
                         _vec3_tuple(matrix[i]))
        api.set_local_scale(ke.Vec3(5, 6, 7))
        _close_tuple(_translation(mixed.compute_local_matrix()), (8, 9, 10))
        _close_tuple(_vec3_tuple(mixed.compute_local_matrix()[0]), (0, 5, 0))
        api.set_local_matrix(matrix)
        api.set_local_rotation_axis_angle(ke.Vec3(0, 0, 1), 0.0)
        _close_tuple(_translation(mixed.compute_local_matrix()), (1, 2, 3))
        for i, scale in enumerate((2, 3, 4)):
            _close_tuple(_vec3_tuple(mixed.compute_local_matrix()[i]),
                         tuple(scale if axis == i else 0 for axis in range(3)))

    parent = scene.define_prim("/World/Parent", ke.scene.PrimType.XFORM)
    child = scene.define_prim("/World/Parent/Child", ke.scene.PrimType.XFORM)

    parent_transform = parent.get_transform_component()
    child_transform = child.get_transform_component()
    if parent_transform is None or child_transform is None:
        raise AssertionError("TransformComponent was not attached by default")
    if not parent_transform.attached or not child_transform.attached:
        raise AssertionError("TransformComponent should be attached")
    if parent_transform.owner is not parent:
        raise AssertionError("parent TransformComponent owner mismatch")
    if "TransformComponent" not in repr(parent_transform):
        raise AssertionError("TransformComponent repr missing type name")

    # Attribute callers and component setters must address the same source.
    version = child_transform.version
    child.set_attribute_vec3("xformOp:translate", ke.Vec3(4, 5, 6))
    assert child_transform.version == version + 1
    _close_tuple(_vec3_tuple(child_transform.get_local_translation()), (4, 5, 6))
    child_transform.set_local_translation(ke.Vec3(7, 8, 9))
    _close_tuple(_vec3_tuple(child.get_attribute_vec3("xformOp:translate")), (7, 8, 9))
    version = child_transform.version
    child.set_attribute_string("label", "child")
    assert child_transform.version == version
    assert child.get_attribute_string("label") == "child"

    parent_version = parent_transform.version
    parent.set_local_translation(ke.Vec3(1.0, 0.0, 0.0))
    if parent_transform.version != parent_version + 1:
        raise AssertionError(
            "set_local_translation should advance TransformComponent version once"
        )
    child.set_local_translation(ke.Vec3(0.0, 2.0, 0.0))
    _close_tuple(_vec3_tuple(child.get_local_translation()), (0.0, 2.0, 0.0))
    _close_tuple(_vec3_tuple(child.get_world_translation()), (1.0, 2.0, 0.0))
    _close_tuple(_translation(child.compute_world_matrix()), (1.0, 2.0, 0.0))
    _close_tuple(_translation(child_transform.compute_world_matrix()), (1.0, 2.0, 0.0))

    version = child_transform.version
    parent.set_local_translation(ke.Vec3(3.0, 0.0, 0.0))
    if child_transform.version != version + 1:
        raise AssertionError(
            "parent transform change should advance child world version once"
        )
    _close_tuple(_translation(child.compute_world_matrix()), (3.0, 2.0, 0.0))

    version = child_transform.version
    child_transform.set_world_translation(ke.Vec3(10.0, 5.0, 0.0))
    if child_transform.version != version + 1:
        raise AssertionError(
            "set_world_translation should advance TransformComponent version once"
        )
    _close_tuple(_translation(child.compute_world_matrix()), (10.0, 5.0, 0.0))
    _close_tuple(_translation(child.compute_local_matrix()), (7.0, 5.0, 0.0))

    child.set_local_rotation_axis_angle(ke.Vec3(0.0, 0.0, 1.0), math.pi / 2.0)
    if not child.has_attribute("xformOp:rotateQuaternion"):
        raise AssertionError("quaternion rotation attribute was not authored")
    local = child.compute_local_matrix()
    _close_tuple((float(local[0].x), float(local[0].y)), (0.0, 1.0))
    expected_local_rotation = (
        math.cos(math.pi / 4.0),
        0.0,
        0.0,
        math.sin(math.pi / 4.0),
    )
    local_rotation = _quat_wxyz(child.get_local_quat())
    if local_rotation[0] < 0.0:
        local_rotation = tuple(-value for value in local_rotation)
    _close_tuple(local_rotation, expected_local_rotation)

    child_transform.set_world_rotation_axis_angle(ke.Vec3(0.0, 1.0, 0.0), math.pi / 4.0)
    expected_world_rotation = (
        math.cos(math.pi / 8.0),
        0.0,
        math.sin(math.pi / 8.0),
        0.0,
    )
    world_rotation = _quat_wxyz(child_transform.get_world_quat())
    if world_rotation[0] < 0.0:
        world_rotation = tuple(-value for value in world_rotation)
    _close_tuple(world_rotation, expected_world_rotation)
    try:
        child.set_local_rotation_axis_angle(ke.Vec3(0.0, 0.0, 0.0), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero rotation axis should be rejected")

    child.set_world_matrix(matrix)
    child.set_world_translation(ke.Vec3(8, 9, 10))
    _close_tuple(_translation(child.compute_world_matrix()), (8, 9, 10))
    for i in range(3):
        _close_tuple(_vec3_tuple(child.compute_world_matrix()[i]),
                     _vec3_tuple(matrix[i]))

    # Translation does not require decomposing scale, shear, or perspective.
    probe = scene.define_prim("/World/Parent/Probe", ke.scene.PrimType.XFORM)
    for kind in ("affine", "shear", "reflection", "singular", "homogeneous", "perspective"):
        value = np.diag([2, 3, 4, 1]).astype(np.float32)
        value[:3, 3] = [4, 6, 8]
        if kind == "shear":
            value[0, 1] = 0.5
        elif kind == "reflection":
            value[0, 0] = -2
        elif kind == "singular":
            value[0, 0] = 0
        elif kind == "homogeneous":
            value *= 2
        elif kind == "perspective":
            value[3] = [0.1, 0.2, 0.3, 2]
        probe.set_local_matrix(value.T.copy())
        np.testing.assert_allclose(probe.get_local_translation(), value[:3, 3] / value[3, 3])
        world = np.asarray(probe.compute_world_matrix())
        np.testing.assert_allclose(probe.get_world_translation(), world[:3, 3] / world[3, 3])
    value[3, 3] = 0
    probe.set_local_matrix(value.T.copy())
    try:
        probe.get_local_translation()
    except ValueError:
        pass
    else:
        raise AssertionError("zero homogeneous w was accepted")

    if not parent.has_transform_component():
        raise AssertionError("Prim should report mandatory TransformComponent")

    if not scene.remove_prim("/World/Parent"):
        raise AssertionError("failed to remove transform subtree")
    if parent_transform.attached or child_transform.attached:
        raise AssertionError("TransformComponent stayed attached after subtree removal")
    try:
        child_transform.compute_world_matrix()
    except RuntimeError:
        pass
    else:
        raise AssertionError("detached TransformComponent remained usable")

    print("PASS: TransformComponent smoke completed")


if __name__ == "__main__":
    main()
