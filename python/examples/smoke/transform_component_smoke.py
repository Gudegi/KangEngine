"""Validate TransformComponent lifecycle and hierarchy dirty propagation."""

from __future__ import annotations

import math

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
    scene = ke.scene.create_backend(ke.scene.BackendType.Native)

    parent = scene.define_prim("/World/Parent", ke.scene.PrimType.Xform)
    child = scene.define_prim("/World/Parent/Child", ke.scene.PrimType.Xform)

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

    parent_version = parent_transform.version
    parent.set_local_translation(ke.vec3(1.0, 0.0, 0.0))
    if parent_transform.version != parent_version + 1:
        raise AssertionError(
            "set_local_translation should advance TransformComponent version once"
        )
    child.set_local_translation(ke.vec3(0.0, 2.0, 0.0))
    _close_tuple(_vec3_tuple(child.get_local_translation()), (0.0, 2.0, 0.0))
    _close_tuple(_vec3_tuple(child.get_world_translation()), (1.0, 2.0, 0.0))
    _close_tuple(_translation(child.compute_world_matrix()), (1.0, 2.0, 0.0))
    _close_tuple(_translation(child_transform.compute_world_matrix()), (1.0, 2.0, 0.0))

    version = child_transform.version
    parent.set_local_translation(ke.vec3(3.0, 0.0, 0.0))
    if child_transform.version != version + 1:
        raise AssertionError(
            "parent transform change should advance child world version once"
        )
    _close_tuple(_translation(child.compute_world_matrix()), (3.0, 2.0, 0.0))

    version = child_transform.version
    child_transform.set_world_translation(ke.vec3(10.0, 5.0, 0.0))
    if child_transform.version != version + 1:
        raise AssertionError(
            "set_world_translation should advance TransformComponent version once"
        )
    _close_tuple(_translation(child.compute_world_matrix()), (10.0, 5.0, 0.0))
    _close_tuple(_translation(child.compute_local_matrix()), (7.0, 5.0, 0.0))

    child.set_local_rotation_axis_angle(ke.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
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
    local_rotation = _quat_wxyz(child.get_local_rotation())
    if local_rotation[0] < 0.0:
        local_rotation = tuple(-value for value in local_rotation)
    _close_tuple(local_rotation, expected_local_rotation)

    child_transform.set_world_rotation_axis_angle(ke.vec3(0.0, 1.0, 0.0), math.pi / 4.0)
    expected_world_rotation = (
        math.cos(math.pi / 8.0),
        0.0,
        math.sin(math.pi / 8.0),
        0.0,
    )
    world_rotation = _quat_wxyz(child_transform.get_world_rotation())
    if world_rotation[0] < 0.0:
        world_rotation = tuple(-value for value in world_rotation)
    _close_tuple(world_rotation, expected_world_rotation)
    try:
        child.set_local_rotation_axis_angle(ke.vec3(0.0, 0.0, 0.0), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero rotation axis should be rejected")

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
