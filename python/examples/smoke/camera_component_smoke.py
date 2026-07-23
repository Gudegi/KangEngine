"""Validate CameraComponent lifecycle without requiring a GL window."""

from __future__ import annotations


import kangengine as ke


def _close_vec3(actual, expected, eps=1.0e-5):
    vals = [float(actual.x), float(actual.y), float(actual.z)]
    for i, (a, b) in enumerate(zip(vals, expected)):
        if abs(a - b) > eps:
            raise AssertionError(f"vec3 value {i} mismatch: {a} != {b}")


def main():
    scene = ke.scene.create_backend(ke.scene.BackendType.Native)

    camera_prim = scene.define_prim("/cameras/main", ke.scene.PrimType.Camera)
    camera_prim.set_local_translation(ke.vec3(1.0, 2.0, 3.0))
    component = camera_prim.add_camera_component()
    if component is None or not component.attached:
        raise AssertionError("add_camera_component did not attach CameraComponent")
    if component.owner.get_path() != camera_prim.get_path():
        raise AssertionError("CameraComponent owner mismatch")
    if component.projection_type != ke.scene.CameraProjectionType.Perspective:
        raise AssertionError("unexpected default projection type")
    if "CameraComponent" not in repr(component):
        raise AssertionError("CameraComponent repr missing type name")

    _close_vec3(component.position(), [1.0, 2.0, 3.0])
    _close_vec3(component.forward(), [0.0, 0.0, -1.0])
    _close_vec3(component.up(), [0.0, 1.0, 0.0])

    version = component.version
    component.set_perspective(60.0, 0.05, 500.0)
    if component.version <= version:
        raise AssertionError("CameraComponent version did not advance")
    if abs(component.vertical_fov_degrees() - 60.0) > 1.0e-5:
        raise AssertionError("perspective fov was not stored")
    if abs(component.near_plane() - 0.05) > 1.0e-5:
        raise AssertionError("near plane was not stored")
    if abs(component.far_plane() - 500.0) > 1.0e-5:
        raise AssertionError("far plane was not stored")

    projection = component.projection_matrix(16.0 / 9.0)
    view = component.view_matrix()
    view_projection = component.view_projection_matrix(16.0 / 9.0)
    if projection is None or view is None or view_projection is None:
        raise AssertionError("camera matrices were not returned")

    component.set_orthographic(4.0, 0.1, 50.0)
    if component.projection_type != ke.scene.CameraProjectionType.Orthographic:
        raise AssertionError("orthographic projection type was not stored")
    if abs(component.orthographic_size() - 4.0) > 1.0e-5:
        raise AssertionError("orthographic size was not stored")

    if not camera_prim.remove_camera_component():
        raise AssertionError("remove_camera_component returned false")
    if camera_prim.get_camera_component() is not None:
        raise AssertionError("CameraComponent still attached after removal")
    try:
        component.position()
    except RuntimeError:
        pass
    else:
        raise AssertionError("detached CameraComponent remained usable")

    mesh_prim = scene.define_prim("/not_a_camera", ke.scene.PrimType.Xform)
    try:
        mesh_prim.add_camera_component()
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-Camera prim accepted CameraComponent")

    child = scene.define_prim("/cameras/subtree/child", ke.scene.PrimType.Camera)
    child_component = child.add_camera_component()
    if not scene.remove_prim("/cameras/subtree"):
        raise AssertionError("failed to remove camera subtree")
    if child_component.attached:
        raise AssertionError("CameraComponent stayed attached after subtree removal")

    print("PASS: CameraComponent smoke completed")


if __name__ == "__main__":
    main()
