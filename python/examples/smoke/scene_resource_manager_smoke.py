"""Validate C++ SceneResourceManager handle/Resource prim mirroring."""

from __future__ import annotations

import kangengine as ke


def main():
    backend = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    registry = ke.scene.SceneResourceManager(backend)

    mesh = ke.scene.Prim.create_plane_data(1.0)
    handle = registry.register_mesh("Unit Plane", mesh, "memory://unit-plane")
    if handle == 0:
        raise AssertionError("SceneResourceManager returned an invalid-looking handle")
    if len(registry) != 1:
        raise AssertionError("SceneResourceManager size mismatch")
    if registry.mesh(handle) is not mesh:
        raise AssertionError("mesh lookup failed")

    prim = registry.resource_prim(handle)
    if prim is None:
        raise AssertionError("SceneResourceManager did not mirror a Resource prim")
    if prim.get_type() != ke.scene.PrimType.RESOURCE:
        raise AssertionError("mirrored prim is not a Resource prim")
    if not prim.get_path().startswith("/.Resources/Meshes/"):
        raise AssertionError("mirrored Resource prim path mismatch")
    if prim.get_transform_component() is not None:
        raise AssertionError("mirrored Resource prim should not own TransformComponent")
    if prim.resolve_manipulation_target() is not None:
        raise AssertionError("mirrored Resource prim should not be manipulable")
    if prim.get_mesh_data() is not None:
        raise AssertionError("Resource prim should be metadata-only")
    component = prim.get_resource_component()
    if component is None:
        raise AssertionError("mirrored Resource prim has no ResourceComponent")
    if component.handle != handle:
        raise AssertionError("ResourceComponent handle mismatch")
    if component.type != ke.scene.ResourceType.MESH:
        raise AssertionError("ResourceComponent type mismatch")
    if component.display_name != "Unit Plane":
        raise AssertionError("ResourceComponent display name mismatch")
    if component.uri != "memory://unit-plane":
        raise AssertionError("ResourceComponent uri mismatch")
    prim_path = prim.get_path()

    assert registry.register_mesh("Alias", mesh, "memory://alias") == handle
    assert len(registry) == 1
    assert component.display_name == "Unit Plane"
    assert component.uri == "memory://unit-plane"

    registry.clear()
    if len(registry) != 0:
        raise AssertionError("SceneResourceManager clear failed")
    if backend.get_root_prim().get_prim_at_path(prim_path) is not None:
        raise AssertionError("SceneResourceManager clear left a stale Resource prim")
    replacement = registry.register_mesh("Reloaded", mesh)
    assert registry.mesh(replacement) is mesh
    assert len(registry) == 1

    print("PASS: SceneResourceManager smoke completed")


if __name__ == "__main__":
    main()
