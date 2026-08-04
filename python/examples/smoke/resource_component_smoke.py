"""Validate ResourceComponent lifecycle without requiring a GL window."""

from __future__ import annotations

import kangengine as ke


def main():
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)

    resource_prim = scene.define_prim(
        "/.Resources/Materials/Gold", ke.scene.PrimType.RESOURCE
    )
    if resource_prim.get_transform_component() is not None:
        raise AssertionError("Resource prim should not own TransformComponent")
    if resource_prim.has_transform_component():
        raise AssertionError("Resource prim reported TransformComponent")
    try:
        resource_prim.compute_world_matrix()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Resource prim transform API should fail explicitly")

    component = resource_prim.add_resource_component()
    if component is None or not component.attached:
        raise AssertionError("add_resource_component did not attach ResourceComponent")
    if component.owner is not resource_prim:
        raise AssertionError("ResourceComponent owner mismatch")
    if "ResourceComponent" not in repr(component):
        raise AssertionError("ResourceComponent repr missing type name")

    version = component.version
    component.type = ke.scene.ResourceType.MATERIAL
    component.display_name = "Gold"
    component.uri = "material://presets/gold"
    if component.type != ke.scene.ResourceType.MATERIAL:
        raise AssertionError("ResourceType was not stored")
    if component.display_name != "Gold":
        raise AssertionError("display_name was not stored")
    if component.uri != "material://presets/gold":
        raise AssertionError("uri was not stored")
    if component.version <= version:
        raise AssertionError("ResourceComponent version did not advance")

    mesh_prim = scene.define_prim("/World/not_resource", ke.scene.PrimType.XFORM)
    try:
        mesh_prim.add_resource_component()
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-Resource prim accepted ResourceComponent")

    if not resource_prim.remove_resource_component():
        raise AssertionError("remove_resource_component returned false")
    if resource_prim.get_resource_component() is not None:
        raise AssertionError("ResourceComponent still attached after removal")
    try:
        component.uri
    except RuntimeError:
        pass
    else:
        raise AssertionError("detached ResourceComponent remained usable")

    child = scene.define_prim("/.Resources/Textures/Wood", ke.scene.PrimType.RESOURCE)
    child_component = child.add_resource_component()
    if not scene.remove_prim("/.Resources/Textures"):
        raise AssertionError("failed to remove resource subtree")
    if child_component.attached:
        raise AssertionError("ResourceComponent stayed attached after subtree removal")
    print("PASS: ResourceComponent smoke completed")


if __name__ == "__main__":
    main()
