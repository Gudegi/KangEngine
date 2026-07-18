"""Validate MeshComponent direct mesh and source-path resolution."""

from __future__ import annotations

import kangengine as ke


def main():
    scene = ke.scene.create_backend(ke.scene.BackendType.Native)

    mesh = ke.scene.Prim.create_plane_data(1.0, ke.UpAxis.Z)
    source = scene.define_prim("/Assets/Plane", ke.scene.PrimType.Mesh)
    source.set_mesh_data(mesh)

    component = source.get_mesh_component()
    if component is None or not component.attached:
        raise AssertionError("set_mesh_data did not create MeshComponent")
    if component.owner is not source:
        raise AssertionError("MeshComponent owner mismatch")
    if "MeshComponent" not in repr(component):
        raise AssertionError("MeshComponent repr missing type name")
    if source.get_mesh_data() is not mesh:
        raise AssertionError("Prim.get_mesh_data did not return direct mesh")
    if component.mesh_data is not mesh:
        raise AssertionError("MeshComponent.mesh_data did not return direct mesh")
    if source.resolve_mesh_data() is not mesh:
        raise AssertionError("Prim.resolve_mesh_data did not return direct mesh")

    version = component.version
    component.resource_handle = 42
    if component.resource_handle != 42:
        raise AssertionError("resource_handle was not stored")
    if component.version != version + 1:
        raise AssertionError("resource_handle change should advance version once")

    inst = scene.define_prim("/World/PlaneInstance", ke.scene.PrimType.MeshInstance)
    inst.set_mesh_source_path("/Assets/Plane")
    inst_component = inst.get_mesh_component()
    if inst_component is None:
        raise AssertionError("set_mesh_source_path did not create MeshComponent")
    if inst_component.mesh_source_path != "/Assets/Plane":
        raise AssertionError("mesh_source_path was not stored")
    if inst.resolve_mesh_data() is not mesh:
        raise AssertionError("MeshInstance did not resolve source mesh")

    xform = scene.define_prim("/World/Group", ke.scene.PrimType.Xform)
    try:
        xform.add_mesh_component()
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-mesh prim accepted MeshComponent")

    registry = ke.scene.SceneResourceManager(scene)
    registered_mesh = ke.scene.Prim.create_plane_data(2.0, ke.UpAxis.Z)
    handle = registry.register_mesh(
        "RegisteredPlane", registered_mesh, "generated://registered_plane"
    )
    resource_prim = registry.resource_prim(handle)
    if resource_prim is None:
        raise AssertionError("registry did not create resource prim")
    if registry.mesh(handle) is not registered_mesh:
        raise AssertionError("SceneResourceManager did not retain mesh payload")
    if resource_prim.get_mesh_data() is not None:
        raise AssertionError("resource mirror prim should not own mesh data")

    by_identity = scene.define_prim("/World/ByIdentity", ke.scene.PrimType.Mesh)
    by_identity.set_mesh_data(registered_mesh)
    by_identity_component = by_identity.get_mesh_component()
    by_identity_component.resource_handle = handle
    if by_identity.resolve_mesh_data() is not registered_mesh:
        raise AssertionError("direct mesh cache should remain the render path")

    if not source.remove_mesh_component():
        raise AssertionError("remove_mesh_component returned false")
    if source.get_mesh_component() is not None:
        raise AssertionError("MeshComponent still attached after removal")
    try:
        component.mesh_data
    except RuntimeError:
        pass
    else:
        raise AssertionError("detached MeshComponent remained usable")

    print("PASS: MeshComponent smoke completed")


if __name__ == "__main__":
    main()
