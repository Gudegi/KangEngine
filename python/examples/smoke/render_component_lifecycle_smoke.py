"""Validate Prim-owned RenderComponent attach/detach semantics."""

import kangengine as ke


def main():
    prim = ke.scene.Prim("mesh", ke.scene.PrimType.Mesh)
    component = prim.add_render_component()

    assert prim.has_render_component()
    assert prim.get_render_component() is component
    assert component.attached
    assert component.owner is prim
    assert "RenderComponent" in repr(component)
    assert "version=" in repr(component)

    initial_version = component.version
    component.double_sided = True
    component.casts_shadow = False
    component.visible = False
    component.transform_source = ke.TransformSource.ExternalBuffer

    assert component.double_sided
    assert not component.casts_shadow
    assert not component.visible
    assert component.transform_source == ke.TransformSource.ExternalBuffer
    assert not prim.is_visible()
    assert component.version > initial_version

    prim_version = component.version
    prim.set_visible(True)
    assert component.visible
    assert component.version > prim_version

    try:
        prim.add_render_component()
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate RenderComponent attach did not fail")

    assert prim.remove_render_component()
    assert not prim.has_render_component()
    assert not component.attached
    assert component.owner is None
    assert "<detached>" in repr(component)
    assert not prim.remove_render_component()

    try:
        _ = component.visible
    except RuntimeError:
        pass
    else:
        raise AssertionError("detached RenderComponent remained usable")

    replacement = prim.add_render_component()
    assert replacement.attached
    assert replacement.owner is prim

    backend = ke.scene.create_backend(ke.scene.BackendType.Native)
    backend.define_prim("/group", ke.scene.PrimType.Xform)
    child = backend.define_prim("/group/mesh", ke.scene.PrimType.Mesh)
    subtree_component = child.add_render_component()
    assert backend.remove_prim("/group")
    assert not subtree_component.attached
    assert subtree_component.owner is None

    print("PASS: RenderComponent attach/detach lifecycle")


if __name__ == "__main__":
    main()
