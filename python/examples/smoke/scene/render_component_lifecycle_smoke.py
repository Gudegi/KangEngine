"""Validate Prim-owned RenderComponent attach/detach semantics."""

import numpy as np

import kangengine as ke


def main():
    prim = ke.scene.Prim("mesh", ke.scene.PrimType.MESH)
    component = prim.add_render_component()

    assert prim.has_render_component()
    assert prim.get_render_component() is component
    assert component.attached
    assert component.owner is prim
    assert "RenderComponent" in repr(component)
    assert "version=" in repr(component)

    assert component.display_color is None
    assert prim.get_display_color() is None
    assert prim.get_display_color_alpha() is None
    prim.set_display_color(ke.Vec3(0.2, 0.4, 0.6))
    np.testing.assert_allclose(component.display_color, [0.2, 0.4, 0.6, 1])
    component.display_color = ke.Vec4(0.8, 0.6, 0.4, 0.3)
    np.testing.assert_allclose(prim.get_display_color(), [0.8, 0.6, 0.4])
    prim.set_display_color(ke.Vec3(0.1, 0.2, 0.3))
    np.testing.assert_allclose(prim.get_display_color_alpha(), [0.1, 0.2, 0.3, 0.3])
    color_version = component.version
    prim.set_display_color(ke.Vec3(0.1, 0.2, 0.3))
    assert component.version == color_version
    assert not prim.has_attribute("primvars:displaycolor")
    assert not prim.has_attribute("primvars:displaycolorAlpha")

    initial_version = component.version
    component.double_sided = True
    component.casts_shadow = False
    component.visible = False
    component.transform_source = ke.render.TransformSource.EXTERNAL_BUFFER

    assert component.double_sided
    assert not component.casts_shadow
    assert not component.visible
    assert component.transform_source == ke.render.TransformSource.EXTERNAL_BUFFER
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
    assert replacement.display_color is None

    colored = ke.scene.Prim("colored", ke.scene.PrimType.MESH)
    colored.set_display_color_alpha(ke.Vec4(0.2, 0.4, 0.6, 0.5))
    assert colored.has_render_component()
    np.testing.assert_allclose(colored.get_render_component().display_color,
                               [0.2, 0.4, 0.6, 0.5])

    backend = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    backend.define_prim("/group", ke.scene.PrimType.XFORM)
    child = backend.define_prim("/group/mesh", ke.scene.PrimType.MESH)
    subtree_component = child.add_render_component()
    assert backend.remove_prim("/group")
    assert not subtree_component.attached
    assert subtree_component.owner is None

    print("PASS: RenderComponent attach/detach lifecycle")


if __name__ == "__main__":
    main()
