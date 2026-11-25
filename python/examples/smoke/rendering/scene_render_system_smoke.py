"""Validate RenderComponent registration and subtree cleanup."""

import gc

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(width=64, height=64, hide_ui=True, headless=True)

    material = app.create_standard_materials().common
    mesh_data = ke.scene.Prim.create_rectangle_data(1.0, 1.0, 1.0)
    view = app.scene.add_mesh(
        "/group/mesh",
        mesh_data,
        material,
        transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
    )
    component = view.prim.get_render_component()
    mesh_component = view.prim.get_mesh_component()
    render_system = app.get_scene_render_system()

    assert component is not None
    assert component.attached
    assert mesh_component is not None
    assert mesh_component.resource_handle != ke.scene.InvalidResourceHandle
    resource_prim = app.resources.resource_prim(mesh_component.resource_handle)
    assert resource_prim is not None
    assert resource_prim.get_path().startswith("/.Resources/Meshes/")
    assert resource_prim.get_resource_component() is not None
    assert view.prim.get_mesh_data() is not None
    assert view.prim.resolve_mesh_data() is not None
    assert "RenderComponent" in repr(component)
    assert "/group/mesh" in repr(component)
    assert render_system.registration_count == 1
    assert render_system.is_registered(component)
    assert not hasattr(view, "_handles")
    assert not hasattr(view, "_external_buffers")

    direct_version = component.version
    component.double_sided = True
    component.casts_shadow = False
    assert component.version > direct_version

    try:
        component.transform_source = ke.render.TransformSource.SCENE_GRAPH
    except RuntimeError:
        pass
    else:
        raise AssertionError("registered transform source remained mutable")

    view.set_double_sided(True)
    view.set_casts_shadow(False)
    view.set_alpha_mode(ke.render.AlphaMode.MASK, 0.4)
    assert component.double_sided
    assert not component.casts_shadow
    assert component.alpha_mode == ke.render.AlphaMode.MASK
    assert abs(component.alpha_cutoff - 0.4) < 1.0e-6
    # create_rectangle_data() creates a box with 24 face-split vertices so
    # each face can retain its hard normal and UVs. Dynamic updates must keep
    # that uploaded topology and vertex count.
    deformed_positions = [
        [position.x * 0.9, position.y, position.z]
        for position in mesh_data.vertices
    ]
    view.update_geometry(deformed_positions, mesh_data.normals)
    try:
        view.update_geometry(
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("update_geometry accepted mismatched normals")

    temporary = app.scene.add_mesh(
        "/temporary",
        ke.scene.Prim.create_rectangle_data(0.5, 0.5, 0.5),
        material,
    )
    temporary_component = temporary.component
    assert render_system.registration_count == 2
    del temporary
    gc.collect()
    assert temporary_component.attached
    assert render_system.is_registered(temporary_component)
    assert app.remove_prim("/temporary")
    assert render_system.registration_count == 1

    # Direct backend removal must still release the private renderer handle.
    assert app.get_native_scene().remove_prim("/group")
    assert not component.attached
    assert component.owner is None
    assert render_system.registration_count == 0
    assert not render_system.is_registered(component)

    try:
        view.set_casts_shadow(True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("detached render view remained usable")

    print("PASS: SceneRenderSystem registration lifecycle")


if __name__ == "__main__":
    main()
