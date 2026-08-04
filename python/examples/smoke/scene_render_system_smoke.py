"""Validate RenderComponent registration and subtree cleanup."""

import gc
import weakref

import kangengine as ke
import torch


def main():
    app = ke.App()
    app.initialize(width=64, height=64, hide_ui=True, headless=True)

    material = app.create_standard_materials().common
    view = app.scene.add_mesh(
        "/group/mesh",
        ke.scene.Prim.create_rectangle_data(1.0, 1.0, 1.0),
        material,
        transform_source=ke.render.TransformSource.ExternalBuffer,
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
        component.transform_source = ke.render.TransformSource.SceneGraph
    except RuntimeError:
        pass
    else:
        raise AssertionError("registered transform source remained mutable")

    view.set_double_sided(True)
    view.set_casts_shadow(False)
    view.set_alpha_mode(ke.render.AlphaMode.Mask, 0.4)
    assert component.double_sided
    assert not component.casts_shadow
    assert component.alpha_mode == ke.render.AlphaMode.Mask
    assert abs(component.alpha_cutoff - 0.4) < 1.0e-6
    view.update_geometry(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    )
    try:
        view.update_geometry(
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("update_geometry accepted mismatched normals")

    debug_lines = app.scene.log_lines(
        "/debug/component_lines",
        material,
        [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
        [[0.5, 0.0, 0.0], [0.0, 0.8, 0.0]],
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
    )
    assert debug_lines.component is not None
    assert render_system.is_registered(debug_lines.component)
    assert not debug_lines.component.casts_shadow
    debug_lines.update_lines(
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.5, 0.0]],
        [[0.0, 0.0, 1.0, 1.0]],
    )

    debug_arrows = app.scene.log_arrows(
        "/debug/component_arrows",
        material,
        [[0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [[1.0, 1.0, 0.0, 1.0]],
    )
    assert render_system.is_registered(debug_arrows.component)
    debug_arrows.update_arrows(
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[1.0, 0.0, 1.0, 1.0]],
    )
    assert debug_lines.remove()
    assert debug_arrows.remove()
    assert render_system.registration_count == 1

    transforms = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    transforms_ref = weakref.ref(transforms)
    view.set_transform_buffer(transforms)
    del transforms
    gc.collect()
    assert transforms_ref() is not None

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
