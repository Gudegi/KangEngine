"""Render a scene authored before App initialization and verify detach lifetimes."""

from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import kangengine as ke


def main():
    scene = ke.scene.create_backend(type=ke.scene.BackendType.NATIVE)
    resources = ke.scene.SceneResourceManager(scene)
    mesh = ke.scene.Prim.create_rectangle_data(1, 1, 1)
    handle = resources.register_mesh("Box", mesh)
    prim = scene.define_prim("/World/Box", ke.scene.PrimType.MESH)
    prim.set_mesh_data(mesh)
    prim.get_mesh_component().resource_handle = handle
    token_id = ke.scene.Token("xformOp:translate").id()
    assert "kangengine.app.application" not in sys.modules

    app = ke.App()
    try:
        app.initialize(width=128, height=128, hide_ui=True, headless=True)
        app.set_camera_view(position=[3, 3, 3], target=[0, 0, 0])
        material = app.create_standard_materials().common
        view = app.scene.add_renderable(prim=prim, material=material)
        component = view.component
        system = app.get_scene_render_system()
        assert system.is_registered(component)
        assert prim.get_mesh_component().resource_handle == handle
        assert resources.mesh(handle) is mesh
        assert ke.scene.Token("xformOp:translate").id() == token_id

        with tempfile.TemporaryDirectory() as directory:
            def capture(name):
                app.update_and_draw()
                path = Path(directory) / f"{name}.png"
                assert app.write_pixels_png(path=str(path), flip_y=True)
                with Image.open(path) as image:
                    return np.asarray(image.convert("RGB")).astype(float)

            before = capture("before")
            prim.set_local_translation(ke.Vec3(1, 0, 0))
            moved = capture("moved")
            assert np.mean(np.abs(before - moved)) > 0.1
            assert scene.remove_prim("/World")
            assert not component.attached and component.owner is None
            assert system.registration_count == 0
            removed = capture("removed")
            assert np.mean(np.abs(moved - removed)) > 0.1

        # Closing the viewer must clear callbacks on a still-live scene.
        surviving = scene.define_prim("/Surviving", ke.scene.PrimType.MESH)
        surviving.set_mesh_data(mesh)
        surviving.get_mesh_component().resource_handle = handle
        view = app.scene.add_renderable(prim=surviving, material=material)
        component = view.component
        app.close()
        assert component.attached
        assert not system.is_registered(component)
        component.visible = False
        assert scene.remove_prim("/Surviving")
        assert not component.attached
    finally:
        app.close()
    print("PASS: scene rendering, transforms, removal, and close")


if __name__ == "__main__":
    main()
