"""Check material texture sharing and explicit Native batch overrides."""

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke
from kangengine._core import _ke


def main():
    app = ke.App()
    app.initialize(width=256, height=192, headless=True, hide_ui=True)
    try:
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        mesh = ke.geometry.create_cube_data(0.8)
        camera = app.scene.define_prim("/camera", ke.scene.PrimType.CAMERA)
        camera.set_local_translation(ke.Vec3(0, 0, 5))
        camera.add_camera_component().set_perspective(45, 0.1, 100)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{name}.png" for name in ("red", "blue")]
            for path, color in zip(paths, ((255, 0, 0), (0, 0, 255))):
                Image.new("RGB", (4, 4), color).save(path)
            red, blue = [app.load_texture(path) for path in paths]

            def capture():
                app.draw_frame()
                return app.read_rgb_pixels().copy()

            for factory, slot in (
                (app.create_color_material, "base_color_texture"),
                (app.create_phong_material, "diffuse_map"),
                (app.create_pbr_material, "base_color_texture"),
            ):
                material = factory()
                material.unlit = True
                setattr(material, slot, red)
                left = app.scene.add_mesh("/left", mesh, material)
                right = app.scene.add_mesh("/right", mesh, material)
                left.set_local_translation(ke.Vec3(-1, 0, 0))
                right.set_local_translation(ke.Vec3(1, 0, 0))
                initial = capture()
                assert np.count_nonzero(initial[:, :, 0] > initial[:, :, 2]) > 100
                setattr(material, slot, blue)
                changed = capture()
                for half in (slice(None, 128), slice(128, None)):
                    assert np.count_nonzero(changed[:, half] != initial[:, half]) > 100

                separate = app._remember_material(material.clone())
                assert separate.unlit and getattr(separate, slot) is blue
                setattr(separate, slot, red)
                assert getattr(material, slot) is blue
                left.set_material(separate)
                split = capture()
                np.testing.assert_array_equal(split[:, :128], initial[:, :128])
                np.testing.assert_array_equal(split[:, 128:], changed[:, 128:])
                if isinstance(material, ke.material.ColorMaterial):
                    frame = _ke._RenderSceneDataCollector().collect(
                        app.scene.native, app.resources, "/camera",
                    )
                    assert all("base_color_texture" in item["textures"]
                               for item in frame["objects"])
                    left.set_material(material)
                    assert app.get_scene_render_system().shares_batch(left.component, right.component)
                    left.set_batch_texture(red)
                    np.testing.assert_array_equal(capture(), initial)
                    # The old spelling keeps its batch-wide behavior.
                    left.set_texture(blue)
                    np.testing.assert_array_equal(capture(), changed)
                    assert material.base_color_texture is blue
                left.remove()
                right.remove()

            material = app.create_color_material(style=ke.material.ColorStyle.TEXTURED)
            surface = app.scene.add_mesh("/normal", mesh, material)
            images = []
            for name, rgb in (("flat", (128, 128, 255)), ("tilted", (255, 128, 128))):
                path = Path(directory) / f"{name}.png"
                Image.new("RGB", (4, 4), rgb).save(path)
                material.normal_texture = app.load_texture(path, srgb=False)
                images.append(capture())
            assert np.count_nonzero(images[0] != images[1]) > 100
            surface.remove()
        print("PASS: shared material textures, isolated materials, sources, and batch overrides")
    finally:
        app.close()


if __name__ == "__main__":
    main()
