"""Verify color/data texture identity and sRGB decoding in a real GL render."""

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke


def main():
    app: ke.App = ke.App()
    app.initialize(width=64, height=64, hide_ui=True, headless=True)
    try:
        app.set_light_intensity(0)
        app.set_light_ambient(ke.Vec3(0, 0, 0))
        camera = app.get_camera()
        camera.set_camera_pos(ke.Vec3(0, 0, 2))
        camera.set_target_pos(ke.Vec3(0, 0, 0))
        material = app.create_pbr_material(
            base_color=ke.Vec4(0, 0, 0, 1),
            emissive_color=ke.Vec3(1, 1, 1),
            emissive_strength=1,
        )
        app.scene.add_mesh(
            path="/patch",
            mesh_data=ke.geometry.create_plane_data(4, ke.UpAxis.Z),
            material=material,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gray.png"
            # Odd RGB row size also exercises image upload alignment.
            Image.fromarray(np.full((3, 3, 3), 128, dtype=np.uint8)).save(source)
            color = app.load_texture(source, srgb=True)
            data = app.load_texture(source, srgb=False)
            assert color is not data
            assert app.load_texture(source, srgb=True) is color
            assert app.load_texture(source, flip=False, srgb=True) is not color
            values = []
            for texture in (color, data):
                material.emissive_texture = texture
                app.draw_frame()
                output = root / "frame.png"
                assert app.write_pixels_png(path=str(output), flip_y=True)
                pixels = np.asarray(Image.open(output).convert("RGB"))
                values.append(pixels[30:34, 30:34].mean(axis=(0, 1)))
            assert np.all(values[1] - values[0] > 25), values
            np.testing.assert_allclose(values[0], values[0].mean(), atol=2)
            np.testing.assert_allclose(values[1], values[1].mean(), atol=2)
            print(f"PASS: sRGB={values[0]}, raw={values[1]}; distinct cached resources")
    finally:
        app.close()


if __name__ == "__main__":
    main()
