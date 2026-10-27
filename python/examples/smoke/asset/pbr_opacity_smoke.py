"""Verify separate PBR opacity masks override color alpha in a real GL render."""

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke
from kangengine import render


def main():
    app = ke.App()
    app.initialize(width=64, height=64, hide_ui=True, headless=True)
    try:
        app.set_light_intensity(0)
        app.set_light_ambient(ke.Vec3(0, 0, 0))
        camera = app.get_camera()
        camera.set_camera_pos(ke.Vec3(0, 0, 2))
        camera.set_target_pos(ke.Vec3(0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Color alpha is zero; a separate red mask must take precedence.
            for name, rgba in (
                ("color", [0, 0, 0, 0]),
                ("opaque", [255, 0, 0, 0]),
                ("transparent", [0, 255, 255, 255]),
            ):
                Image.fromarray(np.full((3, 3, 4), rgba, dtype=np.uint8)).save(root / f"{name}.png")
            opaque = app.load_texture(path=root / "opaque.png")
            transparent = app.load_texture(path=root / "transparent.png")
            material = app.create_pbr_material(
                base_color_texture=app.load_texture(path=root / "color.png"),
                emissive_color=ke.Vec3(1, 0, 0),
                emissive_strength=1,
                opacity_texture=opaque,
            )
            mesh = app.scene.add_mesh(
                path="/patch",
                mesh_data=ke.geometry.create_plane_data(4, ke.UpAxis.Z),
                material=material,
            )
            for mode in (render.AlphaMode.MASK, render.AlphaMode.BLEND):
                mesh.set_alpha_mode(mode, cutoff=0.5)
                values = []
                for texture in (opaque, transparent, None):
                    material.set_opacity_texture(texture)
                    app.draw_frame()
                    output = root / "frame.png"
                    assert app.write_pixels_png(path=str(output), flip_y=True)
                    pixels = np.asarray(Image.open(output).convert("RGB"))
                    values.append(pixels[30:34, 30:34].mean(axis=(0, 1)))
                assert values[0][0] > max(values[1][0], values[2][0]) + 50, (mode, values)
                np.testing.assert_allclose(values[1], values[2], atol=2)
            print("PASS: MASK/BLEND use red-channel opacity and override color alpha")
    finally:
        app.close()


if __name__ == "__main__":
    main()
