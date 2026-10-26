"""Verify rectangle area, orientation, exposure and lifecycle with real pixels."""

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
        camera.set_camera_pos(ke.Vec3(0, 0, 3))
        camera.set_target_pos(ke.Vec3(0, 0, 0))
        material = app.create_pbr_material(
            base_color=ke.Vec4(.5, .5, .5, 1), metallic=0, roughness=1,
        )
        app.scene.add_mesh(
            path="/receiver",
            mesh_data=ke.geometry.create_plane_data(8, ke.UpAxis.Z),
            material=material,
        )
        parent = app.scene.define_prim("/lights/rig", ke.scene.PrimType.XFORM)
        panel = app.scene.define_prim("/lights/rig/panel", ke.scene.PrimType.LIGHT)
        panel.set_local_translation(ke.Vec3(0, 0, 10))
        light = ke.scene.RectLight()
        light.intensity = 100
        light.normalize = True
        panel.set_rect_light(light)

        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.png"

            def sample():
                app.draw_frame()
                assert app.write_pixels_png(path=str(frame), flip_y=True)
                rgb = np.asarray(Image.open(frame).convert("RGB"))[30:34, 30:34] / 255.0
                return np.power(rgb, 2.2).mean(axis=(0, 1))

            baseline = sample()
            assert baseline.min() > .05, baseline
            light.width = light.height = 4
            panel.set_rect_light(light)
            normalized_large = sample()
            ratio = normalized_large.mean() / baseline.mean()
            assert .85 < ratio < 1.1, ratio

            light.width = light.height = 1
            panel.set_rect_light(light)
            panel.set_local_quat_wxyz((0, 1, 0, 0))
            assert sample().max() < .001, "Rectangle back side emitted light"
            panel.set_local_quat_wxyz((1, 0, 0, 0))

            light.exposure = 1
            panel.set_rect_light(light)
            doubled = sample()
            np.testing.assert_allclose(doubled / baseline, 2, rtol=.08)
            light.exposure = 0
            light.enable_color_temperature = True
            light.color_temperature = 4200
            panel.set_rect_light(light)
            warm = sample()
            assert warm[0] > warm[1] > warm[2], warm
            light.enable_color_temperature = False
            panel.set_rect_light(light)

            parent.set_local_scale(ke.Vec3(2, 2, 1))
            scaled = sample()
            np.testing.assert_allclose(scaled, baseline, rtol=.1)
            assert panel.get_rect_light().width == 1
            parent.set_local_scale(ke.Vec3(1, 1, 1))
            light.normalize = False
            light.intensity = 5
            panel.set_rect_light(light)
            small = sample()
            light.width = light.height = 2
            panel.set_rect_light(light)
            large = sample()
            assert large.mean() > small.mean() * 3, (small, large)

            parent.set_active(False)
            assert sample().max() < .001
            parent.set_active(True)
            parent.set_visible(False)
            assert sample().max() < .001
            parent.set_visible(True)
            assert sample().mean() > .01
            app.scene.native.remove_prim("/lights/rig")
            assert sample().max() < .001, "Deleted rectangle left stale GPU lighting"

            sky_prim = app.scene.define_prim("/lights/sky", ke.scene.PrimType.LIGHT)
            sky = ke.scene.DomeLight()
            sky.intensity = .2
            sky_prim.set_dome_light(sky)
            environment = sample()
            np.testing.assert_allclose(environment, .1, atol=.008)
            sky_prim.set_active(False)
            assert sample().max() < .001
            app.scene.native.remove_prim("/lights/sky")
            assert sample().max() < .001
            print(f"PASS: rect normalized area ratio={ratio:.3f}, exposure x2, warm color, transform, visibility, deletion, dome")
    finally:
        app.close()


if __name__ == "__main__":
    main()
