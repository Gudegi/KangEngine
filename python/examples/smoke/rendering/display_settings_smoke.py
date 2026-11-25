"""Check background, exposure, ambient, and shared checker ground updates."""

import argparse
from unittest.mock import patch

import numpy as np
import kangengine as ke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = ke.App()
    adapter = None
    try:
        if args.renderer == "ovrtx":
            from kangengine.render._ovrtx_scene import OvrtxAppAdapter

            adapter = OvrtxAppAdapter(320, 240)
            app.render_backend = "ovrtx"
            app._initialize_scene_renderer(320, 240, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
                                           lambda width, height: adapter)
        else:
            app.initialize(width=320, height=240, hide_ui=True, headless=True)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        app.set_light_intensity(0)
        app.set_light_ambient([0, 0, 0])
        app.set_background_color([0.01, 0.025, 0.08, 1])

        def capture():
            for _ in range(24 if adapter else 1):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        blue_background = capture()
        assert blue_background[10, 10, 2] > blue_background[10, 10, 0] + 10
        app.set_background_color([0.08, 0.025, 0.01, 1])
        red_background = capture()
        assert red_background[10, 10, 0] > red_background[10, 10, 2] + 10
        material = app.create_pbr_material(base_color=[1, 1, 1, 1], roughness=1)
        cube = app.scene.add_mesh("/cube", ke.geometry.create_cube_data(2), material)
        dark = capture()[120, 160]
        app.set_light_ambient([0.1, 0.1, 0.1])
        bright = capture()[120, 160]
        assert bright.mean() > dark.mean() + 15, (dark, bright)
        app.set_exposure(0.25)
        dim = capture()[120, 160]
        assert dim.mean() < bright.mean() - 5, (dim, bright)
        app.set_exposure(1)
        light = app.scene.get_prim_at_path("/lights/default_directional")
        light.set_visible(False)
        assert capture()[120, 160].max() < 5
        light.set_visible(True)
        assert capture()[120, 160].mean() > 15
        light.set_active(False)
        assert capture()[120, 160].max() < 5
        assert light.is_visible()
        light.set_active(True)
        app.set_light_ambient([0.1, 0, 0])
        red = capture()[120, 160]
        assert red[0] > red[2] + 10, red
        app.set_light_ambient([0, 0, 0])
        assert capture()[120, 160].max() < 5
        for invalid in (-1, float("nan"), float("inf")):
            try:
                app.set_exposure(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid exposure accepted")

        # Background color must not replace the dome's illumination.
        dome = ke.scene.DomeLight()
        dome.intensity = 0.1
        environment = app.scene.define_prim("/environment", ke.scene.PrimType.LIGHT)
        environment.set_dome_light(dome)
        illuminated = capture()
        app.set_background_color([0.01, 0.025, 0.08, 1])
        changed_background = capture()
        assert abs(changed_background[110:130, 150:170].mean() -
                   illuminated[110:130, 150:170].mean()) < 5
        environment.set_visible(False)
        cube.remove()

        app.set_camera_view(position=[0, 5, 4], target=[0, 0, 0])
        app.set_light_ambient([0.3, 0.3, 0.3])
        ground_args = dict(checker_a=[0.7, 0.7, 0.7, 1], checker_b=[0.05, 0.05, 0.05, 1],
                           grid_color=[0.7, 0, 0, 1], show_grid=False, use_pbr=True,
                           grid_scale=1, line_width=0.04)
        app.set_ground_style(**ground_args)
        ground = app.scene.add_ground(scale=6)
        source = np.asarray(ground.prim.resolve_mesh_data().vertices).copy()
        plain = capture()
        ground_args["show_grid"] = True
        app.set_ground_style(**ground_args)
        grid = capture()
        assert np.abs(grid - plain).mean() > 1, np.abs(grid - plain).mean()
        ground_args["grid_scale"] = 2
        app.set_ground_style(**ground_args)
        scaled = capture()
        assert np.abs(scaled - grid).mean() > 2
        np.testing.assert_array_equal(ground.prim.resolve_mesh_data().vertices, source)
        if adapter:
            assert len(adapter._textures) == 1
            with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
                capture()
                assert not any(call.args[1] == "texture:data" for call in writes.call_args_list)
            try:
                app.set_tone_map(ke.render.ToneMapMode.ACES_NARKOWICZ, 1)
            except RuntimeError as error:
                assert "Native" in str(error)
            else:
                raise AssertionError("Native tone map was silently substituted")
            try:
                app.set_ground_style(**dict(ground_args, grid_emission=1))
            except ValueError:
                pass
            else:
                raise AssertionError("Unsupported grid emission was ignored")
            capture()
        app.set_exposure(0)
        assert capture().max() < 5
        app.set_exposure(1)
        ground.remove()
        capture()
        if adapter:
            assert not adapter._textures
        print(f"PASS: {args.renderer} background, exposure, ambient visibility, and checker ground")
    finally:
        app.close()


if __name__ == "__main__":
    main()
