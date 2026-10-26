"""Verify emitted-ray directions, hierarchy evaluation and directional shadows."""

import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(width=160, height=160, hide_ui=True, headless=True, up_axis=ke.UpAxis.Z)
    try:
        app.scene.get_prim_at_path("/lights/default_directional").set_visible(False)
        app.set_tone_map(mode=ke.render.ToneMapMode.ACES_NARKOWICZ, exposure=1.0)
        camera = app.get_camera()
        camera.set_camera_pos(ke.Vec3(0, -7, 6))
        camera.set_target_pos(ke.Vec3(0, 0, 0))
        camera.set_near_plane(0.1)
        camera.set_far_plane(20)
        material = app.create_pbr_material(
            base_color=ke.Vec4(.5, .5, .5, 1), metallic=0, roughness=1,
        )
        app.scene.add_mesh(
            path="/receiver", mesh_data=ke.geometry.create_plane_data(16, ke.UpAxis.Z),
            material=material,
        )
        parent = app.scene.define_prim("/lights/rig", ke.scene.PrimType.XFORM)
        parent.set_local_translation(ke.Vec3(1, 0, 0))
        parent.set_local_rotation_axis_angle(ke.Vec3(0, 0, 1), math.pi / 2)
        parent.set_local_scale(ke.Vec3(2, 3, 1))
        subject = app.scene.define_prim("/lights/rig/light", ke.scene.PrimType.LIGHT)
        reference = app.scene.define_prim("/lights/reference", ke.scene.PrimType.LIGHT)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"

            def capture():
                app.draw_frame()
                assert app.write_pixels_png(path=str(path), flip_y=True)
                return np.asarray(Image.open(path).convert("RGB"), dtype=float)

            # The equivalent world pose is computed explicitly, independent of
            # the renderer's light transform helper. Range must not scale.
            for kind, value_type in (
                ("directional", ke.scene.DirectionalLight),
                ("point", ke.scene.PointLight),
                ("spot", ke.scene.SpotLight),
            ):
                local = value_type()
                world = value_type()
                local.intensity = world.intensity = 5 if kind == "directional" else 30
                if kind == "directional":
                    local.ambient = world.ambient = ke.Vec3(0, 0, 0)
                else:
                    local.position = ke.Vec3(.5, 0, 3)
                    world.position = ke.Vec3(1, 1, 3)
                    local.range = world.range = 15
                if kind != "point":
                    local.direction = ke.Vec3(-1, 0, -1)
                    world.direction = ke.Vec3(0, -2, -1)
                getattr(subject, f"set_{kind}_light")(local)
                getattr(reference, f"set_{kind}_light")(world)
                subject.set_visible(True)
                reference.set_visible(False)
                transformed = capture()
                subject.set_visible(False)
                reference.set_visible(True)
                expected = capture()
                np.testing.assert_allclose(transformed, expected, atol=1)
                reference.set_visible(False)
                dark = capture()
                assert np.mean(expected - dark) > 1, f"{kind} did not illuminate the receiver"
                print(f"PASS: {kind} transformed and world-authored images agree")

            sun = ke.scene.DirectionalLight()
            sun.ambient = ke.Vec3(0, 0, 0)
            sun.direction = ke.Vec3(0, 0, -1)
            reference.set_directional_light(sun)
            reference.set_visible(True)
            downward = capture()
            sun.direction = ke.Vec3(0, 0, 1)
            reference.set_directional_light(sun)
            upward = capture()
            assert np.mean(downward - upward) > 10, "Directional rays have the wrong sign"

            # A caster above the floor must shadow it when rays travel downward.
            sun.direction = ke.Vec3(.6, 0, -1)
            reference.set_directional_light(sun)
            caster = app.scene.add_mesh(
                path="/caster", mesh_data=ke.geometry.create_cube_data(scale=1.0),
                material=material,
            )
            caster.set_local_translation(ke.Vec3(0, 0, 1))
            caster.set_casts_shadow(enabled=False)
            unshadowed = capture()
            caster.set_casts_shadow(enabled=True)
            shadowed = capture()
            darker = (unshadowed - shadowed).mean(axis=2) > 10
            assert np.count_nonzero(darker) > 20, "Directional shadow does not reach the floor"
            print("PASS: downward directional illumination and caster shadow")
    finally:
        app.close()


if __name__ == "__main__":
    main()
