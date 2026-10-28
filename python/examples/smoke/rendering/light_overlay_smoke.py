"""Verify selected light guides remain visible through scene geometry."""

from pathlib import Path
import math
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(width=480, height=360, hide_ui=False, headless=True, up_axis=ke.UpAxis.Z)
    try:
        app.set_ui_layout_mode(mode=ke.UILayoutMode.OVERLAY)
        app.set_interaction_mode(mode=ke.InteractionMode.EDIT)
        camera = app.get_camera()
        camera.set_camera_pos(ke.Vec3(10, -15, 10))
        camera.set_target_pos(ke.Vec3(0, 0, 0))
        app.scene.get_prim_at_path("/lights/default_directional").set_visible(False)
        material = app.create_pbr_material(
            base_color=ke.Vec4(.1, .1, .1, 1), metallic=0, roughness=1,
        )
        # All light guides fit inside this opaque box and would fail depth testing.
        app.scene.add_mesh(
            path="/occluder", mesh_data=ke.geometry.create_cube_data(scale=8),
            material=material,
        )
        light_prim = app.scene.define_prim("/light", ke.scene.PrimType.LIGHT)
        parent = app.scene.define_prim("/rig", ke.scene.PrimType.XFORM)
        parent.set_local_translation(ke.Vec3(.4, .2, .1))
        parent.set_local_rotation_axis_angle(ke.Vec3(0, 0, 1), math.pi / 3)
        parent.set_local_scale(ke.Vec3(-1.3, .7, 1.1))
        child = app.scene.define_prim("/rig/light", ke.scene.PrimType.LIGHT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"

            def capture():
                app.draw_frame()
                assert app.write_pixels_png(path=str(path), flip_y=True)
                return np.asarray(Image.open(path).convert("RGB"), dtype=int)

            for kind, value_type in (
                ("directional", ke.scene.DirectionalLight),
                ("point", ke.scene.PointLight),
                ("spot", ke.scene.SpotLight),
                ("rect", ke.scene.RectLight),
                ("dome", ke.scene.DomeLight),
            ):
                light = value_type()
                light.intensity = 0
                if kind == "directional":
                    light.ambient = ke.Vec3(0, 0, 0)
                    light.direction = ke.Vec3(0, 0, -1)
                if kind in ("point", "spot"):
                    light.range = 2
                getattr(light_prim, f"set_{kind}_light")(light)
                app.clear_selection()
                baseline = capture()
                app.select_prim(prim=light_prim)
                selected = capture()
                changed = (selected - baseline).max(axis=2) > 30
                yellow = (selected[:, :, 0] > 100) & (selected[:, :, 1] > 80)
                yellow &= selected[:, :, 1] > selected[:, :, 2] + 25
                assert np.count_nonzero(changed & yellow) > 20, f"{kind} guide is invisible"
                app.clear_selection()
                np.testing.assert_allclose(capture(), baseline, atol=1)
                print(f"PASS: {kind} selection guide through geometry and deselection")

                # A mirrored, nonuniformly scaled hierarchy must match the same
                # world matrix authored directly, including Rect's emitting side.
                getattr(child, f"set_{kind}_light")(light)
                child.set_local_translation(ke.Vec3(.3, -.2, .2))
                child.set_local_rotation_axis_angle(ke.Vec3(1, 0, 0), .4)
                light_prim.set_local_matrix(child.compute_world_matrix())
                app.select_prim(prim=child)
                hierarchical = capture()
                app.select_prim(prim=light_prim)
                np.testing.assert_allclose(capture(), hierarchical, atol=1)

                app.select_prim(prim=child)
                parent.set_visible(False)
                np.testing.assert_allclose(capture(), baseline, atol=1)
                parent.set_visible(True)
                parent.set_active(False)
                np.testing.assert_allclose(capture(), baseline, atol=1)
                parent.set_active(True)
                app.select_prim(prim=parent)
                np.testing.assert_allclose(capture(), baseline, atol=1)
                light_prim.set_local_matrix(ke.Mat4(1.0))
                print(f"PASS: {kind} hierarchy, hidden/inactive ancestors and non-light selection")

            # Inspector edits the selected prim even when transforms target a group.
            child.set_rect_light(ke.scene.RectLight())
            child.set_local_matrix(ke.Mat4(1.0))
            app.select_prim(prim=child)
            ungrouped = capture()
            parent.set_manipulation_policy(ke.scene.ManipulationPolicy.ROOT)
            np.testing.assert_allclose(capture(), ungrouped, atol=1)
            previous = ungrouped
            for field in ("width", "height"):
                light = child.get_light_component().rect_light()
                setattr(light, field, 3.0)
                child.set_rect_light(light)
                resized = capture()
                changed = np.abs(resized - previous).max(axis=2) > 30
                assert np.count_nonzero(changed) > 100, f"Rect {field} edit did not update guide"
                previous = resized
            print("PASS: grouped Rect selection follows inspector width/height edits")
    finally:
        app.close()


if __name__ == "__main__":
    main()
