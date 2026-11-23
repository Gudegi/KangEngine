"""Check external debug poses, colors, picking, visibility, and removal."""

import numpy as np

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(width=192, height=192, hide_ui=True, headless=True)
    try:
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        material = app.create_color_material(style=ke.material.ColorStyle.UNLIT)
        system = app.get_scene_render_system()
        debug = app.debug.geometry

        def capture():
            app.draw_frame()
            return app.read_rgb_pixels().copy()

        def pick():
            return app.raycast(ke.Vec3(0, 0, 5), ke.Vec3(0, 0, -1))

        empty = capture()
        for kind in ("lines", "arrows", "spheres", "axes"):
            if kind == "spheres":
                view = debug.add_spheres("/debug", [[0, 0, 0]], radii=0.2,
                                         material=material)
            elif kind == "axes":
                view = debug.add_axes("/debug", [0, -0.5, 0], length=1,
                                      radius=0.08, material=material)
            else:
                create = debug.add_lines if kind == "lines" else debug.add_arrows
                view = create("/debug", [[0, -0.5, 0]], [[0, 0.5, 0]],
                              radius=0.08, material=material)
            assert view.component.transform_source == ke.render.TransformSource.EXTERNAL_BUFFER
            assert view.prim.get_mesh_component().resource_handle != ke.scene.InvalidResourceHandle
            initial = capture()
            hit = pick()
            assert hit.hit and hit.prim is None
            assert hit.transform_source == ke.render.TransformSource.EXTERNAL_BUFFER

            # Prim edits cannot move externally owned instance matrices.
            view.prim.set_local_translation(ke.Vec3(20, 0, 0))
            np.testing.assert_array_equal(capture(), initial)
            try:
                view.set_world_translation(ke.Vec3(20, 0, 0))
            except RuntimeError as error:
                assert "external transform buffer" in str(error)
            else:
                raise AssertionError("debug view accepted a SceneGraph transform")

            system.set_instance_colors(view.component, [[1, 0, 0, 1]])
            red = capture()
            system.set_instance_colors(view.component, [[0, 0, 1, 1]])
            assert np.count_nonzero(capture() != red) > 100
            assert pick().hit
            view.set_visible(False)
            np.testing.assert_array_equal(capture(), empty)
            assert not pick().hit
            view.set_visible(True)

            if kind == "spheres":
                view.update_spheres([[1.5, 0, 0]], radii=0.2)
            elif kind == "arrows":
                view.update_arrows([[1.5, -0.5, 0]], [[1.5, 0.5, 0]])
            else:
                view.update_lines([[1.5, -0.5, 0]], [[1.5, 0.5, 0]])
            capture()
            assert not pick().hit
            moved = app.raycast(ke.Vec3(1.5, 0, 5), ke.Vec3(0, 0, -1))
            assert moved.hit and moved.handle == hit.handle
            assert view.remove()
            assert not view.component.attached and system.registration_count == 0
            np.testing.assert_array_equal(capture(), empty)
        print("PASS: external debug geometry ownership and lifecycle")
    finally:
        app.close()


if __name__ == "__main__":
    main()
