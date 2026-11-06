"""Verify debug depth, visibility, updates, and removal in a real GL render."""

import numpy as np

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(width=256, height=256, hide_ui=True, headless=True)
    try:
        assert app.debug.overlay is app.debug_overlay
        assert app.debug.geometry is app.scene.debug_geometry
        assert app.debug.world_text is app.world_text
        assert app.debug.screen_text is app.screen_text
        camera = app.get_camera()
        camera.set_camera_pos(ke.Vec3(0, 0, 4))
        camera.set_target_pos(ke.Vec3(0, 0, 0))
        material = app.create_pbr_material(
            base_color=ke.Vec4(0.2, 0.2, 0.2, 1.0), roughness=1.0,
        )
        app.scene.add_mesh(
            path="/occluder",
            mesh_data=ke.geometry.create_plane_data(8, ke.UpAxis.Z),
            material=material,
        )
        app.draw_frame()
        baseline = app.read_rgb_pixels(flip_y=True).copy()

        for kind in ("axes", "arrows"):
            path = f"/debug/{kind}"

            def draw(**options):
                if kind == "axes":
                    app.debug.overlay.axes(
                        path, [-0.4, -0.4, -1], length=0.8, width=3, **options,
                    )
                else:
                    app.debug.overlay.arrows(
                        path, [[-0.4, -0.4, -1]], [[0.4, 0.4, -1]],
                        width=3, **options,
                    )
                app.draw_frame()
                return app.read_rgb_pixels(flip_y=True).copy()

            # Default depth testing hides geometry fully behind the plane.
            np.testing.assert_array_equal(draw(), baseline)
            visible = draw(overlay=True)
            changed = np.any(visible != baseline, axis=2)
            assert changed.sum() > 20, (kind, changed.sum())
            app.debug.overlay.set_visible(path, False)
            app.draw_frame()
            np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)
            app.debug.overlay.set_visible(path, True)
            app.draw_frame()
            np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), visible)
            np.testing.assert_array_equal(draw(overlay=True, hidden=True), baseline)
            np.testing.assert_array_equal(draw(overlay=False), baseline)
            draw(overlay=True)
            app.debug.overlay.clear_lines(path)
            app.draw_frame()
            np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)

        # Lines and points at the same identifier are managed together.
        app.debug.overlay.lines(
            "/shared/item", [[-0.5, 0, -1]], [[0.5, 0, -1]], overlay=True,
        )
        app.debug.overlay.points("/shared/item", [[0, 0, -1]], size=12, overlay=True)
        app.draw_frame()
        shared = app.read_rgb_pixels(flip_y=True).copy()
        assert np.any(shared != baseline)
        app.debug.overlay.set_visible("/shared/item", False)
        app.draw_frame()
        np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)
        app.debug.overlay.set_visible("/shared/item", True)
        app.debug.overlay.remove("/shared")
        app.draw_frame()
        np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), shared)
        app.debug.overlay.remove("/shared/item")
        app.debug.overlay.set_visible("/shared/item", True)
        app.draw_frame()
        np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)
        for path in ("/a", "/b"):
            app.debug.overlay.points(path, [[0, 0, -1]], overlay=True)
            app.debug.overlay.arrows(path, [[0, 0, -1]], [[0.5, 0, -1]], overlay=True)
        app.debug.overlay.remove_all()
        app.draw_frame()
        np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)

        for group, position, options in (
            (app.debug.world_text, ke.Vec3(0, 0, -1), {"depth_test": False}),
            (app.debug.screen_text, ke.Vec2(20, 20), {}),
        ):
            group.set("/text/a", "Visible", position, **options)
            app.draw_frame()
            visible = app.read_rgb_pixels(flip_y=True).copy()
            assert np.any(visible != baseline)
            group.set_visible("/text/a", False)
            group.set_text("/text/a", "Updated")
            app.draw_frame()
            np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)
            group.set_visible("/text/a", True)
            app.draw_frame()
            assert np.any(app.read_rgb_pixels(flip_y=True) != visible)
            group.remove("/text/a")
            group.set_visible("/text/a", True)
            app.draw_frame()
            np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)
            for path in ("/text/a", "/text/b"):
                group.set(path, "Visible", position, **options)
            group.remove_all()
            app.draw_frame()
            np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=True), baseline)
        app.check_error()
        print("PASS: overlay/text depth, visibility, updates, and removal")
    finally:
        app.close()


if __name__ == "__main__":
    main()
