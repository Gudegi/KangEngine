"""Check persistent debug batches, occlusion, and capture on both backends."""

import argparse
from unittest.mock import patch

import numpy as np
import kangengine as ke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    parser.add_argument("--window", action="store_true")
    args = parser.parse_args()
    app = ke.App()
    adapter = None
    try:
        if args.backend == "ovrtx":
            from kangengine.render._ovrtx_scene import OvrtxAppAdapter
            original_init = OvrtxAppAdapter.__init__

            def remember(instance, width, height):
                nonlocal adapter
                original_init(instance, width, height)
                adapter = instance

            with patch.object(OvrtxAppAdapter, "__init__", remember):
                app.initialize(width=320, height=240, hide_ui=True,
                               headless=not args.window, render_backend=args.backend)
        else:
            app.initialize(width=320, height=240, hide_ui=True, headless=not args.window)
        app.set_camera_view(position=[0, 0, 4], target=[0, 0, 0])
        app.set_light_intensity(0)
        app.set_light_ambient([0, 0, 0])
        app.set_background_color([0, 0, 0, 1])
        material = app.create_pbr_material(base_color=[0, 0, 0, 1], roughness=1)
        plane = app.scene.add_mesh("/occluder", ke.geometry.create_plane_data(8, ke.UpAxis.Z), material)
        source = np.asarray(plane.prim.resolve_mesh_data().vertices).copy()
        debug = app.debug.overlay

        def capture():
            app.draw_frame()
            return app.read_rgb_pixels().copy()

        baseline = capture()
        debug.clear("/missing")
        debug.remove("/missing")
        debug.remove_all()
        np.testing.assert_array_equal(capture(), baseline)
        debug.points("/point", [[0, 0, -1]], [1, 0, 0, 1], size=16)
        np.testing.assert_array_equal(capture(), baseline)
        debug.points("/point", [[0, 0, -1]], [1, 0, 0, 1], size=16, overlay=True)
        visible = capture()
        assert (visible[..., 0] > 180).sum() > 120
        debug.set_visible("/point", False)
        np.testing.assert_array_equal(capture(), baseline)
        debug.set_visible("/point", True)
        np.testing.assert_array_equal(capture(), visible)
        debug.points("/point", [[0.6, 0.4, 1]], [0, 1, 0, 1], size=8)
        moved = capture()
        assert (moved[..., 1] > 180).sum() > 20
        assert not (moved[..., 0] > 180).any()
        columns = np.nonzero(moved[..., 1] > 180)[1]
        app.set_camera_view(position=[1, 0, 4], target=[1, 0, 0])
        shifted = capture()
        assert np.nonzero(shifted[..., 1] > 180)[1].mean() < columns.mean() - 20
        app.set_camera_view(position=[0, 0, 8], target=[0, 0, 0])
        farther = capture()
        assert abs((farther[..., 1] > 180).sum() - len(columns)) < 12
        app.set_camera_view(position=[0, 0, 4], target=[0, 0, 0])
        debug.clear_points("/point")
        np.testing.assert_array_equal(capture(), baseline)
        debug.points("/alpha", [[0, 0, 1]], [1, 0, 0, 0.5], size=12)
        assert 180 < capture()[120, 160, 0] < 195
        debug.remove("/alpha")

        # Robot axes, contact normals, center of mass, and reference positions.
        debug.axes("/axes", [-0.5, -0.3, -1], length=0.6, width=3)
        debug.arrows("/contact", [[0, 0, -1]], [[0.7, 0.5, -1]], width=3)
        np.testing.assert_array_equal(capture(), baseline)
        debug.axes("/axes", [-0.5, -0.3, -1], length=0.6, width=3, overlay=True)
        debug.arrows("/contact", [[0, 0, -1]], [[0.7, 0.5, -1]], width=3, overlay=True)
        debug.points("/com", [[0, 0.4, 1]], [1, 1, 0, 1], size=12)
        debug.points("/reference", [[-0.6, 0.5, 1], [0.6, 0.5, 1]], size=7)
        assert np.count_nonzero(capture()) > 100
        debug.remove_all()
        np.testing.assert_array_equal(capture(), baseline)

        debug.lines("/shared", [[-0.5, 0, 1]], [[0.5, 0, 1]], width=3)
        debug.points("/shared", [[0, 0, 1]], size=10)
        shared = capture()
        assert shared.max() > 180
        debug.remove("/sha")
        np.testing.assert_array_equal(capture(), shared)
        debug.clear_lines("/shared")
        assert np.count_nonzero(capture()) < np.count_nonzero(shared)
        debug.remove("/shared")
        np.testing.assert_array_equal(capture(), baseline)

        # Near-plane crossings must clip, and behind-camera points must disappear.
        debug.lines("/clip", [[0, 0, 5]], [[0.8, 0.8, 1]], width=3, overlay=True)
        assert capture().max() > 180
        debug.remove_all()
        debug.points("/behind", [[0, 0, 5]], size=40, overlay=True)
        np.testing.assert_array_equal(capture(), baseline)
        debug.remove_all()
        capture()
        np.testing.assert_array_equal(plane.prim.resolve_mesh_data().vertices, source)
        if adapter:
            assert not adapter.debug_draw._batches
            assert adapter.debug_draw.output_path.endswith("/LdrColor")
            assert not adapter.debug_draw._arrays["primitives"].size
            with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
                capture()
                assert not any(str(call.args[1]).startswith("inputs:") for call in writes.call_args_list)
            debug.points("/idle", [[0, 0, 1]], size=10)
            capture()
            with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
                capture()
                assert not any(str(call.args[1]).startswith("inputs:") for call in writes.call_args_list)
            debug.remove_all()
        print(f"PASS: {args.backend} debug batches, depth, clear, capture, source ownership")
    finally:
        app.close()


if __name__ == "__main__":
    main()
