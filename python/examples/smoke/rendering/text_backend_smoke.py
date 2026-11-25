"""Check text layout, depth, updates, and resize on both render backends."""

import argparse
import ctypes
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

import kangengine as ke
from kangengine._core import _ke


def bounds(image, channel=0):
    rows, columns = np.nonzero(image[..., channel] > 100)
    assert len(rows) > 20, "Text is missing"
    return np.array([columns.min(), rows.min(), columns.max(), rows.max()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("native", "ovrtx"), default="native")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    app = ke.App()
    adapter = None
    try:
        if args.renderer == "ovrtx":
            from kangengine.render._ovrtx_scene import OvrtxAppAdapter
            original_init = OvrtxAppAdapter.__init__

            def remember(instance, width, height):
                nonlocal adapter
                original_init(instance, width, height)
                adapter = instance

            with patch.object(OvrtxAppAdapter, "__init__", remember):
                app.initialize(width=320, height=240, hide_ui=True,
                               headless=not args.window, renderer=args.renderer)
        else:
            app.initialize(width=320, height=240, hide_ui=True, headless=not args.window)
        app.set_vsync(False)
        app.set_camera_view(position=[0, 0, 4], target=[0, 0, 0])
        app.set_light_intensity(0)
        app.set_light_ambient([0, 0, 0])
        app.set_background_color([0, 0, 0, 1])
        material = app.create_pbr_material(base_color=[0, 0, 0, 1], roughness=1)
        app.scene.add_mesh("/occluder", ke.geometry.create_plane_data(8, ke.UpAxis.Z), material)
        world, screen = app.debug.world_text, app.debug.screen_text

        def capture():
            app.draw_frame()
            return app.read_rgb_pixels().copy()

        baseline = capture()
        if adapter:
            assert adapter.text_draw._atlas is None
        for text in (world, screen):
            text.set_text("/missing", "ignored")
            text.set_visible("/missing", False)
            text.remove("/missing")
            text.remove_all()
        np.testing.assert_array_equal(capture(), baseline)

        world.set("/world", "Fgp 123", [0, 0, -1], pixel_size=32)
        np.testing.assert_array_equal(capture(), baseline)
        world.set("/world", "Fgp 123", [0, 0, -1], pixel_size=32, depth_test=False)
        visible = capture()
        original_bounds = bounds(visible)
        world.set_visible("/world", False)
        np.testing.assert_array_equal(capture(), baseline)
        world.set_visible("/world", True)
        np.testing.assert_array_equal(capture(), visible)
        world.set("/world", "Fgp 123", [0, 0, 1], pixel_size=32)
        np.testing.assert_array_equal(bounds(capture()), original_bounds)
        world.set_position("/world", [0.5, 0, 1])
        assert bounds(capture())[0] > original_bounds[0] + 20
        app.set_camera_view(position=[0, 0, 8], target=[0, 0, 0])
        far = bounds(capture())
        assert abs((far[2] - far[0]) - (original_bounds[2] - original_bounds[0])) <= 1
        app.set_camera_view(position=[0, 0, 4], target=[0, 0, 0])
        world.set_position("/world", [0, 0, 5])
        np.testing.assert_array_equal(capture(), baseline)
        world.remove("/world")

        # World and screen text share the same asymmetric glyphs and orientation.
        world.set("/reference", "Fgp 123", [0, 0, 1], pixel_size=32, depth_test=False)
        world_image = capture()
        world.remove_all()
        screen.set("/screen", "Fgp 123", [0, 0], pixel_size=32,
                   anchor=ke.render.ScreenAnchor.CENTER)
        np.testing.assert_allclose(capture().astype(int), world_image.astype(int), atol=3)
        screen.set_text("/screen", "F")
        assert bounds(capture())[2] - bounds(capture())[0] < original_bounds[2] - original_bounds[0]
        screen.set_position("/screen", [20, 15])
        moved = bounds(capture())
        screen.set_position("/screen", [0, 0])
        np.testing.assert_array_equal(moved - bounds(capture()), [20, 15, 20, 15])
        screen.set_visible("/screen", False)
        np.testing.assert_array_equal(capture(), baseline)
        screen.remove_all()

        # Default alignment keeps labels inside all nine screen anchors.
        for name in ("TOP_LEFT", "TOP_CENTER", "TOP_RIGHT", "CENTER_LEFT", "CENTER",
                     "CENTER_RIGHT", "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"):
            screen.set("/anchor", "Fgp", [0, 0], pixel_size=24,
                       anchor=getattr(ke.render.ScreenAnchor, name))
            box = bounds(capture())
            if "LEFT" in name:
                assert box[0] < 5
            elif "RIGHT" in name:
                assert box[2] > 310
            else:
                assert abs((box[0] + box[2]) / 2 - 160) < 5
            if "TOP" in name:
                assert box[1] < 20
            elif "BOTTOM" in name:
                assert box[3] > 230
            else:
                assert 95 < box[1] < 120
        screen.remove_all()

        screen.set("/color", "MMMM", [20, 20], pixel_size=40, color=[1, 0, 0, 0.5])
        colored = capture()
        assert 180 <= colored[..., 0].max() <= 195
        assert not colored[..., 1:].any()
        extra = app.scene.add_mesh("/extra", ke.geometry.create_plane_data(1, ke.UpAxis.Z), material)
        np.testing.assert_array_equal(capture(), colored)
        extra.remove()
        np.testing.assert_array_equal(capture(), colored)
        if adapter:
            with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
                screen.set("/color", "MMMM", [20, 20], pixel_size=40, color=[1, 0, 0, 0.5])
                capture()
                assert not any(str(call.args[1]).startswith("inputs:") for call in writes.call_args_list)
            with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
                screen.set_text("/color", "MMM")
                capture()
                assert not any(call.args[1] == "inputs:font" for call in writes.call_args_list)
        screen.set_text("/color", "")
        np.testing.assert_array_equal(capture(), baseline)
        screen.remove_all()
        np.testing.assert_array_equal(capture(), baseline)
        if adapter:
            assert adapter.debug_draw.output_path.endswith("/LdrColor")

        screen.set("/resize", "Fgp 123", [12, 12], pixel_size=32,
                   anchor=ke.render.ScreenAnchor.BOTTOM_RIGHT)
        before = capture()
        if args.window:
            glfw = ctypes.CDLL(_ke.__file__)
            glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
            glfw.glfwSetWindowSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            for width, height in ((480, 300), (320, 240)):
                glfw.glfwSetWindowSize(glfw.glfwGetCurrentContext(), width, height)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    app.process_events()
                    resized = capture()
                    if resized.shape == (height, width, 3):
                        break
                    time.sleep(0.01)
                assert resized.shape == (height, width, 3)
                np.testing.assert_allclose(bounds(resized) - bounds(before),
                                           [width - 320, height - 240] * 2, atol=1)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            np.save(args.output / f"{args.renderer}-text.npy", before)
        print(f"PASS: {args.renderer} text, depth, layout, updates, color, resize={args.window}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
