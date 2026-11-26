"""Check viewport depth, normals, instance colors, resize, and RGB restoration."""

import ctypes
import time

import numpy as np
import kangengine as ke
from kangengine._core import _ke
from kangengine.utils._example_args import render_example_parser

from physics_inspector_smoke import click_center


def run(renderer):
    app = ke.App()
    try:
        app.initialize(width=640, height=480, hide_ui=True, renderer=renderer)
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        material = app.create_pbr_material()
        box = app.scene.add_mesh("/box", ke.geometry.create_cube_data(1), material)
        backend = app.get_renderer()
        modes = ke.render.RenderViewMode

        def capture():
            for _ in range(5):
                app.draw_frame()
            return app.read_rgb_pixels().copy()

        rgb = capture()
        for _ in range(2):
            backend.set_view_mode(modes.DEPTH, depth_max=10)
            depth = capture()
            np.testing.assert_allclose(depth[240, 320], [115, 115, 115], atol=2)
            np.testing.assert_array_equal(depth[0, 0], [0, 0, 0])
            backend.set_view_mode(modes.DEPTH, depth_max=5)
            np.testing.assert_allclose(capture()[240, 320], [230, 230, 230], atol=2)

            # Inspection includes visible objects even when picking is disabled.
            box.prim.set_pickable(False)
            backend.set_view_mode(modes.NORMALS)
            normals = capture()
            np.testing.assert_allclose(normals[240, 320], [128, 128, 255], atol=2)
            np.testing.assert_array_equal(normals[0, 0], [0, 0, 0])
            box.prim.set_pickable(True)
            capture()
            assert click_center(app, capture).prim.get_path() == "/box"
            app.clear_selection()
            backend.set_view_mode(modes.RGB)
            assert np.abs(capture().astype(float) - rgb).mean() < 2

        box.remove()
        batch = app.scene.add_mesh(
            "/instances", ke.geometry.create_cube_data(.7), material,
            transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
        )
        matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        matrices[:, 3, 0] = [-.7, .7]
        app.get_scene_render_system().update_instances(batch.component, matrices)
        backend.set_view_mode(modes.OBJECT_IDS)
        ids = capture()
        colors, counts = np.unique(ids.reshape(-1, 3), axis=0, return_counts=True)
        assert len(colors[counts > 50]) == 3, "Expected two instance colors and black background"

        # Scene edits must preserve the selected output's AOV connections.
        extra = app.scene.add_mesh("/extra", ke.geometry.create_cube_data(.5), material)
        assert capture()[240, 320].any()
        extra.remove()
        np.testing.assert_array_equal(capture()[240, 320], [0, 0, 0])

        glfw = ctypes.CDLL(_ke.__file__)
        glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
        glfw.glfwSetWindowSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        glfw.glfwSetWindowSize(glfw.glfwGetCurrentContext(), 800, 600)
        # X11 delivers resize events asynchronously.
        for _ in range(60):
            app.process_events()
            resized = capture()
            if resized.shape == (600, 800, 3):
                break
            time.sleep(.01)
        assert resized.shape == (600, 800, 3)
        backend.set_view_mode(modes.DEPTH, depth_max=10)
        assert capture().shape == (600, 800, 3)
        backend.set_view_mode(modes.RGB)
        assert capture().shape == (600, 800, 3)
        print(f"render_output_smoke: {renderer} PASS")
    finally:
        app.close()


if __name__ == "__main__":
    run(render_example_parser(description=__doc__).parse_args().renderer)
