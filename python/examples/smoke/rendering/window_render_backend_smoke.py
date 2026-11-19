"""Check window presentation, camera changes, resize, and explicit readback."""

import argparse
import ctypes
import time

import numpy as np
from OpenGL import GL

import kangengine as ke
from kangengine._core import _ke
from app_backend_smoke import build_scene


class WindowApp(ke.App):
    def post_render(self):
        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        width, height = int(viewport[2]), int(viewport[3])
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_BACK)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
        self.window_pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)[::-1].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    maps = []
    if args.backend == "ovrtx":
        import ovrtx

        original_map = ovrtx.RenderVarOutput.map

        def record_map(self, device=ovrtx.Device.CPU, sync_stream=None):
            maps.append(device)
            return original_map(self, device=device, sync_stream=sync_stream)

        ovrtx.RenderVarOutput.map = record_map

    # Use the engine's GLFW instance; the Python glfw package may load another.
    glfw = ctypes.CDLL(_ke.__file__)
    glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
    glfw.glfwSetWindowSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    app = WindowApp()
    try:
        app.initialize(width=257, height=193, hide_ui=True, render_backend=args.backend)
        app.set_vsync(enabled=False)
        build_scene(app)
        app.draw_frame()
        first = app.window_pixels.copy()
        if args.backend == "ovrtx":
            assert maps and all(device == ovrtx.Device.CUDA_ARRAY for device in maps), maps
        captured = app.read_rgb_pixels()
        np.testing.assert_allclose(first.astype(int), captured.astype(int), atol=1)
        assert first[..., 1].max() > 100
        if args.backend == "ovrtx":
            assert maps.count(ovrtx.Device.CPU) == 1
            app.read_rgb_pixels()
            assert maps.count(ovrtx.Device.CPU) == 1, "readback was not cached"

        app.set_camera_view(position=[1.5, 0.5, 4], target=[0, 0, 0])
        app.draw_frame()
        assert not np.array_equal(first, app.window_pixels)
        glfw.glfwSetWindowSize(glfw.glfwGetCurrentContext(), 311, 217)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            app.process_events()
            app.draw_frame()
            if app.window_pixels.shape == (217, 311, 3):
                break
            time.sleep(0.01)
        resized = app.read_rgb_pixels()
        assert resized.shape == (217, 311, 3), resized.shape
        np.testing.assert_allclose(app.window_pixels.astype(int), resized.astype(int), atol=1)

        app.request_close()
        assert app.should_close()
        app.close()

        app = WindowApp()
        app.initialize(width=800, height=600, render_backend=args.backend)
        app.set_ui_layout_mode(mode=ke.UILayoutMode.EDITOR)
        build_scene(app)
        map_count = len(maps)
        for _ in range(3):
            app.process_events()
            app.draw_frame()
        assert app.window_pixels.max() > 100
        if args.backend == "ovrtx":
            assert all(device == ovrtx.Device.CUDA_ARRAY for device in maps[map_count:])
    finally:
        app.close()
        app.close()
        if args.backend == "ovrtx":
            ovrtx.RenderVarOutput.map = original_map
    print(f"PASS: {args.backend} window, orientation, camera, resize, readback, and close")


if __name__ == "__main__":
    main()
