"""Check window presentation, camera changes, resize, and explicit readback."""

import argparse
import ctypes
from pathlib import Path
import tempfile
import time

import numpy as np
from OpenGL import GL
from PIL import Image

import kangengine as ke
from kangengine._core import _ke
from app_backend_smoke import build_scene


class WindowApp(ke.App):
    viewport_size = None
    split_viewport = False
    show_resize_panel = False

    def pre_render(self):
        if self.hide_ui or self.get_ui_layout_mode() != ke.UILayoutMode.EDITOR:
            return
        if self.split_viewport:
            self.split_viewport = not ke.imgui.dock_window(
                "Resize Test", "Viewport", position="below", ratio=0.35,
            )
            self.show_resize_panel = True
        if ke.imgui.begin("Viewport"):
            self.viewport_size = ke.imgui.content_region_avail()
        ke.imgui.end()
        if self.show_resize_panel:
            ke.imgui.begin("Resize Test")
            ke.imgui.end()

    def post_render(self):
        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        width, height = int(viewport[2]), int(viewport[3])
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_BACK)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
        self.window_pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)[::-1].copy()


def resize_window(app, glfw, width, height):
    glfw.glfwSetWindowSize(glfw.glfwGetCurrentContext(), width, height)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        app.process_events()
        app.draw_frame()
        if app.window_pixels.shape == (height, width, 3):
            return
        time.sleep(0.01)
    raise AssertionError(f"Window did not resize to {width}x{height}")


def check_cube_aspect(image):
    # The front-facing cube must stay square across render aspect changes.
    red, green, blue = image.astype(int).transpose(2, 0, 1)
    rows, columns = np.nonzero((green > 100) & (green > red + 30) & (green > blue + 30))
    assert len(rows), f"Rendered cube is missing: max RGB={image.max(axis=(0, 1))}"
    assert abs(np.ptp(rows) - np.ptp(columns)) <= 3


def check_render_size(app, glfw, backend):
    width, height = app.get_width(), app.get_height()
    if backend == "ovrtx" and app.get_ui_layout_mode() == ke.UILayoutMode.EDITOR:
        logical_width, logical_height = ctypes.c_int(), ctypes.c_int()
        glfw.glfwGetWindowSize(glfw.glfwGetCurrentContext(),
                               ctypes.byref(logical_width), ctypes.byref(logical_height))
        width = int(app.viewport_size[0] * width / logical_width.value + 0.5)
        height = int(app.viewport_size[1] * height / logical_height.value + 0.5)
        assert width * height < app.get_width() * app.get_height()
    assert (app.get_render_width(), app.get_render_height()) == (width, height)
    image = app.read_rgb_pixels()
    assert image.shape == (height, width, 3), image.shape
    projection = np.asarray(app.get_projection_matrix())
    np.testing.assert_allclose(projection[1, 1] / projection[0, 0], width / height, rtol=1e-5)
    check_cube_aspect(image)
    return image


def check_textures(app, material):
    # New contents and paths exercise asset loading without a texture-cache hit.
    rng = np.random.default_rng()
    material.base_color = [1, 1, 1, 1]
    captures = []
    with tempfile.TemporaryDirectory() as directory:
        for name, channel in (("red", 0), ("blue", 2)):
            pixels = rng.integers(10, 30, (1024, 1024, 3), dtype=np.uint8)
            pixels[..., channel] = rng.integers(220, 245, (1024, 1024), dtype=np.uint8)
            path = Path(directory) / f"{name}.png"
            Image.fromarray(pixels).save(path)
            material.set_base_color_texture(app.load_texture(path, srgb=True))
            app.draw_frame()
            first = app.read_rgb_pixels()
            for _ in range(8):
                app.draw_frame()
            settled = app.read_rgb_pixels()
            for image in (first, settled):
                rgb = image.astype(int)
                dominant = rgb[..., channel] - np.max(np.delete(rgb, channel, axis=2), axis=2)
                assert np.count_nonzero(dominant > 50) > 100, name
            captures.append(first)
        assert np.abs(captures[0].astype(float) - captures[1]).mean() > 1
        material.base_color_texture = None
        app.draw_frame()
        assert np.abs(app.read_rgb_pixels().astype(float) - captures[-1]).mean() > 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    maps = []
    progress = []
    if args.backend == "ovrtx":
        import ovrtx

        original_map = ovrtx.RenderVarOutput.map
        original_step = ovrtx.Renderer.step

        def record_map(self, device=ovrtx.Device.CPU, sync_stream=None):
            maps.append(device)
            return original_map(self, device=device, sync_stream=sync_stream)

        ovrtx.RenderVarOutput.map = record_map

        def record_step(self, *args, **kwargs):
            outputs = original_step(self, *args, **kwargs)
            progress.append(outputs["/Render/Camera"].frames[-1].progression)
            return outputs

        ovrtx.Renderer.step = record_step

    # Use the engine's GLFW instance; the Python glfw package may load another.
    glfw = ctypes.CDLL(_ke.__file__)
    glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
    glfw.glfwSetWindowSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    glfw.glfwGetWindowSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_int)]
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
            for _ in range(4):
                app.draw_frame()
            assert progress[-1] > 1, "static rendering repeatedly resets accumulation"

        app.set_camera_view(position=[1.5, 0.5, 4], target=[0, 0, 0])
        app.draw_frame()
        assert not np.array_equal(first, app.window_pixels)
        app.set_camera_view(position=[0, 0, 4], target=[0, 0, 0])
        app.debug.overlay.points("/resize/debug", [[0, 0, 0]],
                                 [1, 0, 0, 1], size=14, overlay=True)
        for width, height in ((311, 217), (640, 360), (257, 193), (311, 217)):
            map_count = len(maps)
            resize_window(app, glfw, width, height)
            if args.backend == "ovrtx":
                assert all(device == ovrtx.Device.CUDA_ARRAY for device in maps[map_count:])
            resized = app.read_rgb_pixels()
            assert resized.shape == (height, width, 3), resized.shape
            check_cube_aspect(resized)
            assert (resized[..., 0] > 240).sum() > 50, "Debug point lost during resize"
            np.testing.assert_allclose(app.window_pixels.astype(int), resized.astype(int), atol=1)
            # Keep the next CUDA output mapped until the following resize.
            app.draw_frame()

        app.debug.overlay.remove_all()
        app.draw_frame()
        assert (app.read_rgb_pixels()[..., 0] > 240).sum() < 50

        app.request_close()
        assert app.should_close()
        app.close()

        app = WindowApp()
        app.initialize(width=800, height=600, render_backend=args.backend)
        app.set_ui_layout_mode(mode=ke.UILayoutMode.EDITOR)
        build_scene(app)
        map_count = len(maps)
        for _ in range(4):
            app.process_events()
            app.draw_frame()
        assert app.window_pixels.max() > 100
        if args.backend == "ovrtx":
            assert all(device == ovrtx.Device.CUDA_ARRAY for device in maps[map_count:])
        for mode, size in ((ke.UILayoutMode.EDITOR, (960, 640)),
                           (ke.UILayoutMode.VIEWER, (640, 480)),
                           (ke.UILayoutMode.EDITOR, (800, 600))):
            app.set_ui_layout_mode(mode=mode)
            resize_window(app, glfw, *size)
            for _ in range(3):
                app.draw_frame()
            check_render_size(app, glfw, args.backend)

        original_size = app.get_render_width(), app.get_render_height()
        app.split_viewport = True
        for _ in range(4):
            app.draw_frame()
        assert not app.split_viewport
        check_render_size(app, glfw, args.backend)
        assert (app.get_width(), app.get_height()) == (800, 600)
        if args.backend == "ovrtx":
            assert app.get_render_height() < original_size[1]
        assert app.read_rgb_pixels_resized(width=160, height=120).shape == (120, 160, 3)

        app.close()
        app = WindowApp()
        app.initialize(width=257, height=193, hide_ui=True, render_backend=args.backend)
        _, _, material = build_scene(app)
        check_textures(app, material)
    finally:
        app.close()
        app.close()
        if args.backend == "ovrtx":
            ovrtx.RenderVarOutput.map = original_map
            ovrtx.Renderer.step = original_step
    print(f"PASS: {args.backend} window, camera, resize, textures, readback, and close")


if __name__ == "__main__":
    main()
