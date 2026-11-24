"""Run the same App scene through Native or optional ovrtx."""

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke


class BackendSmokeApp(ke.App):
    def __init__(self):
        super().__init__()
        self.calls = []

    def pre_render(self):
        self.calls.append("pre")

    def render(self):
        self.calls.append("render")

    def post_render(self):
        self.calls.append("post")


def build_scene(app):
    app.set_camera_view([0, 0, 4], [0, 0, 0])
    light = ke.scene.DirectionalLight()
    light.direction = ke.Vec3(0, 0, -1)
    light.intensity = 3
    light.ambient = ke.Vec3(0, 0, 0)
    app.scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
    material = app.create_pbr_material(base_color=[0.05, 0.8, 0.1, 1], roughness=0.8)
    parent = app.scene.define_prim("/group", ke.scene.PrimType.XFORM)
    surface = app.scene.add_mesh("/group/cube", ke.geometry.create_cube_data(1.5), material)
    return parent, surface, material


def check_animation_physics(app, parent):
    config = ke.physics.PhysicsConfig.y_up()
    config.enable_gpu = False
    config.dt = 1.0 / 120.0
    world = ke.physics.PhysicsWorld(config=config)
    tree = ke.animation.SkeletonTree(["root"], [-1], [[0, 0, 0]], [[1, 0, 0, 0]])
    rotations = np.array([[np.cos(0.2), 0, 0, np.sin(0.2)]], dtype=np.float32)
    initial = ke.animation.SkeletonState.from_rotation_and_root_translation(
        tree, rotations, np.array([-0.3, 0.4, 0], dtype=np.float32),
    )
    source = initial.compute_global_matrices().copy()
    body = world.create_dynamic_sphere(
        radius=0.2, pos=source[0, :3, 3], quat_xyzw=rotations[0, [1, 2, 3, 0]],
    )
    images = []
    for _ in range(3):
        for _ in range(8):
            world.step()
        state = ke.animation.SkeletonState.from_rotation_and_root_translation(
            tree, np.asarray(body.get_root_quat_xyzw(), dtype=np.float32)[None, [3, 0, 1, 2]],
            np.asarray(body.get_root_position(), dtype=np.float32),
        )
        parent.set_world_translation(ke.Vec3(*state.compute_global_matrices()[0, :3, 3]))
        parent.set_world_quat_wxyz(state.get_joint_quat(0))
        app.draw_frame()
        images.append(app.read_rgb_pixels())
    assert body.get_root_position()[1] < source[0, 1, 3] - 0.1
    assert not np.array_equal(images[0], images[-1])
    np.testing.assert_array_equal(initial.compute_global_matrices(), source)
    paused = body.get_root_position().copy()
    app.draw_frame()
    app.draw_frame()
    np.testing.assert_array_equal(body.get_root_position(), paused)


def check_python_readback():
    pixels = np.array([
        [[0, 20, 40], [40, 60, 80]],
        [[80, 100, 120], [120, 140, 160]],
    ], dtype=np.uint8)
    expected = pixels.copy()
    enlarged = np.array([
        [[0, 20, 40], [20, 40, 60], [40, 60, 80]],
        [[40, 60, 80], [60, 80, 100], [80, 100, 120]],
        [[80, 100, 120], [100, 120, 140], [120, 140, 160]],
    ], dtype=np.uint8)

    class ImageAdapter:
        def render(self, frame):
            assert "_snapshot" not in frame
            return pixels

        def close(self):
            pass

    app = ke.App()
    try:
        app._initialize_scene_renderer(
            2, 2, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
            lambda width, height: ImageAdapter(),
        )
        try:
            app.read_rgb_pixels_resized(2, 2)
        except RuntimeError as error:
            assert "No completed scene frame" in str(error), error
        else:
            raise AssertionError("readback before a completed frame succeeded")
        app.draw_frame()
        pixels[:] = 255
        for flip in (True, False):
            image = expected if flip else expected[::-1]
            np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=flip), image)
            np.testing.assert_array_equal(app.read_rgb_pixels_resized(2, 2, flip_y=flip), image)
            np.testing.assert_array_equal(
                app.read_rgb_pixels_resized(3, 3, flip_y=flip),
                enlarged if flip else enlarged[::-1],
            )
            np.testing.assert_array_equal(
                app.read_rgb_pixels_resized(1, 1, flip_y=flip), [[[60, 80, 100]]],
            )
        owned = app.read_rgb_pixels(flip_y=False)
        owned[:] = 0
        np.testing.assert_array_equal(app.read_rgb_pixels(), expected)
        saved = app.read_rgb_pixels_resized(3, 3)
        app.draw_frame()
        np.testing.assert_array_equal(saved, enlarged)
    finally:
        app.close()
    np.testing.assert_array_equal(saved, enlarged)
    print("PASS: Python backend image ownership, row order, and resizing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    check_python_readback()
    app = BackendSmokeApp()
    try:
        app.initialize(width=128, height=96, hide_ui=True, headless=True, render_backend=args.backend)
        assert app.get_renderer().capabilities["native_extensions"] == (args.backend == "native")
        assert app.get_renderer().capabilities["static_scene"]
        assert app.get_renderer().capabilities["texture_pixels_input"]
        assert app.get_renderer().capabilities["cpu_instance_transforms_input"]
        assert not app.get_renderer().capabilities["cuda_image_output"]
        assert app.get_renderer().capabilities["screen_picking"]
        parent, surface, material = build_scene(app)
        app.get_renderer().set_profiler_enabled(True)
        app.draw_frame()
        assert app.calls == ["pre", "render", "post"]
        green = app.read_rgb_pixels()
        assert green.shape == (96, 128, 3) and green.dtype == np.uint8
        assert int(green[48, 64, 1]) > int(green[48, 64, 2]) + 20
        np.testing.assert_array_equal(app.read_rgb_pixels(flip_y=False), green[::-1])
        assert app.read_rgb_pixels_resized(32, 24).shape == (24, 32, 3)

        material.base_color = [0.05, 0.1, 0.8, 1]
        parent.set_local_translation(ke.Vec3(0.2, 0, 0))
        app.draw_frame()
        blue = app.read_rgb_pixels()
        assert int(blue[48, 64, 2]) > int(blue[48, 64, 1]) + 20
        surface.set_visible(False)
        app.draw_frame()
        assert not np.array_equal(app.read_rgb_pixels(), blue)
        surface.set_visible(True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "red.png"
            Image.new("RGB", (8, 8), (240, 20, 20)).save(path)
            texture = app.load_texture(path, srgb=True)
            assert app.load_texture(path, srgb=True) is texture
            material.base_color = [1, 1, 1, 1]
            material.set_base_color_texture(texture)
            assert material.base_color_texture is texture
            app.draw_frame()
            red = app.read_rgb_pixels()
            assert int(red[48, 64, 0]) > int(red[48, 64, 2]) + 20
            material.base_color_texture = None
            assert material.base_color_texture is None
            app.draw_frame()
            assert not np.array_equal(app.read_rgb_pixels(), red)

            if args.backend == "ovrtx":
                other_resources = ke.scene.SceneResourceManager()
                handle = other_resources.register_texture_source("Red", str(path), srgb=True)
                foreign = ke.render.TextureSource(str(path), True, True, handle, other_resources)
                try:
                    material.base_color_texture = foreign
                except ValueError as error:
                    assert "different App" in str(error)
                else:
                    raise AssertionError("foreign scene-local texture handle was accepted")
                assert material.base_color_texture is None

        if args.backend == "ovrtx":
            for operation in (
                lambda: app.set_gamma(2.2),
                lambda: app.raycast([0, 0, 1], [0, 0, -1]),
            ):
                try:
                    operation()
                except RuntimeError as error:
                    assert "Native" in str(error), error
                else:
                    raise AssertionError("Native-only operation was silently accepted")
            assert app.get_renderer().device() is None
            assert app.get_renderer().latest_frame_profile().backend is None
            assert not app.is_key_pressed(32)
            assert not app._is_gamepad_connected()
            assert not app._get_gamepad_state(0)["connected"]
            assert app._list_joysticks() == []
            memory = app.create_texture(np.full((4, 4, 3), 200, dtype=np.uint8), srgb=True)
            assert isinstance(memory, ke.render.TextureSource) and memory.path is None
            material.base_color_texture = memory
            app.draw_frame()
            assert app.read_rgb_pixels()[48, 64].min() > 20
            material.base_color_texture = None
            material.unlit = True
            app.draw_frame()
            assert app.read_rgb_pixels()[48, 64].min() > 200
            material.unlit = False
            app.draw_frame()
        check_animation_physics(app, parent)
        app.scene.remove_prim("/group")
        app.draw_frame()
        assert app.get_scene_render_system().registration_count == 0
        saved = app.read_rgb_pixels()
        app.request_close()
        assert app.should_close()
    finally:
        app.close()
    assert saved.shape == (96, 128, 3)
    print(f"PASS: {args.backend} App scene, animation/physics poses, callbacks, capture, and teardown")


if __name__ == "__main__":
    main()
