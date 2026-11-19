"""Verify immutable image sources, Native uploads, and renderer recreation."""

import gc
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke
from kangengine._core import _ke


def image_pixels(channels=3):
    pixels = np.full((7, 5, channels), 255, dtype=np.uint8)
    pixels[:3, :, :3] = [220, 30, 50]
    pixels[3:, :, :3] = [20, 80, 230]
    return pixels


def check_cpu_sources():
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    resources = ke.scene.SceneResourceManager(scene)
    pixels = image_pixels(4)
    expected = pixels.copy()
    handle = resources.register_texture_image("Image", pixels, srgb=True)
    pixels[:] = 0
    material = ke.material.PBRMaterial()
    material.texture_resources = {"base_color_texture": handle}
    resources.register_material("Surface", material.native)
    mesh = ke.geometry.create_cube_data(1)
    prim = scene.define_prim("/cube", ke.scene.PrimType.MESH)
    prim.set_mesh_data(mesh)
    prim.get_mesh_component().resource_handle = resources.register_mesh("Cube", mesh)
    prim.set_material(material.native)
    prim.add_render_component()
    camera = scene.define_prim("/camera", ke.scene.PrimType.CAMERA)
    camera.add_camera_component().set_perspective(45, 0.1, 100)
    frame = _ke._RenderSceneDataCollector().collect(scene, resources, "/camera")
    source = frame["objects"][0]["textures"]["base_color_texture"]
    view = source["pixels"]
    assert source["path"] == "" and source["srgb"] and source["flip"]
    assert not view.flags.writeable
    np.testing.assert_array_equal(view, expected)
    for invalid in (pixels.astype(np.float32), pixels[:, ::2], pixels[:0], pixels[:, :, :2].copy()):
        try:
            resources.register_texture_image("Invalid", invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid native image layout was accepted")
    del frame, source, resources, scene, prim, camera, material
    gc.collect()
    np.testing.assert_array_equal(view, expected)
    print("PASS: memory source validation, input copy, read-only view, and lifetime")


def collect_native_frames(directory):
    frames = []
    app = ke.App()
    app.initialize(width=96, height=96, hide_ui=True, headless=True)
    try:
        camera = app.scene.define_prim("/camera", ke.scene.PrimType.CAMERA)
        camera.set_local_translation(ke.Vec3(0, 0, 5))
        camera.add_camera_component().set_perspective(45, 0.1, 100)
        app.set_active_scene_camera(camera)
        app.set_light_direction([0, 0, -1])
        app.set_light_intensity(3)
        app.set_light_ambient([0, 0, 0])
        material = app.create_pbr_material(roughness=0.8)
        app.scene.add_mesh("/cube", ke.geometry.create_cube_data(1.5), material)
        collector = _ke._RenderSceneDataCollector()
        for channels in (3, 4):
            for flip in (False, True):
                for srgb in (False, True):
                    pixels = image_pixels(channels)
                    path = directory / f"image-{channels}.png"
                    Image.fromarray(pixels).save(path)
                    material.base_color_texture = app.load_texture(path, srgb=srgb, flip=flip)
                    app.draw_frame()
                    expected = app.read_rgb_pixels()
                    texture = app.create_texture(pixels, srgb=srgb, flip=flip)
                    pixels[:] = 0
                    material.base_color_texture = texture
                    app.draw_frame()
                    np.testing.assert_array_equal(app.read_rgb_pixels(), expected)
                    frame = collector.collect(app.scene.native, app.resources, "/camera")
                    source = frame["objects"][0]["textures"]["base_color_texture"]
                    np.testing.assert_array_equal(source["pixels"], image_pixels(channels))
                    assert not frame["objects"][0]["material"]["has_native_textures"]
                    frames.append((frame, expected))
    finally:
        app.close()
    return frames


def main():
    check_cpu_sources()
    with tempfile.TemporaryDirectory() as directory:
        frames = collect_native_frames(Path(directory))
    from kangengine.render._native_scene import NativeSceneAdapter
    from kangengine.render.scene_renderer import _validate_frame

    adapter = NativeSceneAdapter(96, 96)
    try:
        for frame, expected in frames:
            frame = _validate_frame(frame, {}, set())
            np.testing.assert_array_equal(adapter.render(frame), expected)
        adapter._app.get_renderer().set_profiler_enabled(True)
        adapter.render(frames[-1][0])
        profile = adapter._app.get_renderer().latest_frame_profile()
        assert profile.counters.texture_upload_bytes == 0
    finally:
        adapter.close()
    print("PASS: RGB/RGBA, sRGB/linear, row flip, file parity, source recreation, and GPU reuse")


if __name__ == "__main__":
    main()
