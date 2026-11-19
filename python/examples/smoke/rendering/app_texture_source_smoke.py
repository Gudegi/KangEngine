"""Check App texture sources and rendering after the source App closes."""

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import kangengine as ke
from kangengine._core import _ke


def collect_frames(directory):
    red_path = directory / "red.png"
    blue_path = directory / "blue.png"
    Image.new("RGB", (8, 8), (220, 30, 30)).save(red_path)
    Image.new("RGB", (8, 8), (30, 30, 220)).save(blue_path)
    app = ke.App()
    app.initialize(width=96, height=96, hide_ui=True, headless=True)
    try:
        red = app.load_texture(red_path, srgb=True)
        assert app.load_texture(red_path, srgb=True) is red
        blue = app.load_texture(blue_path, srgb=True)
        linear = app.load_texture(red_path, srgb=False, flip=False)
        material = app.create_pbr_material(base_color_texture=red, roughness=0.8)
        view = app.scene.add_mesh("/cube", ke.geometry.create_cube_data(1.5), material=material)
        camera = app.scene.define_prim("/camera", ke.scene.PrimType.CAMERA)
        camera.set_local_translation(ke.Vec3(0, 0, 5))
        camera.add_camera_component().set_perspective(45, 0.1, 100)
        app.set_light_direction(ke.Vec3(0, 0, -1))
        app.set_light_intensity(3)
        app.set_light_ambient(ke.Vec3(0, 0, 0))
        collector = _ke._RenderSceneDataCollector()

        def collect():
            return collector.collect(app.scene.native, app.resources, "/camera")

        initial = collect()
        source = initial["objects"][0]["textures"]["base_color_texture"]
        assert source["path"] == str(red_path.resolve())
        assert source["srgb"] and source["flip"]
        assert not initial["objects"][0]["material"]["has_native_textures"]
        handle = source["id"]
        assert app.resources.texture(handle) is red
        app.resources.invalidate_usage_cache()
        assert app.resources.usage_count(handle) == 1
        assert collect()["objects"][0]["textures"]["base_color_texture"]["id"] == handle

        material.base_color_texture = linear
        variant = collect()["objects"][0]["textures"]["base_color_texture"]
        assert variant["id"] != handle and not variant["srgb"] and not variant["flip"]
        material.base_color_texture = red
        material.texture_resources = {"base_color_texture": handle}
        assert not collect()["objects"][0]["material"]["has_native_textures"]
        other_handle = app.resources.register_texture_source("Other", str(blue_path), srgb=True)
        material.texture_resources = {"base_color_texture": other_handle}
        try:
            collect()
        except ValueError as error:
            assert "bindings disagree" in str(error), error
        else:
            raise AssertionError("conflicting Native/source textures were accepted")
        material.texture_resources = {}

        # Every existing material slot uses the same resource lookup.
        material.base_color_texture = None
        for slot in ("base_color_texture", "normal_texture", "metallic_roughness_texture",
                     "metallic_texture", "roughness_texture", "ao_texture", "orm_texture",
                     "emissive_texture", "opacity_texture"):
            setattr(material, slot, red)
            item = collect()["objects"][0]
            assert list(item["textures"]) == [slot] and item["textures"][slot]["id"] == handle
            assert not item["material"]["has_native_textures"]
            setattr(material, slot, None)
        phong = app.create_phong_material()
        view.set_material(phong)
        for slot in ("diffuse_map", "specular_map", "alpha_map", "normal_map"):
            setattr(phong, slot, red)
            item = collect()["objects"][0]
            assert list(item["textures"]) == [slot] and item["textures"][slot]["id"] == handle
            assert not item["material"]["has_native_textures"]
            setattr(phong, slot, None)
        view.set_material(material)

        raw = app.get_renderer().device().create_texture(str(red_path), True, srgb=True)
        app._remember_textures(raw)
        material.base_color_texture = raw
        assert collect()["objects"][0]["material"]["has_native_textures"]
        try:
            ke.render.SceneRenderer(app.scene.native, app.resources, camera_path="/camera")
        except ValueError as error:
            assert "texture_resources" in str(error), error
        else:
            raise AssertionError("a texture without source metadata was silently accepted")

        material.base_color_texture = red
        app.draw_frame()
        restored = collect()
        material.base_color_texture = blue
        replaced = collect()
        material.base_color_texture = None
        removed = collect()
        assert not removed["objects"][0]["textures"]
        return restored, replaced, removed
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        adapter = None
        if args.backend == "ovrtx":
            # This SDK stalls when first initialized after the source App closes.
            from kangengine.render._ovrtx_scene import OvrtxSceneAdapter
            adapter = OvrtxSceneAdapter(96, 96)
        try:
            frames = collect_frames(Path(directory))
            if adapter is None:
                from kangengine.render._native_scene import NativeSceneAdapter
                adapter = NativeSceneAdapter(96, 96)
            images = [adapter.render(frame) for frame in frames]
            for image in images:
                assert image.shape == (96, 96, 3) and image.dtype == np.uint8
            red, blue, plain = [image[40:56, 40:56].mean(axis=(0, 1)) for image in images]
            assert red[0] > red[2] + 30, red
            assert blue[2] > blue[0] + 30, blue
            assert abs(plain[0] - plain[2]) < 10, plain
        finally:
            if adapter is not None:
                adapter.close()
    print(f"PASS: {args.backend} App texture sources, slots, replacement, removal, and lifetime")


if __name__ == "__main__":
    main()
