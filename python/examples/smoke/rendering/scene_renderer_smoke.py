"""Verify identical static scene authoring with either render adapter."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import kangengine as ke

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rendering"))
from static_scene_renderer import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    texture = np.zeros((64, 64, 3), dtype=np.uint8)
    texture[:32, :32] = [240, 30, 30]
    texture[:32, 32:] = [30, 240, 30]
    texture[32:, :32] = [30, 30, 240]
    texture[32:, 32:] = [240, 240, 30]
    texture_path = args.output / "texture.png"
    Image.fromarray(texture).save(texture_path)
    scene, resources, material, add_cube = build_scene(texture_path)
    parent = scene.get_prim_at_path("/group")
    images = {}

    with ke.render.SceneRenderer(scene, resources, camera_path="/camera",
                                 backend=args.backend, width=256, height=256) as renderer:
        def capture(name):
            pixels = renderer.render()
            assert pixels.shape == (256, 256, 3), pixels.shape
            assert pixels.dtype == np.uint8
            images[name] = pixels
            Image.fromarray(pixels).save(args.output / f"{name}.png")
            return pixels

        initial = capture("initial")
        saved_initial = initial.copy()
        if args.backend == "native":
            app = renderer._adapter._app
            assert app.get_scene_render_system().registration_count == 0
            assert app.get_scene().get_prim_at_path("/instances") is None
            np.testing.assert_array_equal(renderer.render(), initial)
            textures = material.texture_resources
            cwd = Path.cwd()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    os.chdir(directory)
                    for _ in range(3):
                        parent.set_visible(False)
                        material.texture_resources = {}
                        renderer.render()
                        material.texture_resources = textures
                        renderer.render()
                        parent.set_visible(True)
                        np.testing.assert_array_equal(renderer.render(), initial)
            finally:
                os.chdir(cwd)
                material.texture_resources = textures
                parent.set_visible(True)
        if args.backend == "ovrtx":
            # Rejected asset paths must leave the live stage registrations intact.
            valid_textures = material.texture_resources
            adapter = renderer._adapter
            for filename in ("invalid@texture.png", "invalid\ntexture.png"):
                invalid_path = args.output / filename
                Image.fromarray(texture).save(invalid_path)
                material.texture_resources = {"base_color_texture": resources.register_texture_source(
                    "Invalid", str(invalid_path), srgb=True)}
                references = adapter._references.copy()
                ordinal = adapter._ordinal
                try:
                    renderer.render()
                except ValueError as error:
                    assert "ovrtx texture asset paths" in str(error), error
                else:
                    raise AssertionError("unsupported texture path was silently accepted")
                finally:
                    material.texture_resources = valid_textures
                assert adapter._references == references, "rejected path changed live registrations"
                assert adapter._ordinal == ordinal, "rejected path advanced the stage ordinal"
            capture("after_path_rejection")

        cube = scene.get_prim_at_path("/group/cube")
        cube.set_display_color(ke.Vec3(0.05, 1, 1))
        object_tinted = capture("object_tinted")
        assert object_tinted[:, :, 0].mean() < initial[:, :, 0].mean() - 1
        cube.get_render_component().display_color = ke.Vec4(1, 1, 1, 1)

        parent.set_local_translation(ke.Vec3(0.9, 0, 0))
        capture("moved")
        material.base_color = ke.Vec4(0.15, 1, 1, 1)
        capture("tinted")
        parent.set_visible(False)
        capture("hidden")
        parent.set_active(False)
        parent.set_visible(True)
        capture("inactive")
        parent.set_active(True)
        capture("restored")
        scene.remove_prim("/group/cube")
        capture("removed")
        cube = add_cube()
        capture("recreated")
        cube.set_local_translation(ke.Vec3(-0.65, 0, 0))
        capture("recreated_moved")
        scene.remove_prim("/group/cube")
        add_cube()
        capture("replaced")
        swapped_path = args.output / "texture-swapped.png"
        Image.fromarray(np.flip(texture, axis=1).copy()).save(swapped_path)
        material.texture_resources = {"base_color_texture": resources.register_texture_source(
            "Swapped", str(swapped_path), srgb=True)}
        capture("texture_replaced")
        material.texture_resources = {}
        capture("untextured")

        light_prim = scene.get_prim_at_path("/light")
        light = light_prim.get_directional_light()
        light.color = ke.Vec3(0.1, 0.1, 1)
        light_prim.set_directional_light(light)
        capture("blue_light")
        light.color = ke.Vec3(1, 1, 1)
        light_prim.set_directional_light(light)
        camera = scene.get_prim_at_path("/camera")
        camera.set_local_translation(ke.Vec3(-0.4, 0, 5))
        capture("camera_moved")
        camera.set_local_translation(ke.Vec3(0, 0, 5))

        cube = scene.get_prim_at_path("/group/cube")
        small_mesh = ke.geometry.create_cube_data(0.7)
        cube.set_mesh_data(small_mesh)
        cube.get_mesh_component().resource_handle = resources.register_mesh("Small", small_mesh)
        capture("mesh_replaced")

        material.unlit = True
        if args.backend == "native":
            try:
                renderer.render()
            except ValueError:
                pass
            else:
                raise AssertionError("unsupported Native snapshot material was accepted")
        else:
            capture("unlit")
        material.unlit = False
        capture("after_rejection")
        np.testing.assert_array_equal(initial, saved_initial)

    background = images["removed"].astype(np.int16)
    for name in ("hidden", "inactive"):
        assert np.abs(images[name].astype(np.int16) - background).max() <= 3, name
    centers = {}
    for name in images.keys() - {"hidden", "inactive", "removed"}:
        mask = np.max(np.abs(images[name].astype(np.int16) - background), axis=2) > 12
        ys, xs = np.nonzero(mask)
        assert len(xs) > 100, name
        centers[name] = float(xs.mean())
    assert centers["moved"] > centers["initial"] + 15, centers
    if args.backend == "ovrtx":
        assert abs(centers["after_path_rejection"] - centers["initial"]) < 5, centers
    assert abs(centers["recreated_moved"] - centers["initial"]) < 5, centers
    assert abs(centers["replaced"] - centers["moved"]) < 5, centers
    assert np.abs(images["moved"].astype(float) - images["tinted"]).mean() > 1
    assert np.abs(images["replaced"].astype(float) - images["untextured"]).mean() > 1
    for first, second in (("replaced", "texture_replaced"), ("untextured", "blue_light"),
                           ("untextured", "camera_moved"), ("untextured", "mesh_replaced")):
        assert np.abs(images[first].astype(float) - images[second]).mean() > 1, (first, second)
    np.testing.assert_array_equal(initial, saved_initial)
    assert scene.get_prim_at_path("/group/cube") is not None
    print(f"PASS: {args.backend} scene renderer, {len(images)} saved frames; centers={centers}")


if __name__ == "__main__":
    main()
