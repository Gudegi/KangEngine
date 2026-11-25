"""Check dome emission, hierarchy visibility, replacement, and HDRI rotation."""

import argparse
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import kangengine as ke

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rendering"))
from static_scene_renderer import build_scene


def write_environment(path, reverse=False, *, poles=False):
    # Linear RGBE values above one exercise HDR loading without an external asset.
    pixels = np.empty((16, 32, 4), dtype=np.uint8)
    pixels[:, :16] = (128, 8, 8, 130)
    pixels[:, 16:] = (8, 8, 128, 130)
    if poles:
        pixels[:8] = (128, 8, 8, 130)
        pixels[8:] = (8, 8, 128, 130)
    if reverse:
        pixels = pixels[:, ::-1]
    path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 16 +X 32\n" + pixels.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    scene, resources, material, _ = build_scene(
        Path(ke.__file__).parent / "assets/textures/awesomeface.png",
    )
    material.texture_resources = {}
    material.base_color = ke.Vec4(1, 1, 1, 1)
    scene.get_prim_at_path("/light").set_visible(False)
    parent = scene.define_prim("/environment", ke.scene.PrimType.XFORM)
    dome = scene.define_prim("/environment/sky", ke.scene.PrimType.LIGHT)
    light = ke.scene.DomeLight()
    light.intensity = 0.3
    dome.set_dome_light(light)

    with tempfile.TemporaryDirectory() as directory, ke.render.SceneRenderer(
        scene, resources, camera_path="/camera", renderer=args.renderer, width=128, height=128,
    ) as renderer:
        def capture():
            for _ in range(16):
                image = renderer.render()
            assert image.shape == (128, 128, 3)
            return image.astype(float)

        bright = capture()
        parent.set_visible(False)
        dark = capture()
        assert bright[50:78, 50:78].mean() > dark[50:78, 50:78].mean() + 10
        parent.set_visible(True)
        parent.set_active(False)
        inactive = capture()
        assert abs(inactive[50:78, 50:78].mean() - dark[50:78, 50:78].mean()) < 3
        assert parent.is_visible(), "active state changed authored visibility"
        parent.set_active(True)
        light.color = ke.Vec3(1, 0.05, 0.05)
        dome.set_dome_light(light)
        red = capture()[50:78, 50:78].mean(axis=(0, 1))
        assert red[0] > red[2] + 10, red
        light.intensity = 0
        dome.set_dome_light(light)
        zero = capture()
        assert zero[50:78, 50:78].mean() < bright[50:78, 50:78].mean() - 10

        scene.remove_prim("/environment/sky")
        capture()
        dome = scene.define_prim("/environment/sky", ke.scene.PrimType.LIGHT)
        light.color = ke.Vec3(1, 1, 1)
        light.intensity = 0.3
        dome.set_dome_light(light)
        restored = capture()
        assert restored[50:78, 50:78].mean() > dark[50:78, 50:78].mean() + 10

        second = scene.define_prim("/environment/fill", ke.scene.PrimType.LIGHT)
        second.set_dome_light(light)
        combined = capture()
        assert combined[50:78, 50:78].mean() > restored[50:78, 50:78].mean() + 5
        scene.remove_prim("/environment/fill")

        if args.renderer == "ovrtx":
            texture = Path(directory) / "environment.hdr"
            replacement = Path(directory) / "replacement.hdr"
            write_environment(texture)
            write_environment(replacement, reverse=True)
            light.texture_path = str(texture)
            light.texture_format = "latlong"
            dome.set_dome_light(light)
            textured = capture()
            parent.set_local_rotation_axis_angle(axis=ke.Vec3(0, 1, 0), angle_radians=math.pi)
            rotated = capture()
            assert np.abs(rotated[:24] - textured[:24]).mean() > 10
            light.texture_path = str(replacement)
            dome.set_dome_light(light)
            replaced = capture()
            assert np.abs(replaced[:24] - rotated[:24]).mean() > 10

            # The HDRI top is +Y locally; a +90-degree X rotation makes it +Z.
            poles = Path(directory) / "poles.hdr"
            write_environment(poles, poles=True)
            light.texture_path = str(poles)
            dome.set_dome_light(light)
            scene.remove_prim("/group")
            camera = scene.get_prim_at_path("/camera")
            for angle in (0, math.pi / 2):
                parent.set_local_rotation_axis_angle(ke.Vec3(1, 0, 0), angle)
                for offset, channel in ((math.pi / 2, 0), (-math.pi / 2, 2)):
                    camera.set_local_rotation_axis_angle(ke.Vec3(1, 0, 0), angle + offset)
                    color = capture()[50:78, 50:78].mean(axis=(0, 1))
                    assert color[channel] > color[2 - channel] + 50, (angle, offset, color)

            light.texture_path = str(Path(directory) / "missing.hdr")
            dome.set_dome_light(light)
            try:
                renderer.render()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Missing HDRI was accepted")
            light.texture_path = ""
            dome.set_dome_light(light)
            capture()

    print(f"PASS: {args.renderer} dome emission, hierarchy, and removal")
    if args.renderer == "ovrtx":
        print("PASS: HDRI loading, rotation, replacement, and error recovery")


if __name__ == "__main__":
    main()
