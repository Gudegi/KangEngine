"""Check material models, unlit color, alpha, PBR maps, and images."""

from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import kangengine as ke
from kangengine.render._ovrtx_scene import OvrtxAppAdapter


def check_opacity(app, view, capture):
    material = app.create_pbr_material(base_color=[1, 0, 0, 1], roughness=0.5)
    view.set_material(material)
    view.set_alpha_mode(ke.render.AlphaMode.BLEND)
    backdrop_material = app.create_pbr_material(base_color=[0, 1, 0, 1])
    backdrop_material.unlit = True
    backdrop = app.scene.add_mesh("/opacity_background", ke.geometry.create_cube_data(4), backdrop_material)
    backdrop.set_local_translation(ke.Vec3(0, 0, -4))
    app.set_light_direction([0, 0, -1])
    app.set_light_intensity(3)

    opaque = capture()[96, 128]
    material.base_color = [1, 0, 0, 0.05]
    transparent = capture()[96, 128]
    assert transparent[1] > opaque[1] + 50, (opaque, transparent)

    # A separate opacity map multiplies the material's opacity.
    material.base_color = [1, 0, 0, 1]
    material.opacity_texture = app.create_texture(
        np.full((8, 8, 3), [13, 255, 255], dtype=np.uint8), srgb=False,
    )
    mapped = capture()[96, 128]
    assert np.abs(mapped - transparent).max() < 15, (mapped, transparent)
    material.opacity_texture = None
    assert capture()[96, 128, 1] < transparent[1] - 50

    backdrop.remove()
    view.set_alpha_mode(ke.render.AlphaMode.OPAQUE)


def check_pbr_maps(app, adapter, view, capture):
    material = app.create_pbr_material(base_color=[0.7, 0.3, 0.1, 1], metallic=1, roughness=0.8)
    view.set_material(material)
    app.set_light_intensity(3)
    app.set_light_direction([-0.4, 0, -1])
    for attribute in ("metallic", "roughness"):
        slot = attribute + "_texture"
        factor = getattr(material, attribute)
        # Only red carries the scalar map; green and blue must not affect it.
        pixels = np.full((16, 16, 3), [128, 0, 255], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{attribute}.png"
            Image.fromarray(pixels).save(path)
            setattr(material, slot, app.load_texture(path, srgb=False))
            file_image = capture()
            setattr(material, slot, app.create_texture(pixels, srgb=False))
            mapped = capture()
            assert np.abs(mapped - file_image).mean() < 2

        setattr(material, slot, None)
        setattr(material, attribute, factor * 128 / 255)
        scalar = capture()
        assert np.abs(mapped - scalar).mean() < 2
        setattr(material, attribute, factor)
        assert np.abs(capture() - mapped).mean() > 0.5

        texture = app.create_texture(pixels, srgb=False)
        setattr(material, slot, texture)
        setattr(material, attribute, factor * 0.5)
        scaled = capture()
        setattr(material, slot, None)
        setattr(material, attribute, factor * 0.5 * 128 / 255)
        assert np.abs(capture() - scaled).mean() < 2

        setattr(material, slot, texture)
        material.unlit = True
        unlit = capture()
        setattr(material, slot, None)
        assert np.abs(capture() - unlit).mean() < 2
        material.unlit = False
        setattr(material, attribute, factor)

    # Both maps can share one image while keeping independent scalar factors.
    material.metallic_texture = texture
    material.roughness_texture = texture
    capture()
    with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
        capture()
        assert not any(call.args[1] == "texture:data" for call in writes.call_args_list)
    material.metallic_texture = None
    material.roughness_texture = None
    capture()
    assert not adapter._textures


def check_emissive(app, adapter, view, capture):
    material = app.create_pbr_material(
        base_color=[0, 0, 0, 1], emissive_color=[1, 1, 1], emissive_strength=1,
    )
    view.set_material(material)
    view.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
    app.set_light_intensity(0)
    pixels = np.full((16, 16, 3), [255, 0, 0], dtype=np.uint8)
    pixels[:8] = [0, 0, 255]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "emissive.png"
        Image.fromarray(pixels).save(path)
        material.emissive_texture = app.load_texture(path, srgb=True)
        file_image = capture()
        material.emissive_texture = app.create_texture(pixels, srgb=True)
        mapped = capture()
        assert np.abs(mapped - file_image).mean() < 2
    assert np.count_nonzero(mapped[..., 0] > mapped[..., 2] + 50) > 50
    assert np.count_nonzero(mapped[..., 2] > mapped[..., 0] + 50) > 50

    material.emissive_strength = 0
    assert capture()[96, 128].max() < 15
    # Keep the strength comparison below the output's saturation range.
    app.set_exposure(0.001)
    material.emissive_strength = 0.25
    dim = capture()
    material.emissive_strength = 1
    bright = capture()
    assert bright.mean() > dim.mean() + 1, (dim.mean(), bright.mean())
    app.set_exposure(1)
    material.emissive_color = [1, 0, 0]
    assert capture()[..., 2].max() < 15
    material.emissive_color = [1, 1, 1]
    material.emissive_texture = None
    assert capture()[96, 128].min() > 200
    assert not adapter._textures

    green = app.create_texture(np.full((8, 8, 3), [0, 255, 0], dtype=np.uint8))
    material.emissive_texture = green
    assert capture()[96, 128, 1] > 200
    with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
        capture()
        assert not any(call.args[1] == "texture:data" for call in writes.call_args_list)

    # Unlit uses base color, regardless of the separate emissive map.
    material.base_color = [1, 0, 0, 1]
    material.unlit = True
    unlit = capture()[96, 128]
    assert unlit[0] > 200 and unlit[1] < 15
    material.unlit = False
    material.base_color = [0, 0, 0, 1]
    assert capture()[96, 128, 1] > 200

    # External instances share emission without tinting it by instance color.
    view.set_visible(False)
    batch = app.scene.add_mesh(
        "/emissive_batch", ke.geometry.create_cube_data(1), material,
        transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
    )
    matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    matrices[:, 3, 0] = [-0.8, 0.8]
    system = app.get_scene_render_system()
    system.update_instances(batch.component, matrices)
    system.set_instance_colors(batch.component, [[1, 0, 0, 1], [0, 0, 1, 1]])
    instances = capture()
    assert np.count_nonzero(instances[:, :128, 1] > 200) > 50
    assert np.count_nonzero(instances[:, 128:, 1] > 200) > 50
    material.emissive_strength = 0
    assert capture()[..., 1].max() < 15
    batch.remove()
    view.set_visible(True)


def main():
    app = ke.App()
    adapter = OvrtxAppAdapter(256, 192)
    try:
        app.render_backend = "ovrtx"
        app._initialize_scene_renderer(256, 192, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
                                       lambda width, height: adapter)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        app.set_light_direction([0, 0, -1])
        app.set_light_ambient([0, 0, 0])
        app.set_light_intensity(3)
        material = app.create_pbr_material(roughness=1)
        view = app.scene.add_mesh("/cube", ke.geometry.create_cube_data(2), material)

        def capture():
            for _ in range(16):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        # The same source color remains visible without scene lights.
        for material in (material, app.create_color_material(),
                         app.create_phong_material(diffuse=[1, 1, 1], specular=[0, 0, 0])):
            view.set_material(material)
            material.unlit = True
            app.set_light_intensity(0)
            assert capture()[96, 128].min() > 200
            material.unlit = False
            assert capture()[96, 128].max() < 15
            app.set_light_intensity(3)

        material = app.create_pbr_material(roughness=1)
        material.unlit = True
        view.set_material(material)
        pixels = np.full((16, 16, 4), [255, 0, 0, 255], dtype=np.uint8)
        pixels[:8] = [0, 0, 255, 255]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "color.png"
            Image.fromarray(pixels).save(path)
            material.base_color_texture = app.load_texture(path, srgb=True)
            file_image = capture()
            material.base_color_texture = app.create_texture(pixels, srgb=True)
            memory_image = capture()
            assert np.abs(memory_image - file_image).mean() < 2, np.abs(memory_image - file_image).mean()
        with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
            capture()
            assert not any(call.args[1] == "texture:data" for call in writes.call_args_list)
        color_material = app.create_color_material(style=ke.material.ColorStyle.TEXTURED)
        color_material.base_color_texture = material.base_color_texture
        view.set_material(color_material)
        textured = capture()
        color_material.set_style(ke.material.ColorStyle.UNTEXTURED)
        untextured = capture()
        assert np.abs(textured - untextured).mean() > 2
        view.set_material(material)
        material.base_color_texture = None
        material.base_color = [1, 0, 0, 0.25]
        view.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
        opaque = capture()[96, 128, 0]
        view.set_alpha_mode(ke.render.AlphaMode.MASK, 0.5)
        assert capture()[96, 128, 0] < 5
        view.set_alpha_mode(ke.render.AlphaMode.BLEND)
        try:
            capture()
        except ValueError as error:
            assert "AlphaMode.BLEND" in str(error)
        else:
            raise AssertionError("Unlit fading was accepted as USD opacity")
        view.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
        view.prim.set_display_color_alpha(ke.Vec4(1, 1, 1, 0.5))
        try:
            capture()
        except ValueError as error:
            assert "alpha blending" in str(error)
        else:
            raise AssertionError("Legacy display-alpha blending was silently ignored")
        view.prim.set_display_color_alpha(ke.Vec4(1, 1, 1, 1))
        material.base_color = [0.2, 0.2, 0.2, 1]
        material.unlit = False
        app.set_light_intensity(1)
        plain = capture()
        normal = np.full((8, 8, 3), [240, 128, 170], dtype=np.uint8)
        material.normal_texture = app.create_texture(normal, srgb=False)
        mapped = capture()
        assert np.abs(mapped - plain).mean() > 1, (plain[96, 128], mapped[96, 128])
        phong = app.create_phong_material(diffuse=[1, 1, 1])
        phong.unlit = True
        phong.alpha_map = app.create_texture(np.zeros((8, 8, 3), dtype=np.uint8), srgb=False)
        view.set_material(phong)
        view.set_alpha_mode(ke.render.AlphaMode.MASK)
        assert capture()[96, 128].max() < 5
        phong.alpha_map = None
        assert capture()[96, 128].min() > 200
        check_opacity(app, view, capture)
        check_pbr_maps(app, adapter, view, capture)
        check_emissive(app, adapter, view, capture)
        view.remove()
        capture()
        assert not adapter._textures and not adapter._texture_pixels
        print("PASS: Color/Phong/PBR, unlit, alpha mask, PBR maps, and texture lifetime")
    finally:
        app.close()


if __name__ == "__main__":
    main()
