"""Check material models, unlit color, alpha, normal maps, and memory images."""

from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import kangengine as ke
from kangengine.render._ovrtx_scene import OvrtxAppAdapter


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
            raise AssertionError("USD transmission was accepted as alpha blending")
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
        view.remove()
        capture()
        assert not adapter._textures and not adapter._texture_pixels
        print("PASS: Color/Phong/PBR, unlit, alpha mask, blend rejection, normal map, and memory texture lifetime")
    finally:
        app.close()


if __name__ == "__main__":
    main()
