"""View the oak wood PBR texture set."""

from __future__ import annotations

from pathlib import Path

import kangengine as ke
from kangengine import imgui, scene


ROOT = Path("assets/external/PBR/woods/oak-wood-bare-bl")


class PBROakWoodViewer(ke.App):
    def setup(self):
        self.shaders = self.create_standard_shaders()
        self.textures = []
        self.texture_flags = {
            "albedo": True,
            "normal": True,
            "metallic": True,
            "roughness": True,
            "ao": True,
        }
        self.oak_textures = {}

        self.material = self._make_oak_material()
        oak_view = self.scene.add_mesh(
            "/pbr/oak_wood",
            ke.geometry.create_plane_data(3.0, ke.UpAxis.Z),
            self.material,
        )
        oak_view.prim.add_translate_op(ke.vec3(0.0, 0.0, 0.02))

        self.set_light_direction(ke.vec3(-0.35, 0.45, 0.82))
        self.set_light_color(ke.vec3(1.0, 1.0, 1.0))
        self.set_light_intensity(1.2)
        self.set_light_ambient(ke.vec3(0.22, 0.22, 0.22))
        self.set_tone_map(ke.render.ToneMapMode.AcesNarkowicz, 1.0)
        self.set_bloom(False)

        self.set_camera_view([0.0, -3.0, 2.2], [0.0, 0.0, 0.15])
        self.set_camera_move_speed(1.4)
        self.check_error()

    def _load_texture(self, name):
        texture = self.get_renderer().device().create_texture(str(ROOT / name), True)
        self.textures.append(texture)
        return texture

    def _make_oak_material(self):
        self.oak_textures = {
            "albedo": self._load_texture("oak-wood-bare_albedo.png"),
            "normal": self._load_texture("oak-wood-bare_normal-ogl.png"),
            "metallic": self._load_texture("oak-wood-bare_metallic.png"),
            "roughness": self._load_texture("oak-wood-bare_roughness.png"),
            "ao": self._load_texture("oak-wood-bare_ao.png"),
        }
        material = self.create_pbr_material(metallic=1.0, roughness=1.0)
        self._apply_texture_flags(material)
        return material

    def _apply_texture_flags(self, material=None):
        if material is None:
            material = self.material
        material.base_color_texture = (
            self.oak_textures["albedo"] if self.texture_flags["albedo"] else None
        )
        material.normal_texture = (
            self.oak_textures["normal"] if self.texture_flags["normal"] else None
        )
        material.metallic_texture = (
            self.oak_textures["metallic"] if self.texture_flags["metallic"] else None
        )
        material.roughness_texture = (
            self.oak_textures["roughness"] if self.texture_flags["roughness"] else None
        )
        material.ao_texture = self.oak_textures["ao"] if self.texture_flags["ao"] else None

    def render(self):
        imgui.begin("PBR Oak Wood")
        imgui.text("Texture set: oak-wood-bare-bl")
        changed = False
        for key, label in [
            ("albedo", "albedo"),
            ("normal", "normal"),
            ("metallic", "metallic"),
            ("roughness", "roughness"),
            ("ao", "ao"),
        ]:
            item_changed, self.texture_flags[key] = imgui.checkbox(
                label,
                self.texture_flags[key],
            )
            changed = changed or item_changed
        if changed:
            self._apply_texture_flags()
        imgui.end()


if __name__ == "__main__":
    app = PBROakWoodViewer()
    app.initialize(1600, 1000, False, ke.UpAxis.Z)
    app.start()
