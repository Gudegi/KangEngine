"""PBR preset color checker under neutral lighting."""

from __future__ import annotations

import kangengine as ke
from kangengine import imgui, scene


PRESETS = [
    ("Gray Card", ke.material.PBRMaterialType.GRAY_CARD),
    ("White Plastic", ke.material.PBRMaterialType.WHITE_PLASTIC),
    ("Black Plastic", ke.material.PBRMaterialType.BLACK_PLASTIC),
    ("Black Rubber", ke.material.PBRMaterialType.BLACK_RUBBER),
    ("Charcoal", ke.material.PBRMaterialType.CHARCOAL),
    ("Carrot", ke.material.PBRMaterialType.CARROT),
    ("Concrete", ke.material.PBRMaterialType.CONCRETE),
    ("Red Brick", ke.material.PBRMaterialType.RED_BRICK),
    ("Aluminum", ke.material.PBRMaterialType.ALUMINUM),
    ("Chrome", ke.material.PBRMaterialType.CHROME),
    ("Copper", ke.material.PBRMaterialType.COPPER),
    ("Gold", ke.material.PBRMaterialType.GOLD),
]


class PBRPresetViewer(ke.App):
    def setup(self):
        self.shaders = self.create_standard_shaders()

        self.add_ground(shader=self.shaders.ground, scale=12.0)
        self._build_preset_grid()

        self.set_light_direction(ke.vec3(-0.35, 0.45, 0.82))
        self.set_light_color(ke.vec3(1.0, 1.0, 1.0))
        self.set_light_intensity(0.8)
        self.set_light_ambient(ke.vec3(0.55, 0.55, 0.55))
        self.set_tone_map(ke.render.ToneMapMode.Off, 1.0)
        self.set_bloom(False)

        self.set_camera_view([0.0, -6.4, 4.0], [0.0, 0.0, 0.75])
        self.set_camera_move_speed(1.6)
        self.check_error()

    def _make_material(self, preset):
        return self.create_pbr_material(preset)

    def _add_sphere(self, path, position, material):
        view = self.scene.add_mesh(
            path,
            ke.geometry.create_sphere_data(0.32, 48, 24),
            material,
        )
        view.prim.add_translate_op(ke.vec3(*position))
        return view

    def _build_preset_grid(self):
        columns = 4
        spacing_x = 1.25
        spacing_y = 1.15
        for index, (_, preset) in enumerate(PRESETS):
            row = index // columns
            col = index % columns
            x = (col - (columns - 1) * 0.5) * spacing_x
            y = (1.0 - row) * spacing_y
            material = self._make_material(preset)
            self._add_sphere(f"/pbr_presets/{index}", [x, y, 0.62], material)

    def render(self):
        imgui.begin("PBR Presets")
        imgui.text("Tone map and bloom are off.")
        imgui.text("Neutral white light, high gray ambient.")
        imgui.separator()
        for row_start in range(0, len(PRESETS), 4):
            imgui.text(" / ".join(name for name, _ in PRESETS[row_start : row_start + 4]))
        imgui.end()


if __name__ == "__main__":
    app = PBRPresetViewer()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
