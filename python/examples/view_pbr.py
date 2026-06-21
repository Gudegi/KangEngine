"""Minimal forward PBR material viewer."""

from __future__ import annotations

import kangengine as ke
from kangengine import imgui, scene


class PBRViewer(ke.App):
    def setup(self):
        self.shaders = self.create_standard_shaders()
        self.pbr_shader = self.create_asset_shader("common.vs", "pbr_forward.fs")
        self.materials = []

        self.add_ground(shader=self.shaders.ground, scale=16.0)
        self._build_material_grid()
        self._add_emissive_sphere()

        self.set_light_direction(ke.vec3(-0.45, 0.55, 0.72))
        self.set_light_color(ke.vec3(1.0, 0.96, 0.88))
        self.set_light_intensity(3.0)
        self.set_light_ambient(ke.vec3(0.035, 0.04, 0.05))
        self.set_tone_map(ke.ToneMapMode.AcesNarkowicz, 1.0)
        self.set_bloom(
            True,
            threshold=1.4,
            intensity=0.2,
            iterations=6,
            downsample=2,
        )

        self.set_camera_view([0.0, -7.5, 3.4], [0.0, 0.0, 0.9])
        self.set_camera_move_speed(1.8)
        self.check_error()

    def _make_material(self, preset, roughness=None):
        material = ke.PBRMaterial()
        material.set_shader(self.pbr_shader)
        material.load_from_preset(preset)
        if roughness is not None:
            material.roughness = float(roughness)
        self.materials.append(material)
        return material

    def _add_pbr_sphere(self, path, position, material, radius=0.34):
        prim = self.get_scene().define_prim(path, scene.PrimType.Mesh)
        prim.set_mesh_data(scene.Prim.create_sphere_data(radius, 48, 24))
        prim.add_translate_op(ke.vec3(*position))
        return self.add_renderable(material, prim)

    def _build_material_grid(self):
        presets = [
            ke.PBRMaterialType.CARROT,
            ke.PBRMaterialType.CHARCOAL,
            ke.PBRMaterialType.GOLD,
        ]
        roughness_values = [0.18, 0.42, 0.78]

        for row, preset in enumerate(presets):
            for col, roughness in enumerate(roughness_values):
                material = self._make_material(preset, roughness)
                x = (col - 1) * 1.25
                y = (row - 1) * 1.2
                self._add_pbr_sphere(
                    f"/pbr/m{row}_r{col}",
                    [x, y, 0.65],
                    material,
                )

    def _add_emissive_sphere(self):
        material = self._make_material(ke.PBRMaterialType.EMISSIVE_BLUE)
        handle = self._add_pbr_sphere(
            "/pbr/emissive",
            [2.45, -1.7, 0.9],
            material,
            0.28,
        )
        self.set_renderable_casts_shadow(handle, False)

    def render(self):
        imgui.begin("PBR")
        imgui.text("Rows: carrot / charcoal / gold")
        imgui.text("Columns: roughness 0.18 / 0.42 / 0.78")
        imgui.text("The small blue sphere is emissive and feeds HDR bloom.")
        imgui.end()


if __name__ == "__main__":
    app = PBRViewer()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
