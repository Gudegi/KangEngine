"""Minimal forward PBR material viewer."""

from __future__ import annotations

import kangengine as ke
from kangengine import imgui, scene


class PBRViewer(ke.App):
    def setup(self):
        self.shaders = self.create_standard_shaders()
        self.local_lights_enabled = True
        self.local_light_prims = []

        self.add_ground(shader=self.shaders.ground, scale=16.0)
        self._build_material_grid()
        self._add_emissive_sphere()
        self._configure_local_lights()

        self.set_light_direction(ke.vec3(-0.45, 0.55, 0.72))
        self.set_light_color(ke.vec3(1.0, 0.96, 0.88))
        self.set_light_intensity(0.45)
        self.set_light_ambient(ke.vec3(0.025, 0.028, 0.032))
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
        return self.create_pbr_material(preset, roughness=roughness)

    def _add_pbr_sphere(self, path, position, material, radius=0.34):
        view = self.scene.add_mesh(
            path,
            scene.Prim.create_sphere_data(radius, 48, 24),
            material,
        )
        view.prim.add_translate_op(ke.vec3(*position))
        return view

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
        view = self._add_pbr_sphere(
            "/pbr/emissive",
            [2.45, -1.7, 0.9],
            material,
            0.28,
        )
        view.set_casts_shadow(False)

    def _configure_local_lights(self):
        warm_light = ke.PointLight()
        warm_light.position = ke.vec3(-1.45, -1.35, 1.25)
        warm_light.color = ke.vec3(1.0, 0.55, 0.28)
        warm_light.intensity = 85.0
        warm_light.range = 5.0
        warm_prim = self.scene.define_prim(
            "/lights/warm_point", scene.PrimType.Light
        )
        warm_prim.set_point_light(warm_light)

        cool_light = ke.PointLight()
        cool_light.position = ke.vec3(1.45, 0.95, 1.15)
        cool_light.color = ke.vec3(0.25, 0.55, 1.0)
        cool_light.intensity = 70.0
        cool_light.range = 4.8
        cool_prim = self.scene.define_prim(
            "/lights/cool_point", scene.PrimType.Light
        )
        cool_prim.set_point_light(cool_light)

        spot_light = ke.SpotLight()
        spot_light.position = ke.vec3(0.0, -3.2, 2.8)
        spot_light.direction = ke.vec3(0.0, 0.82, -0.58)
        spot_light.color = ke.vec3(1.0, 0.92, 0.78)
        spot_light.intensity = 90.0
        spot_light.range = 6.0
        spot_light.inner_cone_angle = 0.34
        spot_light.outer_cone_angle = 0.62
        spot_prim = self.scene.define_prim(
            "/lights/soft_spot", scene.PrimType.Light
        )
        spot_prim.set_spot_light(spot_light)

        self.local_light_prims = [warm_prim, cool_prim, spot_prim]
        self._apply_local_lights()

    def _apply_local_lights(self):
        for prim in self.local_light_prims:
            prim.set_visible(self.local_lights_enabled)

    def render(self):
        imgui.begin("PBR")
        imgui.text("Rows: carrot / charcoal / gold")
        imgui.text("Columns: roughness 0.18 / 0.42 / 0.78")
        changed, self.local_lights_enabled = imgui.checkbox(
            "local lights",
            self.local_lights_enabled,
        )
        if changed:
            self._apply_local_lights()
        imgui.text("Local lights: warm point / cool point / soft spot")
        imgui.text("The small blue sphere is emissive and feeds HDR bloom.")
        imgui.end()


if __name__ == "__main__":
    app = PBRViewer()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
