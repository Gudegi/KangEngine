from __future__ import annotations

import kangengine as ke
from kangengine import imgui, scene


ROUGHNESS_VALUES = (0.04, 0.18, 0.38, 0.68)


class PBRStudioViewer(ke.App):
    def setup(self):
        self.exposure = 1.0
        self.use_aces = True

        floor = self.create_pbr_material(
            base_color=ke.Vec4(0.18, 0.18, 0.18, 1.0),
            metallic=0.0,
            roughness=0.82,
        )
        self.add_ground(scale=14.0, material=floor)

        self._add_material_row(
            "chromium",
            y=0.55,
            base_color=(0.654, 0.685, 0.701),
            metallic=1.0,
        )
        self._add_material_row(
            "dielectric",
            y=-0.75,
            base_color=(0.55, 0.12, 0.055),
            metallic=0.0,
        )
        self._add_material_row(
            "blue_plastic",
            y=-2.05,
            base_color=(0.055, 0.19, 0.52),
            metallic=0.0,
        )
        self._configure_studio_lights()
        self._apply_tone_map()

        self.set_camera_view([0.0, -8.4, 3.7], [0.0, -0.6, 0.72])
        self.set_camera_move_speed(1.5)
        self.check_error()

    def _add_material_row(self, name, y, base_color, metallic):
        sphere = ke.geometry.create_sphere_data(0.43, 64, 32)
        for index, roughness in enumerate(ROUGHNESS_VALUES):
            material = self.create_pbr_material(
                base_color=ke.Vec4(*base_color, 1.0),
                metallic=metallic,
                roughness=roughness,
            )
            x = (index - 1.5) * 1.25
            view = self.scene.add_mesh(
                f"/studio/{name}/roughness_{index}",
                sphere,
                material,
            )
            view.prim.set_local_translation(ke.Vec3(x, y, 0.58))

    def _configure_studio_lights(self):
        # Broad, neutral base illumination. Point/spot lights below provide
        # highlight shapes because rectangular area lights are not available.
        self.set_light_direction(ke.Vec3(-0.38, 0.48, 0.79))
        self.set_light_color(ke.Vec3(1.0, 0.98, 0.95))
        self.set_light_intensity(0.65)
        self.set_light_ambient(ke.Vec3(0.035, 0.035, 0.035))

        key = scene.SpotLight()
        key.position = ke.Vec3(-2.8, -3.2, 4.2)
        key.direction = ke.Vec3(0.48, 0.62, -0.62)
        key.color = ke.Vec3(1.0, 0.93, 0.84)
        key.intensity = 115.0
        key.range = 9.0
        key.inner_cone_angle = 0.52
        key.outer_cone_angle = 0.92
        self.scene.define_prim("/lights/key", scene.PrimType.LIGHT).set_spot_light(key)

        fill = scene.PointLight()
        fill.position = ke.Vec3(3.0, -1.0, 2.6)
        fill.color = ke.Vec3(0.72, 0.82, 1.0)
        fill.intensity = 34.0
        fill.range = 8.0
        self.scene.define_prim("/lights/fill", scene.PrimType.LIGHT).set_point_light(
            fill
        )

        rim = scene.PointLight()
        rim.position = ke.Vec3(0.0, 3.0, 2.8)
        rim.color = ke.Vec3(1.0, 0.96, 0.9)
        rim.intensity = 48.0
        rim.range = 7.0
        self.scene.define_prim("/lights/rim", scene.PrimType.LIGHT).set_point_light(rim)

    def _apply_tone_map(self):
        mode = (
            ke.render.ToneMapMode.ACES_NARKOWICZ
            if self.use_aces
            else ke.render.ToneMapMode.OFF
        )
        self.set_tone_map(mode, self.exposure)

    def render(self):
        imgui.begin("PBR Studio")
        imgui.text("Rows: chromium / red dielectric / blue plastic")
        imgui.text("Roughness: 0.04 / 0.18 / 0.38 / 0.68")
        imgui.text("No HDRI; directional + spot + two point lights")
        changed, self.use_aces = imgui.checkbox("ACES tone mapping", self.use_aces)
        exposure_changed, self.exposure = imgui.slider_float(
            "Exposure", self.exposure, 0.2, 2.5
        )
        if changed or exposure_changed:
            self._apply_tone_map()
        imgui.end()


if __name__ == "__main__":
    app = PBRStudioViewer()
    app.initialize(1600, 900, False, ke.UpAxis.Z)
    app.start()
