"""Small HDR bloom scene."""

from __future__ import annotations

import math

import kangengine as ke
from kangengine import imgui, scene


class BloomViewer(ke.App):
    def setup(self):
        self.bloom_enabled = True
        self.bloom_threshold = 1.0
        self.bloom_intensity = 0.55
        self.bloom_iterations = 8
        self.bloom_downsample = 2
        self.exposure = 1.0

        self.shaders = self.create_standard_shaders()
        self.add_ground(shader=self.shaders.ground, scale=18.0)

        self._add_glow_sphere(
            "/glow/core", [-1.4, 0.0, 1.2], 0.45, [70.0, 4.8, 1.3, 1.0]
        )
        self._add_glow_sphere(
            "/glow/cyan", [0.0, 0.0, 1.0], 0.35, [1.0, 5.5, 7.0, 1.0]
        )
        self._add_glow_sphere(
            "/glow/pink", [1.2, 0.0, 1.4], 0.28, [6.5, 1.2, 4.8, 1.0]
        )

        for i in range(14):
            x = -3.5 + i * 0.55
            z = 0.08 + 0.02 * math.sin(i * 1.7)
            color = [0.24 + i * 0.02, 0.22, 0.28, 1.0]
            self._add_sphere(f"/beads/{i}", [x, 1.3, z], 0.08, color)

        self.set_light_direction(ke.vec3(-0.35, 0.65, 0.55))
        self.set_light_color(ke.vec3(1.0, 0.94, 0.84))
        self.set_light_intensity(0.45)
        self.set_light_ambient(ke.vec3(0.035, 0.04, 0.05))
        self.set_tone_map(ke.render.ToneMapMode.AcesNarkowicz, self.exposure)
        self._apply_bloom()

        self.set_camera_view([0.0, -5.2, 2.4], [0.0, 0.0, 0.9])
        self.set_camera_move_speed(1.5)
        self.check_error()

    def _add_sphere(self, path, pos, radius, color):
        view = self.scene.add_mesh(
            path,
            ke.geometry.create_sphere_data(radius, 32, 16),
            self.shaders.common,
            color=ke.vec4(*color),
        )
        view.prim.add_translate_op(ke.vec3(*pos))
        return view

    def _add_glow_sphere(self, path, pos, radius, color):
        view = self._add_sphere(path, pos, radius, color)
        # view.set_casts_shadow(False)
        return view

    def _apply_bloom(self):
        self.set_bloom(
            self.bloom_enabled,
            self.bloom_threshold,
            self.bloom_intensity,
            self.bloom_iterations,
            self.bloom_downsample,
        )

    def render(self):
        imgui.begin("Bloom")
        changed, self.bloom_enabled = imgui.checkbox("enabled", self.bloom_enabled)
        if changed:
            self._apply_bloom()

        changed, self.bloom_threshold = imgui.slider_float(
            "threshold", self.bloom_threshold, 0.0, 5.0
        )
        if changed:
            self._apply_bloom()

        changed, self.bloom_intensity = imgui.slider_float(
            "intensity", self.bloom_intensity, 0.0, 2.0
        )
        if changed:
            self._apply_bloom()

        changed, self.exposure = imgui.slider_float("exposure", self.exposure, 0.1, 3.0)
        if changed:
            self.set_tone_map(ke.render.ToneMapMode.AcesNarkowicz, self.exposure)

        imgui.text("Bright spheres use HDR colors above 1.0.")
        imgui.end()


if __name__ == "__main__":
    app = BloomViewer()
    app.initialize(1280, 720, False, ke.UpAxis.Z)
    app.start()
