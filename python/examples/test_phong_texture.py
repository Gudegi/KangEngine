"""PhongMaterial diffuse-map smoke viewer.

Shows the same diffuse texture twice:

- left: white material diffuse, so the texture is shown unchanged
- right: red-tinted material diffuse, so the texture is multiplied by diffuse

Useful after changing ``assets/shaders/phong.fs`` because this exercises the
Material-owned diffuse_map path rather than the legacy ``view.set_texture``
path used by commonTex.fs examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
from kangengine import imgui, scene


class PhongTextureViewer(ke.App):
    def __init__(self):
        super().__init__()
        self.textures = []

    def setup(self):
        self.create_standard_shaders()

        device = self.get_renderer().device()
        texture = device.create_texture(
            self.package_asset_path("textures", "awesomeface.png"),
            True,
        )
        self.textures.append(texture)

        white = self.create_phong_material(
            shader=self.shaders.phong,
            diffuse=ke.vec3(1.0, 1.0, 1.0),
            specular=ke.vec3(0.05, 0.05, 0.05),
            shininess=16.0,
            diffuse_map=texture,
        )
        tinted = self.create_phong_material(
            shader=self.shaders.phong,
            diffuse=ke.vec3(1.0, 0.35, 0.35),
            specular=ke.vec3(0.05, 0.05, 0.05),
            shininess=16.0,
            diffuse_map=texture,
        )

        mesh_data = ke.geometry.create_plane_data(1.8, ke.UpAxis.Z)
        left = self.scene.add_mesh(
            "/phong_texture/white_diffuse",
            mesh_data,
            white,
            color=ke.vec4(1.0, 1.0, 1.0, 1.0),
        )
        left.prim.set_local_translation(ke.vec3(-1.15, 0.0, 0.0))
        left.set_alpha_mode(ke.render.AlphaMode.Blend)

        right = self.scene.add_mesh(
            "/phong_texture/red_tint",
            mesh_data,
            tinted,
            color=ke.vec4(1.0, 1.0, 1.0, 1.0),
        )
        right.prim.set_local_translation(ke.vec3(1.15, 0.0, 0.0))
        right.set_alpha_mode(ke.render.AlphaMode.Blend)

        self.set_light_direction(ke.vec3(0.0, 0.0, 1.0))
        self.set_light_color(ke.vec3(1.0, 1.0, 1.0))
        self.set_light_intensity(1.0)
        self.set_light_ambient(ke.vec3(0.35, 0.35, 0.35))
        self.set_tone_map(ke.render.ToneMapMode.Off, 1.0)
        self.set_bloom(False)
        self.set_camera_view([0.0, 0.0, 4.0], [0.0, 0.0, 0.0])
        self.set_camera_move_speed(1.0)

    def render(self):
        imgui.begin("Phong Texture Smoke")
        imgui.text("left: diffuse=(1,1,1), texture unchanged")
        imgui.text("right: diffuse=(1,0.35,0.35), same texture tinted")
        imgui.end()


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    app = PhongTextureViewer()
    app.initialize(900, 500, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
