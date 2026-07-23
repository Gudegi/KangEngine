"""Normal map test using assets/external/brickwall."""

from __future__ import annotations

from pathlib import Path

import kangengine as ke
from kangengine import imgui, scene


NORMAL_DEBUG_MODES = [
    "off",
    "vertex normal",
    "tangent",
    "normal map",
    "final normal",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def brickwall_dir() -> Path:
    return repo_root() / "assets" / "external" / "brickwall"


class BrickwallNormalMapViewer(ke.App):
    def setup(self):
        self.normal_maps_enabled = True
        self.normal_debug_mode = 0

        device = self.get_renderer().device()
        vs = package_asset_path("shaders", "common.vs")
        tex_fs = package_asset_path("shaders", "commonTex.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.wall_shader = device.create_shader_from_file(vs, tex_fs)
        self.ground_shader = device.create_shader_from_file(vs, checker_fs)

        for shader in (self.wall_shader, self.ground_shader):
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)
            shader.set_uniform_block_binding("shadowUBO", 2)
            shader.set_int("normalDebugMode", 0)

        self.ground_shader.use()
        self.ground_shader.set_vec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.set_vec4("checkerColor2", ke.vec4(0.62, 0.82, 0.68, 1.0))

        root = brickwall_dir()
        self.diffuse_texture = device.create_texture(str(root / "brickwall.jpg"), True)
        self.normal_texture = device.create_texture(str(root / "brickwall_normal.jpg"), True)

        self.wall_view = self.scene.add_mesh(
            "/brickwall",
            ke.geometry.create_plane_data(4.0, ke.UpAxis.Z),
            self.wall_shader,
            color=ke.vec4(1.0, 1.0, 1.0, 1.0),
        )
        self.wall_view.set_texture(self.diffuse_texture, 0)
        self.wall_view.set_texture(self.normal_texture, 5)
        self.wall_view.set_double_sided(True)

        ground_view = self.scene.add_mesh(
            "/ground",
            ke.geometry.create_plane_data(6.0, ke.UpAxis.Y),
            self.ground_shader,
        )
        ground_view.prim.set_local_translation(ke.vec3(0.0, -2.0, 0.0))

        self.set_light_direction(ke.vec3(-0.45, 0.35, 0.82))
        self.set_light_color(ke.vec3(1.0, 0.96, 0.88))
        self.set_light_intensity(1.4)
        self.set_light_ambient(ke.vec3(0.22, 0.22, 0.22))

        camera = self.get_camera()
        camera.set_camera_pos(ke.vec3(0.0, 0.0, 5.0))
        camera.set_target_pos(ke.vec3(0.0, 0.0, 0.0))
        camera.set_near_plane(0.01)
        camera.set_far_plane(50.0)
        camera.set_fov(45.0)
        self.set_camera_move_speed(1.0)

        print("Brickwall normal map test loaded")
        print(f"  diffuse: {root / 'brickwall.jpg'}")
        print(f"  normal : {root / 'brickwall_normal.jpg'}")
        self.check_error()

    def preRender(self):
        self.check_error()

    def render(self):
        imgui.begin("Brickwall Normal Map")
        changed, self.normal_maps_enabled = imgui.checkbox(
            "normal map", self.normal_maps_enabled
        )
        if changed:
            self.wall_view.set_texture(
                self.normal_texture if self.normal_maps_enabled else None,
                5,
            )
        if imgui.button(f"debug: {NORMAL_DEBUG_MODES[self.normal_debug_mode]}"):
            self.normal_debug_mode = (self.normal_debug_mode + 1) % len(
                NORMAL_DEBUG_MODES
            )
            self.wall_shader.use()
            self.wall_shader.set_int("normalDebugMode", self.normal_debug_mode)
        imgui.text("Use the debug modes to inspect tangent-space data.")
        imgui.end()

    def postRender(self):
        pass


if __name__ == "__main__":
    app = BrickwallNormalMapViewer()
    app.initialize(1280, 720, False, ke.UpAxis.Y)
    app.start()
