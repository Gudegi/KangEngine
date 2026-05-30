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

        device = self.getGraphicsDevice()
        vs = package_asset_path("shaders", "common.vs")
        tex_fs = package_asset_path("shaders", "commonTex.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.wall_shader = device.createShaderFromFile(vs, tex_fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)

        for shader in (self.wall_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)
            shader.setInt("normalDebugMode", 0)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.62, 0.82, 0.68, 1.0))

        root = brickwall_dir()
        self.diffuse_texture = device.createTexture(str(root / "brickwall.jpg"), True)
        self.normal_texture = device.createTexture(str(root / "brickwall_normal.jpg"), True)

        wall = self.getScene().define_prim("/brickwall", scene.PrimType.Mesh)
        wall.set_mesh_data(scene.Prim.create_plane_data(4.0, ke.UpAxis.Z))
        wall.set_display_color_alpha(ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.wall_handle = self.addShape(self.wall_shader, wall)
        self.setShapeTexture(self.wall_handle, self.diffuse_texture, 0)
        self.setShapeTexture(self.wall_handle, self.normal_texture, 5)
        self.setShapeDoubleSided(self.wall_handle, True)

        ground = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(6.0, ke.UpAxis.Y))
        ground.add_translate_op(ke.vec3(0.0, -2.0, 0.0))
        self.addShape(self.ground_shader, ground)

        self.setLightDirection(ke.vec3(-0.45, 0.35, 0.82))
        self.setLightColor(ke.vec3(1.0, 0.96, 0.88))
        self.setLightIntensity(1.4)
        self.setLightAmbient(ke.vec3(0.22, 0.22, 0.22))

        camera = self.getCamera()
        camera.set_camera_pos(ke.vec3(0.0, 0.0, 5.0))
        camera.set_target_pos(ke.vec3(0.0, 0.0, 0.0))
        camera.set_near_plane(0.01)
        camera.set_far_plane(50.0)
        camera.set_fov(45.0)
        self.set_camera_move_speed(1.0)

        print("Brickwall normal map test loaded")
        print(f"  diffuse: {root / 'brickwall.jpg'}")
        print(f"  normal : {root / 'brickwall_normal.jpg'}")
        self.checkError()

    def preRender(self):
        self.checkError()

    def render(self):
        imgui.begin("Brickwall Normal Map")
        changed, self.normal_maps_enabled = imgui.checkbox(
            "normal map", self.normal_maps_enabled
        )
        if changed:
            self.setShapeTexture(
                self.wall_handle,
                self.normal_texture if self.normal_maps_enabled else None,
                5,
            )
        if imgui.button(f"debug: {NORMAL_DEBUG_MODES[self.normal_debug_mode]}"):
            self.normal_debug_mode = (self.normal_debug_mode + 1) % len(
                NORMAL_DEBUG_MODES
            )
            self.wall_shader.use()
            self.wall_shader.setInt("normalDebugMode", self.normal_debug_mode)
        imgui.text("Use the debug modes to inspect tangent-space data.")
        imgui.end()

    def postRender(self):
        pass


if __name__ == "__main__":
    app = BrickwallNormalMapViewer()
    app.initialize(1280, 720, False, ke.UpAxis.Y)
    app.start()
