"""
Render Prim Scene — Python equivalent of test_prim_scene.cpp.
Demonstrates the Prim scene graph with common UBO-based shaders.
"""

import os
import kangengine as ke
from kangengine import scene


def asset_path(*parts):
    base = os.path.join(os.path.dirname(ke.__file__), "assets")
    return os.path.join(base, *parts)


class MyApp(ke.App):
    def setup(self):
        device = self.get_renderer().device()

        vs = asset_path("shaders", "common.vs")
        fs = asset_path("shaders", "common.fs")
        checker_fs = asset_path("shaders", "checkerboard.fs")

        self.obj_shader = device.create_shader_from_file(vs, fs)
        self.ground_shader = device.create_shader_from_file(vs, checker_fs)

        for shader in (self.obj_shader, self.ground_shader):
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)

        self.ground_shader.use()
        self.ground_shader.set_vec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.set_vec4("checkerColor2", ke.vec4(0.6, 0.9, 0.6, 1.0))

        # Ground plane (Y-up)
        self.scene.add_mesh(
            "/ground",
            scene.Prim.create_plane_data(30.0),
            self.ground_shader,
        )

        # Box
        box = self.scene.add_mesh(
            "/box",
            scene.Prim.create_square_data(1.0),
            self.obj_shader,
            color=ke.vec4(0.8, 0.3, 0.02, 1.0),
        )
        box.prim.add_translate_op(ke.vec3(0.0, 2.0, 0.0))

        # Sphere
        sphere = self.scene.add_mesh(
            "/sphere",
            scene.Prim.create_sphere_data(0.5, 16, 12),
            self.obj_shader,
            color=ke.vec4(0.2, 0.4, 0.9, 1.0),
        )
        sphere.prim.add_translate_op(ke.vec3(2.5, 0.5, 0.0))

        self.check_error()

    def preRender(self):
        self.check_error()

    def render(self):
        pass

    def postRender(self):
        pass


if __name__ == "__main__":
    app = MyApp()
    app.initialize(1920, 1080, False, ke.UpAxis.Y)
    app.start()
