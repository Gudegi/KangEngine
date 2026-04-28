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
        device = self.getGraphicsDevice()

        vs = asset_path("shaders", "common.vs")
        fs = asset_path("shaders", "common.fs")
        checker_fs = asset_path("shaders", "checkerboard.fs")

        self.obj_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)

        for shader in (self.obj_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.6, 0.9, 0.6, 1.0))

        sc = self.getScene()

        # Ground plane (Y-up)
        gnd = sc.define_prim("/ground", scene.PrimType.Mesh)
        gnd.set_mesh_data(scene.Prim.create_plane_data(30.0))
        self.addShape(self.ground_shader, gnd)

        # Box
        box = sc.define_prim("/box", scene.PrimType.Mesh)
        box.set_mesh_data(scene.Prim.create_square_data(1.0))
        box.set_display_color_alpha(ke.vec4(0.8, 0.3, 0.02, 1.0))
        box.add_translate_op(ke.vec3(0.0, 2.0, 0.0))
        self.addShape(self.obj_shader, box)

        # Sphere
        sphere = sc.define_prim("/sphere", scene.PrimType.Mesh)
        sphere.set_mesh_data(scene.Prim.create_sphere_data(0.5, 16, 12))
        sphere.set_display_color_alpha(ke.vec4(0.2, 0.4, 0.9, 1.0))
        sphere.add_translate_op(ke.vec3(2.5, 0.5, 0.0))
        self.addShape(self.obj_shader, sphere)

        self.checkError()

    def preRender(self):
        self.checkError()

    def render(self):
        pass

    def postRender(self):
        pass


if __name__ == "__main__":
    app = MyApp()
    app.initialize(1920, 1080, False, ke.UpAxis.Y)
    app.start()
