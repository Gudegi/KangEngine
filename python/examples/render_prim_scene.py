"""
Render Prim Scene — Python equivalent of test_prim_scene.cpp.
Demonstrates the material-first Prim scene graph.
"""

import numpy as np

import kangengine as ke


class MyApp(ke.App):
    def setup(self):
        self.standard_materials = self.create_standard_materials()

        ke.utils.log_debug_axes(
            self,
            "/debug/box_axes",  # is not shown in the scene graph.
            origin=np.array([0.0, 1.0, 0.0]),
            rotation=np.eye(3),
            length=1.0,
            width=5.0,
        )

        # Ground plane (Y-up)
        self.scene.add_ground("/ground", scale=30.0)

        # Box
        box = self.scene.add_mesh(
            "/box",
            ke.geometry.create_cube_data(1.0),
            self.standard_materials.common,
            color=ke.vec4(0.8, 0.3, 0.02, 1.0),
        )
        box.prim.set_local_translation(ke.vec3(0.0, 2.0, 0.0))
        box.prim.set_local_rotation_axis_angle(
            ke.vec3(0.0, 1.0, 0.0), np.deg2rad(25.0)
        )

        # Box2
        box2 = self.scene.add_mesh(
            "/box/box2",
            ke.geometry.create_cube_data(1.0),
            self.standard_materials.common,
            color=ke.vec4(0.3, 0.3, 0.02, 1.0),
        )
        box2.prim.set_local_translation(ke.vec3(0.0, 1.5, 0.0))
        box2.prim.set_local_rotation(np.array([0.924, 0, 0, 0.383]))
        box2.prim.set_local_scale(ke.vec3(0.5, 0.5, 0.5))

        w_trans = box2.prim.get_world_translation()
        w_ori = box2.prim.get_world_rotation()
        ke.scene.DebugDraw.log_coordinate_axes(
            self,
            self.standard_materials.common.shader,
            "/debug/box2_axes",
            w_trans,
            w_ori,
            length=0.8,
            radius=0.01,
            segments=8,
        )

        print(box2.prim.get_local_translation())
        print(box2.prim.get_local_rotation())
        print(box2.prim.get_local_rotation().to_wxyz())
        print(box2.prim.get_local_rotation().to_xyzw())
        print(box2.prim.get_world_translation())
        print(np.array(w_ori))

        # Sphere
        sphere = self.scene.add_mesh(
            "/sphere",
            ke.geometry.create_sphere_data(0.5, 16, 12),
            self.standard_materials.common,
            color=ke.vec4(0.2, 0.4, 0.9, 1.0),
        )
        sphere.prim.set_local_translation(ke.vec3(2.5, 0.5, 0.0))

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
