"""Start paused; Enter plays/pauses, Space steps, and R resets the ball."""

import numpy as np
import kangengine as ke

from kangengine.utils._example_args import render_example_parser


class PhysicsInspectorApp(ke.App):
    inspector = None

    def setup(self):
        self.set_ui_layout_mode(mode=ke.UILayoutMode.EDITOR)
        self.world = ke.physics.PhysicsWorld()
        self.set_fixed_update_hz(60)
        self.set_simulation_hotkeys_enabled(True)
        self.set_simulation_paused(True)
        floor = self.world.add_static_box(half_extents=[3, 0.1, 3], pos=[0, -0.1, 0])
        material = self.create_pbr_material(roughness=0.8)
        floor_visual = self.scene.add_mesh(
            path="/floor", mesh_data=ke.geometry.create_box_data(6, 0.2, 6),
            material=material,
        ).set_world_translation([0, -0.1, 0])

        box = self.world.add_static_box(half_extents=[0.5, 0.5, 0.5], pos=[-1.2, 0.5, 0])
        box_visual = self.scene.add_mesh(
            path="/box", mesh_data=ke.geometry.create_box_data(1, 1, 1),
            material=material,
        ).set_world_translation([-1.2, 0.5, 0])

        mesh = ke.geometry.create_sphere_data(radius=0.6, num_longitudes=8, num_latitudes=6)
        part = ke.physics.ConvexMeshPart(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            indices=np.asarray(mesh.indices, dtype=np.uint32),
        )
        top_mesh = ke.geometry.create_sphere_data(radius=0.3, num_longitudes=8, num_latitudes=6)
        top_part = ke.physics.ConvexMeshPart(
            vertices=np.asarray(top_mesh.vertices, dtype=np.float32),
            indices=np.asarray(top_mesh.indices, dtype=np.uint32),
            local_position=[0, 0.8, 0],
        )
        convex = self.world.create_static_convex_compound(
            parts=[part, top_part], pos=[1.2, 0.6, 0],
        )
        convex_visual = self.scene.add_mesh(path="/convex", mesh_data=mesh, material=material)
        convex_visual.set_world_translation([1.2, 0.6, 0])
        convex_top = self.scene.add_mesh(path="/convex_top", mesh_data=top_mesh, material=material)
        convex_top.set_world_translation([1.2, 1.4, 0])

        self.ball = self.world.create_dynamic_sphere(radius=0.35, pos=[0, 2.5, 1.2])
        self.ball_visual = self.scene.add_mesh(
            path="/ball", mesh_data=ke.geometry.create_sphere_data(0.35, 24, 16),
            material=material,
        )
        self.inspector = self.physics_inspector
        self.inspector.set_source(ke.physics.PhysXInspectionSource(self.world))
        self.inspector.bind_visual(floor_visual, floor)
        self.inspector.bind_visual(box_visual, box)
        self.inspector.bind_visual(convex_visual, convex)
        self.inspector.bind_visual(convex_top, convex)
        self.inspector.bind_visual(self.ball_visual, self.ball)
        self.set_camera_view(position=[5, 4, 7], target=[0, 0.5, 0])
        self.set_light_direction([0.5, -1, -0.5])

    def pre_update(self):
        if self.was_key_pressed(ke.keys.R):
            self.ball.set_root_state(
                pos=[0, 2.5, 1.2], quat_xyzw=[0, 0, 0, 1],
                linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0],
            )

    def fixed_update(self, fixed_dt):
        self.world.set_dt(fixed_dt)
        self.world.step()

    def pre_render(self):
        self.ball_visual.set_world_translation(self.ball.get_root_position())
        self.ball_visual.set_world_quat_wxyz(ke.Quat.from_xyzw(self.ball.get_root_quat_xyzw()))


def main():
    args = render_example_parser(description=__doc__).parse_args()
    app = PhysicsInspectorApp()
    try:
        app.initialize(width=1280, height=800, renderer=args.renderer)
        app.start()
    finally:
        app.close()


if __name__ == "__main__":
    main()
