"""Visual example of SceneGraph-authored rigid bodies.

The example deliberately avoids the low-level ``PhysicsWorld.create_*`` actor
helpers.  Render meshes and collision authoring live on the same scene prim;
``ScenePhysicsSystem`` creates and synchronizes the PhysX actors.
"""

from __future__ import annotations

import math

import kangengine as ke
from kangengine import imgui, keys


class ScenePhysicsRigidBodiesApp(ke.App):
    sim_dt = 1.0 / 240.0
    control_dt = 1.0 / 60.0
    physics_substeps = 4

    def __init__(self):
        super().__init__()
        self.physics = None
        self.physics_system = None
        self.dynamic_bodies = []
        self.kinematic_platform = None
        self.kinematic_time = 0.0

    def setup(self):
        self.standard_materials = self.create_standard_materials()
        self.set_light_direction(ke.Vec3(-0.35, 0.82, -0.45))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.Vec3(0.30, 0.30, 0.30))

        config = ke.physics.PhysicsConfig.y_up()
        # Match mjcf_dof_control.py's PhysX substep.  Its drag gains
        # (750/75/300) are tuned for 240 Hz and become visibly unstable when
        # applied to light rigid bodies at the default 60 Hz timestep.
        config.dt = self.sim_dt
        self.physics = ke.physics.PhysicsWorld(config)
        self.bind_scene_physics_world(self.physics)
        self.physics_system = self.get_scene_physics_system()
        self.configure_timing(
            ke.SimulationTimingConfig.from_dt(
                physics_dt=config.dt,
                fixed_dt=self.control_dt,
                render_hz=60.0,
            )
        )
        self.set_simulation_hotkeys_enabled(True)

        self._create_ground()
        self._create_kinematic_platform()
        self._spawn_dynamic_bodies()
        self._setup_camera()

    def _create_material(self, diffuse):
        return self.create_phong_material(
            ambient=diffuse * 0.22,
            diffuse=diffuse,
            specular=ke.Vec3(0.12, 0.12, 0.12),
            shininess=24.0,
        )

    def _author_rigid_body(
        self,
        view,
        body_type,
        shape_type,
        shape_size,
        *,
        density=1.0,
        restitution=0.05,
    ):
        rigid_body = view.prim.add_rigid_body_component()
        rigid_body.body_type = body_type
        rigid_body.density = density

        collision = view.prim.add_collision_shape_component()
        collision.shape_type = shape_type
        collision.size = shape_size
        collision.static_friction = 0.8
        collision.dynamic_friction = 0.65
        collision.restitution = restitution

        self.physics_system.register_rigid_body(view.prim)
        return rigid_body

    def _create_ground(self):
        view = self.scene.add_mesh(
            "/scene_physics/ground",
            ke.geometry.create_box_data(14.0, 0.5, 10.0),
            self._create_material(ke.Vec3(0.28, 0.30, 0.33)),
        )
        view.set_world_translation(ke.Vec3(0.0, -0.25, 0.0))
        self._author_rigid_body(
            view,
            ke.scene.RigidBodyType.STATIC,
            ke.scene.CollisionShapeType.BOX,
            ke.Vec3(7.0, 0.25, 5.0),
        )

    def _create_kinematic_platform(self):
        view = self.scene.add_mesh(
            "/scene_physics/kinematic_platform",
            ke.geometry.create_box_data(3.0, 0.4, 2.5),
            self._create_material(ke.Vec3(0.18, 0.72, 0.38)),
        )
        view.set_world_translation(ke.Vec3(0.0, 2.2, 0.0))
        rigid_body = self._author_rigid_body(
            view,
            ke.scene.RigidBodyType.KINEMATIC,
            ke.scene.CollisionShapeType.BOX,
            ke.Vec3(1.5, 0.2, 1.25),
        )
        self.kinematic_platform = (view, rigid_body)

    def _spawn_dynamic_bodies(self):
        box = self.scene.add_mesh(
            "/scene_physics/dynamic_box",
            ke.geometry.create_cube_data(1.0),
            self._create_material(ke.Vec3(0.92, 0.32, 0.12)),
        )
        box.set_world_translation(ke.Vec3(-0.55, 5.2, 0.0))
        box.set_world_rotation_axis_angle(ke.Vec3(0.0, 0.0, 1.0), 0.35)
        box_body = self._author_rigid_body(
            box,
            ke.scene.RigidBodyType.DYNAMIC,
            ke.scene.CollisionShapeType.BOX,
            ke.Vec3(0.5, 0.5, 0.5),
            density=1.5,
            restitution=0.12,
        )

        sphere = self.scene.add_mesh(
            "/scene_physics/dynamic_sphere",
            ke.geometry.create_sphere_data(0.55, 32, 16),
            self._create_material(ke.Vec3(0.16, 0.42, 0.95)),
        )
        sphere.set_world_translation(ke.Vec3(0.65, 7.0, 0.15))
        sphere_body = self._author_rigid_body(
            sphere,
            ke.scene.RigidBodyType.DYNAMIC,
            ke.scene.CollisionShapeType.SPHERE,
            ke.Vec3(0.55, 0.0, 0.0),
            density=1.0,
            restitution=0.25,
        )

        self.dynamic_bodies = [
            (box, box_body, ke.Vec3(-0.55, 5.2, 0.0)),
            (sphere, sphere_body, ke.Vec3(0.65, 7.0, 0.15)),
        ]

    def _setup_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.02)
        camera.set_far_plane(200.0)
        camera.set_fov(55.0)
        camera.set_target_pos(ke.Vec3(0.0, 2.4, 0.0))
        camera.set_camera_pos(ke.Vec3(8.0, 5.8, 10.5))
        self.set_camera_move_speed(3.0)

    def _reset_dynamic_bodies(self):
        # Re-registering is the public authoring-level way to replace a runtime
        # actor.  It also clears the old velocity without exposing PxRigidActor.
        for index, (view, rigid_body, position) in enumerate(self.dynamic_bodies):
            self.physics_system.unregister(rigid_body)
            view.set_world_translation(position)
            view.set_world_rotation_axis_angle(
                ke.Vec3(0.0, 0.0, 1.0), 0.35 if index == 0 else 0.0
            )
            self.physics_system.register_rigid_body(view.prim)

    def pre_update(self):
        self.kinematic_time += self.get_delta_time()
        if self.kinematic_platform is not None:
            view, _ = self.kinematic_platform
            view.set_world_translation(
                ke.Vec3(math.sin(self.kinematic_time * 0.8) * 2.25, 2.2, 0.0)
            )
        if self.was_key_pressed(keys.R):
            self._reset_dynamic_bodies()

    def fixed_update(self, fixed_dt):
        self.step_scene_physics(self.physics_substeps)

    def render(self):
        imgui.begin("Scene Physics System")
        imgui.text("SceneGraph components create and own the actor registration")
        imgui.separator()
        imgui.text(f"registered bodies: {self.physics_system.registration_count}")
        imgui.text(f"PhysX body actors: {self.physics.num_body_actors()}")
        for view, rigid_body, _ in self.dynamic_bodies:
            imgui.text(
                f"{view.path}: contacts={self.physics_system.contact_count(rigid_body)}"
            )
        if imgui.button("Reset dynamic bodies (R)"):
            self._reset_dynamic_bodies()
        imgui.text("Force mode + Shift+Left Drag: apply spring force")
        imgui.text("Red arrow: applied force    Magenta point: drag target")
        imgui.text("Enter: play/pause    Space: pause/step    R: reset")
        imgui.text("Select a prim to inspect its physics components")
        imgui.end()


if __name__ == "__main__":
    app = ScenePhysicsRigidBodiesApp()
    app.initialize(1440, 900, False, ke.UpAxis.Y)
    app.start()
