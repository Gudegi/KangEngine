"""
H1 Robot Ragdoll — Python equivalent of examples/physics/physx_h1_ragdoll.cpp.

Loads an MJCF robot, builds a free-base PhysX articulation, and lets it fall
under gravity.  With kp=0 and small kd this behaves as a limp ragdoll; increase
kp in code for active pose holding.
"""

import os

import kangengine as ke
from kangengine import asset, imgui, keys, scene, visual


def asset_path(*parts):
    base = os.path.join(os.path.dirname(ke.__file__), "assets")
    return os.path.join(base, *parts)


class RagdollApp(ke.App):
    def setup(self):
        self.spawn_height_offset = 1.5
        self.kp = 0.0
        self.kd = 5.0
        self.show_collision = False

        device = self.get_renderer().device()

        vs = asset_path("shaders", "common.vs")
        fs = asset_path("shaders", "common.fs")
        checker_fs = asset_path("shaders", "checkerboard.fs")

        self.robot_shader = device.create_shader_from_file(vs, fs)
        self.ground_shader = device.create_shader_from_file(vs, checker_fs)

        for shader in (self.robot_shader, self.ground_shader):
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)

        self.ground_shader.use()
        self.ground_shader.set_vec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.set_vec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

        physics_config = ke.physics.PhysicsConfig.z_up()
        self.physics = ke.physics.PhysicsWorld(physics_config)
        self.timing = self.configure_timing(
            ke.SimulationTimingConfig.from_dt(
                physics_dt=physics_config.dt,
                fixed_dt=physics_config.dt,
                render_hz=60.0,
            )
        )
        self.set_simulation_hotkeys_enabled(True)
        self.physics.add_default_ground()

        self.scene.add_ground(scale=100.0, shader=self.ground_shader)

        mjcf = asset_path("external", "retargetted", "kw", "kw5.xml")
        mjcf_data = asset.MJCFLoader.load(mjcf)

        self.articulation = ke.physics.Articulation.build(
            self.physics,
            mjcf_data,
            ke.physics.ArticulationConfig.free_base(),
        )

        self.robot = visual.ArticulationVisual.from_mjcf(
            mjcf,
            self.scene.native,
            "/robot",
            1.0,
            "DFS",
        )

        self.physics_bridge = ke.physics.PhysicsBridge()
        self.physics_bridge.add(self.articulation, self.robot)

        for prim in self.robot.body_prims():
            self.scene.add_renderable(prim, self.robot_shader)

        collision_prims = self.physics_bridge.add_collision_visuals(
            self.articulation,
            self.scene.native,
            "/collision",
            self.show_collision,
        )
        for prim in collision_prims:
            self.scene.add_renderable(prim, self.robot_shader)

        self.targets = [0.0] * self.articulation.num_dofs()
        self.reset()

        print(
            "Ragdoll loaded: "
            f"{self.articulation.num_links()} links, "
            f"{self.articulation.num_dofs()} DOFs"
        )
        print("Defaults: kp=0.0, kd=5.0, free base, Z-up")
        self.check_error()

    def reset(self):
        self.articulation.reset_root(
            ke.vec3(0.0, 0.0, self.spawn_height_offset),
            ke.quat(1.0, 0.0, 0.0, 0.0),
        )

    def preUpdate(self):
        if self.was_key_pressed(keys.R):
            self.reset()

    def fixedUpdate(self, fixed_dt):
        self.articulation.set_drive_targets(self.targets, self.kp, self.kd)
        self.physics.step()

    def preRender(self):
        self.physics_bridge.sync()
        self.check_error()

    def render(self):
        num_links = self.articulation.num_links()
        num_dofs = self.articulation.num_dofs()
        state = "PAUSED" if self.is_simulation_paused() else "running"

        imgui.begin("H1 Ragdoll")
        imgui.text(f"Bodies: {num_links}  DOFs: {num_dofs}  |  {state}")
        imgui.text("Enter: play/pause    Space: pause/step    R: reset")
        imgui.separator()

        mode = "Pure ragdoll (kp=0)" if self.kp < 1.0 else "Active ragdoll (kp>0)"
        imgui.text(f"Mode: {mode}")
        _, self.kp = imgui.slider_float("kp (stiffness, 0=limp)", self.kp, 0.0, 500.0)
        _, self.kd = imgui.slider_float("kd (damping)", self.kd, 0.0, 50.0)
        _, self.spawn_height_offset = imgui.slider_float(
            "Spawn height offset (m)",
            self.spawn_height_offset,
            0.5,
            5.0,
        )
        imgui.separator()

        if imgui.button("Reset targets to zero"):
            self.targets = [0.0] * len(self.targets)
        imgui.same_line()
        if imgui.button("Reset"):
            self.reset()
        imgui.same_line()
        changed, self.show_collision = imgui.checkbox(
            "Show collision",
            self.show_collision,
        )
        if changed:
            self.physics_bridge.set_collision_visible(self.show_collision)
            self._set_robot_alpha(0.12 if self.show_collision else 1.0)

        if self.kp >= 1.0:
            imgui.text("Joint targets (rad):")
            imgui.begin_child("joints", 0.0, 350.0, True)
            dof_idx = 0
            joints = self.articulation.joints()
            for link_idx in range(1, num_links):
                for joint in joints.get(link_idx, []):
                    _, self.targets[dof_idx] = imgui.slider_float(
                        joint.name,
                        self.targets[dof_idx],
                        joint.lo_limit,
                        joint.hi_limit,
                    )
                    dof_idx += 1
            imgui.end_child()
        else:
            imgui.text_disabled("(Set kp > 0 to enable joint targets)")

        imgui.end()

    def _set_robot_alpha(self, alpha):
        for prim in self.robot.body_prims():
            color = prim.get_display_color_alpha()
            if color is None:
                color = ke.vec4(0.15, 0.15, 0.15, 1.0)
            prim.set_display_color_alpha(ke.vec4(color.x, color.y, color.z, alpha))

    def postRender(self):
        pass

    def cleanup(self):
        if hasattr(self, "physics_bridge"):
            self.physics_bridge = None
        if hasattr(self, "articulation"):
            self.articulation.release()
            self.articulation = None
        if hasattr(self, "physics"):
            self.physics = None


if __name__ == "__main__":
    physics = getattr(ke, "physics", None)
    missing = [
        name
        for name in (
            "PhysicsConfig",
            "PhysicsWorld",
            "ArticulationConfig",
            "Articulation",
            "PhysicsBridge",
        )
        if physics is None or not hasattr(physics, name)
    ]
    if missing:
        raise RuntimeError(
            "This example requires KangEngine to be built with PhysX bindings. "
            f"Missing: {', '.join(missing)}"
        )

    app = RagdollApp()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
