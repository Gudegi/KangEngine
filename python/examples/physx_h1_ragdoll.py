"""
H1 Robot Ragdoll — Python equivalent of examples/physics/physx_h1_ragdoll.cpp.

Loads an MJCF robot, builds a free-base PhysX articulation, and lets it fall
under gravity.  With kp=0 and small kd this behaves as a limp ragdoll; increase
kp in code for active pose holding.
"""

import os

import kangengine as ke
from kangengine import animation, imgui, keys, scene


def asset_path(*parts):
    base = os.path.join(os.path.dirname(ke.__file__), "assets")
    return os.path.join(base, *parts)


class H1RagdollApp(ke.App):
    def setup(self):
        self.spawn_height_offset = 1.5
        self.kp = 0.0
        self.kd = 5.0
        self.paused = False
        self.show_collision = False

        device = self.getGraphicsDevice()

        vs = asset_path("shaders", "common.vs")
        fs = asset_path("shaders", "common.fs")
        checker_fs = asset_path("shaders", "checkerboard.fs")

        self.robot_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)

        for shader in (self.robot_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

        self.physics = ke.PhysicsWorld(ke.PhysicsConfig.z_up())
        self.physics.add_default_ground()

        gnd = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
        gnd.set_mesh_data(scene.Prim.create_plane_data(100.0, ke.UpAxis.Z))
        self.addShape(self.ground_shader, gnd)

        mjcf = asset_path("external", "retargetted", "kw", "kw5.xml")
        mjcf_data = animation.MJCFLoader.load(mjcf)

        self.articulation = ke.Articulation.build(
            self.physics,
            mjcf_data,
            ke.ArticulationConfig.free_base(),
        )

        self.robot = animation.SkeletonBridge.from_mjcf(
            mjcf,
            self.getScene(),
            "/robot",
            1.0,
            "DFS",
        )

        self.physics_bridge = ke.PhysicsBridge(self)
        self.physics_bridge.add(self.articulation, self.robot)

        for prim in self.robot.body_prims():
            self.addShape(self.robot_shader, prim)

        collision_prims = self.physics_bridge.add_collision_visuals(
            self.articulation,
            self.getScene(),
            "/collision",
            self.show_collision,
        )
        for prim in collision_prims:
            self.addShape(self.robot_shader, prim)

        self.targets = [0.0] * self.articulation.num_dofs()
        self.reset()

        print(
            "Ragdoll loaded: "
            f"{self.articulation.num_links()} links, "
            f"{self.articulation.num_dofs()} DOFs"
        )
        print("Defaults: kp=0.0, kd=5.0, free base, Z-up")
        self.checkError()

    def reset(self):
        self.articulation.reset_root(
            ke.vec3(0.0, 0.0, self.spawn_height_offset),
            ke.quat(1.0, 0.0, 0.0, 0.0),
        )

    def preRender(self):
        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused

        if self.was_key_pressed(keys.R):
            self.reset()

        if self.paused:
            return

        self.articulation.set_drive_targets(self.targets, self.kp, self.kd)
        self.physics.step()
        self.physics_bridge.sync()
        self.checkError()

    def render(self):
        num_links = self.articulation.num_links()
        num_dofs = self.articulation.num_dofs()
        state = "PAUSED" if self.paused else "running"

        imgui.begin("H1 Ragdoll")
        imgui.text(f"Bodies: {num_links}  DOFs: {num_dofs}  |  {state}")
        imgui.text("Space: pause/resume    R: reset")
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
    missing = [
        name
        for name in (
            "PhysicsConfig",
            "PhysicsWorld",
            "ArticulationConfig",
            "Articulation",
            "PhysicsBridge",
        )
        if not hasattr(ke, name)
    ]
    if missing:
        raise RuntimeError(
            "This example requires KangEngine to be built with PhysX bindings. "
            f"Missing: {', '.join(missing)}"
        )

    app = H1RagdollApp()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
