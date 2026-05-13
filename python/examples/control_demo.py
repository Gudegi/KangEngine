"""Direct KangEngine Python API control demo.
- KangSimWorld owns PhysX simulation and state cache.
- KangWorldVisualBridge syncs simulation objects into the viewer.
- App owns rendering, camera, input, and UI.
"""

from __future__ import annotations

import math
import os

import torch

import kangengine as ke
from kangengine import imgui, keys, scene


def asset_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(ke.__file__), "assets", *parts)


class ControlDemo(ke.App):
    def setup(self):
        self.paused = False
        self.elapsed = 0.0
        self.kp = 180.0
        self.kd = 12.0
        self.drive_amp = 0.45
        self.force_duration = 1.2
        self.force_time_left = 0.0
        self.force_vector = torch.tensor([-16.0, 2.0, 0.0], dtype=torch.float32)
        self.arrow_color = torch.tensor([[1.0, 0.9, 0.05, 1.0]], dtype=torch.float32)
        self.empty_vec3 = torch.empty((0, 3), dtype=torch.float32)
        self.empty_vec4 = torch.empty((0, 4), dtype=torch.float32)
        self.force_arrow_scale = 0.05
        self.force_arrow_handle = None
        self.force_arrow_visible = False

        device = self.getGraphicsDevice()
        vs = asset_path("shaders", "common.vs")
        fs = asset_path("shaders", "common.fs")
        checker_fs = asset_path("shaders", "checkerboard.fs")

        self.robot_shader = device.createShaderFromFile(vs, fs)
        self.rigid_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)
        for shader in (self.robot_shader, self.rigid_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)

        self.ground_shader.use()
        self.ground_shader.setVec4(
            "checkerColor1", ke.vec4([1.0, 1.0, 1.0, 1.0])
        )
        self.ground_shader.setVec4(
            "checkerColor2", ke.vec4([0.77, 0.93, 0.78, 1.0])
        )

        self.world = ke.KangSimWorld(num_envs=1, sim_dt=1.0 / 120.0, add_ground=True)
        self.visual = ke.KangWorldVisualBridge(self, self.world)

        ground = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(100.0, ke.UpAxis.Z))
        self.addShape(self.ground_shader, ground)

        self.robot_xml = asset_path("characters", "kw", "kw5.xml")
        self.ball_xml = asset_path("objects", "ball.xml")

        robot_data = self.world.load_mjcf(self.robot_xml, order="DFS")
        self.robot_obj_id = 0
        self.robot = self.world.add_articulation(
            robot_data,
            env_id=0,
            obj_id=self.robot_obj_id,
            name="kw5",
            config=ke.ArticulationConfig.free_base(),
        ).articulation

        ball_data = self.world.load_mjcf(self.ball_xml)
        self.ball_obj_id = 1
        self.ball = self.world.add_rigid(
            ball_data,
            env_id=0,
            obj_id=self.ball_obj_id,
            name="ball",
            pos=torch.tensor([1.1, 0.0, 1.0], dtype=torch.float32),
            density=1000.0,  # approx 7kg
        ).rigid

        self.visual.add_articulation(
            0,
            0,
            self.robot_xml,
            prim_base_path="/robot",
            order="DFS",
            shader=self.robot_shader,
            color=torch.tensor([0.25, 0.42, 0.95, 1.0], dtype=torch.float32),
        )
        self.visual.add_rigid(
            0,
            1,
            self.ball_xml,
            prim_base_path="/ball",
            order="DFS",
            shader=self.rigid_shader,
            color=torch.tensor([0.95, 0.18, 0.12, 1.0], dtype=torch.float32),
        )

        self.num_dofs = self.robot.num_dofs()
        self.dof_names = self.world.state.get_obj_dof_names(0)
        self.driven_dofs = list(range(self.num_dofs))
        self._reset()

        print("Own control demo")
        print(f"Robot links={self.robot.num_links()} dofs={self.num_dofs}")
        print("First DOFs:", ", ".join(self.dof_names[:8]))

    def _reset(self):
        self.elapsed = 0.0
        zero3 = torch.zeros(3, dtype=torch.float32)
        ident_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self.world.set_root_state(
            None,
            0,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            ident_quat,
            linear_velocity=zero3,
            angular_velocity=zero3,
        )
        self.world.set_dof_state(None, 0, torch.zeros(self.num_dofs, dtype=torch.float32))
        self.world.set_root_state(
            None,
            1,
            torch.tensor([1.1, 0.0, 1.0], dtype=torch.float32),
            ident_quat,
            linear_velocity=zero3,
            angular_velocity=zero3,
        )
        self.world.clear_cmd()
        self.world.step(substeps=0, apply_commands=False)
        self.visual.sync()

    def _joint_targets(self):
        targets = torch.zeros(self.num_dofs, dtype=torch.float32)
        for local_idx, dof_id in enumerate(self.driven_dofs):
            phase = self.elapsed * 2.0 + local_idx * 0.65
            targets[dof_id] = self.drive_amp * math.sin(phase)
        return targets

    def preRender(self):
        if self.force_arrow_visible and self.force_time_left <= 0.0:
            ke.scene.DebugDraw.update_arrows(
                self,
                self.force_arrow_handle,
                self.empty_vec3,
                self.empty_vec3,
                self.empty_vec4,
            )
            self.force_arrow_visible = False

        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused
        if self.was_key_pressed(keys.R):
            self._reset()
        if self.was_key_pressed(keys.F):
            self.force_time_left = self.force_duration

        if self.paused:
            return

        dt = 1.0 / 60.0
        self.elapsed += dt
        self.world.set_cmd(
            None,
            0,
            self._joint_targets(),
            mode=ke.ControlMode.POS,
            kp=self.kp,
            kd=self.kd,
        )

        if self.force_time_left > 0.0:
            obj_id = self.ball_obj_id
            # obj_id = self.robot_obj_id
            self.world.set_body_force(None, obj_id, 0, self.force_vector)
            self._log_force_arrow(self.force_vector, obj_id)
            self.force_time_left = max(0.0, self.force_time_left - dt)

        self.world.step(substeps=2)
        self.visual.sync()
        self.checkError()

    def _log_force_arrow(self, force, obj_id):
        end = torch.as_tensor(
            self.world.state.get_body_pos(obj_id)[0, 0], dtype=torch.float32
        )
        start = end - force.to(dtype=torch.float32) * self.force_arrow_scale
        starts = start.reshape(1, 3)
        ends = end.reshape(1, 3)

        if self.force_arrow_handle is None:
            self.force_arrow_handle = ke.scene.DebugDraw.log_arrows(
                self,
                self.rigid_shader,
                "/debug/force_arrow",
                starts,
                ends,
                self.arrow_color,
                0.035,
                12,
            )
        else:
            ke.scene.DebugDraw.update_arrows(
                self, self.force_arrow_handle, starts, ends, self.arrow_color
            )
        self.force_arrow_visible = True

    def render(self):
        if getattr(self, "_hideUI", False):
            return

        robot_root = self.world.state.get_root_pos(0)[0]
        ball_root = self.world.state.get_root_pos(1)[0]
        ball_contact = torch.linalg.vector_norm(
            torch.as_tensor(
                self.world.state.get_ground_contact_forces(1), dtype=torch.float32
            )
        ).item()

        imgui.begin("Control Demo")
        imgui.text("Native KangEngine Python API")
        imgui.text(f"State: {'paused' if self.paused else 'running'}")
        imgui.text("Space: pause/resume    R: reset    F: kick ball")
        imgui.separator()
        _, self.kp = imgui.slider_float("kp", self.kp, 0.0, 600.0)
        _, self.kd = imgui.slider_float("kd", self.kd, 0.0, 80.0)
        _, self.drive_amp = imgui.slider_float(
            "drive amplitude", self.drive_amp, 0.0, 1.2
        )
        imgui.separator()
        imgui.text(
            f"robot root: {robot_root[0]: .2f}, {robot_root[1]: .2f}, {robot_root[2]: .2f}"
        )
        imgui.text(
            f"ball root:  {ball_root[0]: .2f}, {ball_root[1]: .2f}, {ball_root[2]: .2f}"
        )
        imgui.text(f"ball ground contact |F|: {ball_contact: .2f}")
        imgui.text(f"driven DOFs: {len(self.driven_dofs)} / {self.num_dofs}")
        imgui.end()

    def cleanup(self):
        if hasattr(self, "world"):
            self.world.release()


if __name__ == "__main__":
    app = ControlDemo()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
