"""SO101-specific preset for the generic MJCF DOF control viewer."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from mjcf_dof_control import MjcfDofControlApp
import kangengine as ke
from kangengine import imgui

# python python/examples/so101_dof_control.py   --robot-port /dev/ttyACM1   --robot-id my_so101_follower   --max-relative-target 10


SO101_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# for IK
SO101_IK_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

def default_so101_xml() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "references"
        / "SO-ARM100"
        / "Simulation"
        / "SO101"
        / "so101_new_calib.xml"
    )

from kangengine.lbfgs_ik import MJCFLBFGSIK
class SO101LBFGSIK(MJCFLBFGSIK):
    """SO-101 preset with LeRobot-style action conversion."""

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        site_name: str = "gripperframe",
        scale: float = 1.0,
        order: str = "DFS",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            xml_path or default_so101_xml(),
            site_name=site_name,
            joint_names=SO101_IK_JOINT_NAMES,
            scale=scale,
            order=order,
            dtype=dtype,
            device=device,
        )

    def q_to_lerobot_action(
        self, q: torch.Tensor, *, gripper_percent: float = 50.0
    ) -> dict[str, float]:
        action = {
            f"{name}.pos": value
            for name, value in self.q_to_joint_positions(q, degrees=True).items()
        }
        action["gripper.pos"] = float(gripper_percent)
        return action

class SO101DofControlApp(MjcfDofControlApp):
    window_title = "SO101 DOF Control"
    object_name = "so101"
    prim_base_path = "/so101"
    camera_pos = (0.55, -0.75, 0.45)
    camera_target = (-0.08, 0.0, 0.12)
    ground_size = 2.0
    root_pos = (0.0, 0.0, 0.0)
    fixed_base = True
    default_kp = 120.0
    default_kd = 8.0
    default_anim_amp = 0.35
    default_anim_speed = 1.0

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        robot_port: str | None = None,
        robot_id: str = "so101_follower",
        max_relative_target: float | None = None,
        robot_send_hz: float = 30.0,
        send_on_start: bool = False,
    ):
        super().__init__(xml_path or default_so101_xml())
        self.robot_port = robot_port
        self.robot_id = robot_id
        self.max_relative_target = max_relative_target
        self.robot_send_period = 1.0 / max(float(robot_send_hz), 1e-6)
        self.send_to_robot = bool(send_on_start)
        self._last_robot_send_t = 0.0
        self.lerobot_robot = None
        self.last_robot_action: dict[str, float] | None = None
        self.robot_error: str | None = None
        self.require_ik_success_for_robot = True
        self.ik = None
        self.ik_default_target = [0.18, 0.0, 0.16]
        self.ik_target = [0.18, 0.0, 0.16]
        self.ik_follow_ball = False
        self.ik_auto_ball = False
        self.ik_solve_on_target_change = True
        self.ik_solve_hz = 15.0
        self._last_ik_solve_t = 0.0
        self.ik_gripper_percent = 50.0
        self.target_ball_radius = 0.025
        self.target_ball_prim = None
        self.ik_error: str | None = None
        self.last_ik_result = None

    def setup(self):
        super().setup()
        self._setup_ik()
        self._create_target_ball()
        if self.robot_port:
            self.animate = False
            self._connect_lerobot_robot()

    def preRender(self):
        self._update_ik_ball_follow()
        super().preRender()
        self._send_lerobot_action_if_due()

    def render(self):
        super().render()
        self._render_ik_panel()
        self._render_lerobot_panel()

    def cleanup(self):
        if self.lerobot_robot is not None:
            self.lerobot_robot.disconnect()
            self.lerobot_robot = None
        super().cleanup()

    def _connect_lerobot_robot(self):
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

            config = SO101FollowerConfig(
                port=self.robot_port,
                id=self.robot_id,
                max_relative_target=self.max_relative_target,
                use_degrees=True,
            )
            self.lerobot_robot = SO101Follower(config)
            self.lerobot_robot.connect()
            print(f"Connected LeRobot SO101 follower on {self.robot_port}")
        except Exception as exc:
            self.robot_error = str(exc)
            self.lerobot_robot = None
            print(f"Failed to connect LeRobot SO101 follower: {exc}")

    def _setup_ik(self):
        try:
            self.ik = SO101LBFGSIK(self.mjcf_path)
        except Exception as exc:
            self.ik = None
            self.ik_error = str(exc)
            print(f"Failed to initialize SO101 IK: {exc}")

    def _send_lerobot_action_if_due(self):
        if self.lerobot_robot is None or not self.send_to_robot:
            return
        if (
            self.require_ik_success_for_robot
            and self.ik_follow_ball
            and self.last_ik_result is not None
            and not self.last_ik_result.success
        ):
            return

        now = time.perf_counter()
        if now - self._last_robot_send_t < self.robot_send_period:
            return

        try:
            action = self._targets_to_lerobot_action()
            self.last_robot_action = self.lerobot_robot.send_action(action)
            self.robot_error = None
            self._last_robot_send_t = now
        except Exception as exc:
            self.robot_error = str(exc)
            self.send_to_robot = False
            print(f"Stopped robot output after send error: {exc}")

    def _targets_to_lerobot_action(self) -> dict[str, float]:
        action = {}
        for motor in SO101_MOTOR_NAMES:
            if motor not in self.dof_names:
                continue
            dof_idx = self.dof_names.index(motor)
            target_rad = float(self.targets[dof_idx])
            if motor == "gripper":
                action[f"{motor}.pos"] = self._gripper_percent(target_rad, dof_idx)
            else:
                action[f"{motor}.pos"] = float(np.rad2deg(target_rad))
        return action

    def _gripper_percent(self, target_rad: float, dof_idx: int) -> float:
        lo, hi = self.dof_limits[dof_idx]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0
        alpha = (target_rad - float(lo)) / float(hi - lo)
        return float(np.clip(alpha, 0.0, 1.0) * 100.0)

    def _solve_ik_to_targets(self):
        if self.ik is None:
            return
        try:
            current_q = [
                float(self.targets[self.dof_names.index(name)])
                for name in self.ik.joint_names
            ]
            result = self.ik.solve_position(
                self.ik_target,
                initial_q=current_q,
                max_iter=80,
                tolerance_m=1e-4,
            )
            for i, name in enumerate(self.ik.joint_names):
                self.targets[self.dof_names.index(name)] = float(result.q[i])
            self._apply_ik_gripper_target()
            self.animate = False
            self.last_ik_result = result
            self.ik_error = None
            self._apply_ik_targets_to_world()
        except Exception as exc:
            self.ik_error = str(exc)
            print(f"IK solve failed: {exc}")

    def _apply_ik_targets_to_world(self):
        self.world.set_cmd(
            None,
            self.obj_id,
            self.targets,
            mode=ke.ControlMode.POS,
            kp=self.kp,
            kd=self.kd,
        )
        if self.paused:
            self.world.set_dof_state(None, self.obj_id, self.targets)
            self.world.step(substeps=0)
            self.visual.sync()
            self._update_contact_force_arrows()

    def _apply_ik_gripper_target(self):
        if "gripper" not in self.dof_names:
            return
        dof_idx = self.dof_names.index("gripper")
        lo, hi = self.dof_limits[dof_idx]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        alpha = float(np.clip(self.ik_gripper_percent, 0.0, 100.0)) / 100.0
        self.targets[dof_idx] = float(lo + alpha * (hi - lo))

    def _create_target_ball(self):
        self.target_ball_prim = self.scene.define_prim(
            "/ik_target_ball", ke.scene.PrimType.Mesh
        )
        self.target_ball_prim.set_mesh_data(
            ke.geometry.create_sphere_data(float(self.target_ball_radius), 32, 16)
        )
        self.target_ball_prim.set_display_color_alpha(ke.vec4(0.95, 0.18, 0.12, 1.0))
        self._sync_target_ball_prim()
        self.scene.add_renderable(self.target_ball_prim, self.robot_shader)

    def _sync_target_ball_prim(self):
        if self.target_ball_prim is None:
            return
        self.target_ball_prim.set_world_translation(
            ke.vec3(
                float(self.ik_target[0]),
                float(self.ik_target[1]),
                float(self.ik_target[2]),
            ),
        )

    def _read_target_ball_prim(self):
        if self.target_ball_prim is None:
            return None
        try:
            pos = self.target_ball_prim.compute_world_matrix()[3]
            return [float(pos.x), float(pos.y), float(pos.z)]
        except Exception:
            return None

    def _update_target_from_ball_prim(self) -> bool:
        pos = self._read_target_ball_prim()
        if pos is None:
            return False
        if np.allclose(np.asarray(pos), np.asarray(self.ik_target), atol=1e-5):
            return False
        self.ik_target = pos
        return True

    def _update_auto_ball(self):
        if not self.ik_auto_ball:
            return
        t = float(self.elapsed)
        self.ik_target[0] = 0.22 + 0.07 * math.sin(0.55 * t)
        self.ik_target[1] = 0.10 * math.sin(0.38 * t)
        self.ik_target[2] = 0.17 + 0.04 * math.sin(0.73 * t + 0.4)

    def _update_ik_ball_follow(self):
        if self.ik_auto_ball:
            self._update_auto_ball()
            self._sync_target_ball_prim()
        elif self._update_target_from_ball_prim() and self.ik_solve_on_target_change:
            self._solve_ik_to_targets()
            self._last_ik_solve_t = time.perf_counter()
        if not self.ik_follow_ball:
            return
        now = time.perf_counter()
        period = 1.0 / max(float(self.ik_solve_hz), 1e-6)
        if now - self._last_ik_solve_t < period:
            return
        self._solve_ik_to_targets()
        self._last_ik_solve_t = now

    def _render_ik_panel(self):
        imgui.begin("SO101 LBFGS IK")
        if self.ik is None:
            imgui.text("IK: unavailable")
            if self.ik_error:
                imgui.text(f"Error: {self.ik_error}")
            imgui.end()
            return

        target_changed = False
        changed, self.ik_target[0] = imgui.slider_float(
            "target x (m)", self.ik_target[0], -0.1, 0.45
        )
        target_changed = target_changed or changed
        changed, self.ik_target[1] = imgui.slider_float(
            "target y (m)", self.ik_target[1], -0.3, 0.3
        )
        target_changed = target_changed or changed
        changed, self.ik_target[2] = imgui.slider_float(
            "target z (m)", self.ik_target[2], 0.0, 0.45
        )
        target_changed = target_changed or changed
        changed, self.ik_gripper_percent = imgui.slider_float(
            "gripper (%)", self.ik_gripper_percent, 0.0, 100.0
        )
        if changed:
            self._apply_ik_gripper_target()
            self._apply_ik_targets_to_world()
        if target_changed:
            self._sync_target_ball_prim()

        _, self.ik_solve_on_target_change = imgui.checkbox(
            "Solve while editing target", self.ik_solve_on_target_change
        )
        _, self.ik_follow_ball = imgui.checkbox(
            "Follow ball with IK", self.ik_follow_ball
        )
        if self.ik_follow_ball:
            self.animate = False
        _, self.ik_auto_ball = imgui.checkbox(
            "Animate target ball", self.ik_auto_ball
        )
        _, self.ik_solve_hz = imgui.slider_float(
            "IK solve Hz", self.ik_solve_hz, 1.0, 60.0
        )
        if target_changed and self.ik_solve_on_target_change:
            self._solve_ik_to_targets()
            self._last_ik_solve_t = time.perf_counter()
        if imgui.button("Solve IK to GUI targets"):
            self._solve_ik_to_targets()
            self._last_ik_solve_t = time.perf_counter()
        imgui.same_line()
        if imgui.button("Reset IK target"):
            self.ik_target = list(self.ik_default_target)
            self._sync_target_ball_prim()
            if self.ik_solve_on_target_change:
                self._solve_ik_to_targets()
                self._last_ik_solve_t = time.perf_counter()

        if self.last_ik_result is not None:
            result = self.last_ik_result
            imgui.separator()
            imgui.text(f"Error: {result.position_error * 1000.0:.3f} mm")
            imgui.text(f"Success: {result.success}")
            for i, name in enumerate(self.ik.joint_names):
                imgui.text(f"{name}: {float(np.rad2deg(result.q[i])): .2f} deg")
        if self.ik_error:
            imgui.separator()
            imgui.text(f"Error: {self.ik_error}")
        imgui.end()

    def _render_lerobot_panel(self):
        imgui.begin("LeRobot SO101 Output")
        if self.lerobot_robot is None:
            imgui.text("Robot: disconnected")
            if self.robot_port:
                imgui.text(f"Port: {self.robot_port}")
        else:
            imgui.text("Robot: connected")
            imgui.text(f"Port: {self.robot_port}")
            changed, self.send_to_robot = imgui.checkbox(
                "Send GUI targets to robot",
                self.send_to_robot,
            )
            if changed and self.send_to_robot:
                self.animate = False
                self._last_robot_send_t = 0.0
            _, self.require_ik_success_for_robot = imgui.checkbox(
                "Only sync solved IK",
                self.require_ik_success_for_robot,
            )
            if self.last_robot_action:
                imgui.separator()
                imgui.text("Last sent action")
                for key, value in self.last_robot_action.items():
                    imgui.text(f"{key}: {float(value): .2f}")
        if self.robot_error:
            imgui.separator()
            imgui.text(f"Error: {self.robot_error}")
        imgui.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--robot-port", default=None)
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--max-relative-target", type=float, default=None)
    parser.add_argument("--robot-send-hz", type=float, default=30.0)
    parser.add_argument("--send-on-start", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    app = SO101DofControlApp(
        args.xml_path,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        max_relative_target=args.max_relative_target,
        robot_send_hz=args.robot_send_hz,
        send_on_start=args.send_on_start,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
