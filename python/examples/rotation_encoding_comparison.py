"""Compare projected-heading rotation observations on three KW robots.

Each robot uses one heading plane (XY, YZ, or XZ). Heading is derived from a
separate source body (Hips by default, matching sample_isaac2), then applied to
the selected target body. Edit only the target's ancestor-chain joints and
compare quaternion, AMP tangent-normal, and continuous matrix-6D encodings.

Run:
    python ./python/examples/rotation_encoding_comparison.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import kangengine as ke
from kangengine import imgui
from kangengine.utils.batched_rotations import (
    matrix_to_rotation_6d,
    quat_xyzw_from_angle_axis,
    quat_xyzw_heading_quat_inverse,
    quat_xyzw_heading_xy,
    quat_xyzw_heading_xz,
    quat_xyzw_heading_yz,
    quat_xyzw_multiply,
    quat_xyzw_rotate,
    quat_xyzw_to_matrix,
    quat_xyzw_to_tangent_normal,
)


@dataclass
class RobotExperiment:
    env_id: int
    plane: str
    heading_source_body: int
    target_body: int
    root_position: np.ndarray
    root_euler: np.ndarray
    targets: np.ndarray
    arrow_view: object | None = None


class RotationEncodingComparison(ke.App):
    PLANES = ("xy", "yz", "xz")
    PLANE_COLORS = {
        "xy": np.array([1.0, 0.32, 0.18, 1.0], dtype=np.float32),
        "yz": np.array([0.25, 0.9, 0.35, 1.0], dtype=np.float32),
        "xz": np.array([0.2, 0.55, 1.0, 1.0], dtype=np.float32),
    }

    def setup(self):
        self._panel_layout_initialized = False
        self.set_light_direction(ke.Vec3(-0.35, 0.55, 0.76))
        self.set_light_intensity(2.0)
        self.set_light_ambient(ke.Vec3(0.18, 0.18, 0.2))
        self.set_camera_view([0.0, -10.5, 4.2], [0.0, 0.0, 1.1])

        self.world = ke.sim.KangSimWorld(num_envs=3, add_ground=False)
        self.visual = ke.visual.sim.SimWorldVisualizer(self, self.world)
        self.materials = self.create_standard_materials()
        self.scene.add_ground(scale=18.0, material=self.materials.ground)

        self.kw_path = str(
            Path(ke.__file__).resolve().parent
            / "assets"
            / "characters"
            / "kw"
            / "kw5.xml"
        )
        data = self.world.load_mjcf(self.kw_path, order="DFS")
        self.body_names = list(data.skeleton_tree.node_names())
        self.parent_indices = list(data.skeleton_tree.parent_indices())
        config = ke.physics.ArticulationConfig.free_base()

        colors = (
            np.array([0.92, 0.35, 0.22, 1.0], dtype=np.float32),
            np.array([0.3, 0.88, 0.42, 1.0], dtype=np.float32),
            np.array([0.24, 0.52, 0.95, 1.0], dtype=np.float32),
        )
        root_positions = (-3.2, 0.0, 3.2)
        self.robots = []
        for env_id, (plane, x, color) in enumerate(
            zip(self.PLANES, root_positions, colors)
        ):
            robot = self.world.add_articulation(
                data,
                env_id=env_id,
                obj_id=0,
                name=f"kw_{plane}",
                config=config,
            )
            self.world.set_root_state(
                [env_id],
                0,
                np.array([x, 0.0, 1.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                immediate=True,
            )
            targets = np.zeros(robot.num_dofs, dtype=np.float32)
            self.world.set_dof_state([env_id], 0, targets, immediate=True)
            self.visual.add_articulation_scene_graph(
                env_id,
                0,
                self.kw_path,
                path=f"/robots/{plane}",
                order="DFS",
                material=self.materials.pbr,
                color=color,
            )
            hips = self.body_names.index("Hips")
            chest = self.body_names.index("Chest")
            self.robots.append(
                RobotExperiment(
                    env_id,
                    plane,
                    hips,
                    chest,
                    np.array([x, 0.0, 1.0], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    targets,
                )
            )

        self.dof_names = self.world.state.get_obj_dof_names(0)
        self.dof_limits = np.asarray(
            self.world.state.get_obj_dof_limits(0), dtype=np.float32
        )
        self._dof_body = [self._dof_body_name(name) for name in self.dof_names]
        self.world.step(substeps=0, apply_commands=False)
        self.visual.sync()
        self._update_heading_arrows()
        self._print_snapshot()

    @staticmethod
    def _dof_body_name(dof_name: str) -> str:
        for suffix in ("_x", "_y", "_z"):
            if dof_name.endswith(suffix):
                return dof_name[: -len(suffix)]
        return dof_name

    def _body_chain(self, body_id: int) -> set[str]:
        chain = set()
        while body_id >= 0:
            chain.add(self.body_names[body_id])
            body_id = int(self.parent_indices[body_id])
        return chain

    def _influencing_dofs(self, body_id: int) -> list[int]:
        chain = self._body_chain(body_id)
        return [i for i, body in enumerate(self._dof_body) if body in chain]

    def _slider_limits(self, dof_id: int) -> tuple[float, float]:
        lo, hi = (float(v) for v in self.dof_limits[dof_id])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return -np.pi, np.pi
        return lo, hi

    def _apply_pose(self, experiment: RobotExperiment):
        self.world.set_root_state(
            [experiment.env_id],
            0,
            experiment.root_position,
            self._root_quaternion(experiment.root_euler),
            immediate=True,
        )
        self.world.set_dof_state(
            [experiment.env_id], 0, experiment.targets, immediate=True
        )
        self.world.step(substeps=0, apply_commands=False)

    @staticmethod
    def _root_quaternion(euler_xyz: np.ndarray) -> np.ndarray:
        """Compose roll/pitch/yaw as qz * qy * qx in XYZW convention."""
        angles = torch.as_tensor(euler_xyz, dtype=torch.float32)
        axes = torch.eye(3, dtype=torch.float32)
        qx = quat_xyzw_from_angle_axis(angles[0:1], axes[0:1])[0]
        qy = quat_xyzw_from_angle_axis(angles[1:2], axes[1:2])[0]
        qz = quat_xyzw_from_angle_axis(angles[2:3], axes[2:3])[0]
        quat = quat_xyzw_multiply(qz, quat_xyzw_multiply(qy, qx))
        return quat.detach().cpu().numpy()

    @staticmethod
    def _heading_angle(plane: str, quat: torch.Tensor) -> torch.Tensor:
        if plane == "xy":
            return quat_xyzw_heading_xy(quat)
        if plane == "yz":
            return quat_xyzw_heading_yz(quat)
        return quat_xyzw_heading_xz(quat)

    def _observation(self, experiment: RobotExperiment) -> dict[str, torch.Tensor]:
        body_rotations = self.world.state.get_body_rot(0)[experiment.env_id]
        source_rotation = body_rotations[experiment.heading_source_body].to(
            dtype=torch.float32
        )
        target_rotation = body_rotations[experiment.target_body].to(
            dtype=torch.float32
        )
        inverse_heading = quat_xyzw_heading_quat_inverse(
            source_rotation, experiment.plane
        )
        local_rotation = quat_xyzw_multiply(inverse_heading, target_rotation)
        return {
            "source_world_quat": source_rotation,
            "target_world_quat": target_rotation,
            "heading": self._heading_angle(experiment.plane, source_rotation),
            "inverse_heading": inverse_heading,
            "local_quat": local_rotation,
            "amp_tan_norm": quat_xyzw_to_tangent_normal(local_rotation),
            "matrix_6d": matrix_to_rotation_6d(quat_xyzw_to_matrix(local_rotation)),
        }

    def _heading_directions(
        self, plane: str, quat: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if plane == "xy":
            reference = torch.tensor([1.0, 0.0, 0.0])
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif plane == "yz":
            reference = torch.tensor([0.0, 1.0, 0.0])
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            reference = torch.tensor([1.0, 0.0, 0.0])
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        actual = quat_xyzw_rotate(quat, reference).detach().cpu().numpy()
        heading = float(self._heading_angle(plane, quat))
        if plane == "xy":
            projected = np.array([np.cos(heading), np.sin(heading), 0.0])
        elif plane == "yz":
            projected = np.array([0.0, np.cos(heading), np.sin(heading)])
        else:
            projected = np.array([np.cos(heading), 0.0, np.sin(heading)])
        return actual.astype(np.float32), projected.astype(np.float32), normal

    def _update_heading_arrows(self):
        body_pos = self.world.state.get_body_pos(0)
        for experiment in self.robots:
            obs = self._observation(experiment)
            source_origin = (
                body_pos[experiment.env_id, experiment.heading_source_body]
                .detach()
                .cpu()
                .numpy()
            )
            target_origin = (
                body_pos[experiment.env_id, experiment.target_body]
                .detach()
                .cpu()
                .numpy()
            )
            source_actual, projected, _ = self._heading_directions(
                experiment.plane, obs["source_world_quat"]
            )
            target_axis = quat_xyzw_rotate(
                obs["target_world_quat"], torch.tensor([1.0, 0.0, 0.0])
            ).detach().cpu().numpy()
            encoded_axis = quat_xyzw_rotate(
                obs["local_quat"], torch.tensor([1.0, 0.0, 0.0])
            ).detach().cpu().numpy()
            starts = np.stack(
                (source_origin, source_origin, target_origin, target_origin)
            ).astype(np.float32)
            ends = starts + np.stack(
                (
                    source_actual * 0.75,
                    projected * 1.1,
                    target_axis * 0.75,
                    encoded_axis * 1.05,
                )
            )
            colors = np.stack(
                (
                    np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32),
                    self.PLANE_COLORS[experiment.plane],
                    np.array([0.95, 0.95, 0.95, 1.0], dtype=np.float32),
                    np.array([1.0, 0.15, 0.85, 1.0], dtype=np.float32),
                )
            )
            if experiment.arrow_view is None:
                experiment.arrow_view = self.scene.log_arrows(
                    f"/debug/{experiment.plane}_heading",
                    self.materials.debug,
                    starts,
                    ends,
                    colors,
                    0.018,
                    10,
                )
            else:
                experiment.arrow_view.update_arrows(starts, ends, colors)

    @staticmethod
    def _format_tensor(value: torch.Tensor) -> str:
        flat = value.detach().cpu().flatten().tolist()
        return "[" + ", ".join(f"{v:+.3f}" for v in flat) + "]"

    def _print_snapshot(self):
        print("\nRotation encoding snapshot")
        for experiment in self.robots:
            obs = self._observation(experiment)
            body = self.body_names[experiment.target_body]
            source = self.body_names[experiment.heading_source_body]
            print(
                f"{experiment.plane.upper()} source={source} target={body} "
                f"heading={float(obs['heading']):+.4f}"
            )
            print("  local quat   ", self._format_tensor(obs["local_quat"]))
            print("  AMP tan/norm ", self._format_tensor(obs["amp_tan_norm"]))
            print("  matrix 6D    ", self._format_tensor(obs["matrix_6d"]))

    def pre_render(self):
        self.visual.sync()
        self._update_heading_arrows()

    def render(self):
        pose_changed = False
        for panel_index, experiment in enumerate(self.robots):
            obs = self._observation(experiment)
            label = f"{experiment.plane.upper()} heading observation"
            if not self._panel_layout_initialized:
                imgui.set_next_window_pos(10.0 + panel_index * 500.0, 40.0)
                imgui.set_next_window_size(490.0, 780.0)
            imgui.begin(label)
            imgui.text(f"plane: {experiment.plane.upper()}")
            imgui.text(
                "heading source: "
                f"{self.body_names[experiment.heading_source_body]}"
            )
            if imgui.button(f"Previous source##{experiment.plane}"):
                experiment.heading_source_body = (
                    experiment.heading_source_body - 1
                ) % len(self.body_names)
            imgui.same_line()
            if imgui.button(f"Next source##{experiment.plane}"):
                experiment.heading_source_body = (
                    experiment.heading_source_body + 1
                ) % len(self.body_names)
            if imgui.button(f"Source = Hips##{experiment.plane}"):
                experiment.heading_source_body = self.body_names.index("Hips")
            imgui.same_line()
            if imgui.button(f"Source = target##{experiment.plane}"):
                experiment.heading_source_body = experiment.target_body

            imgui.separator()
            imgui.text(f"target body: {self.body_names[experiment.target_body]}")
            if imgui.button(f"Previous body##{experiment.plane}"):
                experiment.target_body = (experiment.target_body - 1) % len(
                    self.body_names
                )
            imgui.same_line()
            if imgui.button(f"Next body##{experiment.plane}"):
                experiment.target_body = (experiment.target_body + 1) % len(
                    self.body_names
                )

            imgui.separator()
            imgui.text(f"heading: {float(obs['heading']):+.4f} rad")
            imgui.text("local quaternion XYZW")
            imgui.text(self._format_tensor(obs["local_quat"]))
            imgui.text("AMP tangent + normal")
            imgui.text(self._format_tensor(obs["amp_tan_norm"]))
            imgui.text("matrix continuous 6D")
            imgui.text(self._format_tensor(obs["matrix_6d"]))
            imgui.text("gray=source axis, color=source projected heading")
            imgui.text("white=target world axis, magenta=heading-removed axis")

            imgui.separator()
            imgui.text("Hips / articulation root rotation")
            for axis_index, axis_name in enumerate(("X", "Y", "Z")):
                changed, value = imgui.slider_float(
                    f"Root {axis_name}##{experiment.plane}",
                    float(experiment.root_euler[axis_index]),
                    -np.pi,
                    np.pi,
                )
                if changed:
                    experiment.root_euler[axis_index] = value
                    self._apply_pose(experiment)
                    pose_changed = True

            imgui.separator()
            dofs = self._influencing_dofs(experiment.target_body)
            imgui.text(f"influencing joints: {len(dofs)} DOFs")
            for dof_id in dofs:
                lo, hi = self._slider_limits(dof_id)
                changed, value = imgui.slider_float(
                    f"{self.dof_names[dof_id]}##{experiment.plane}",
                    float(experiment.targets[dof_id]),
                    lo,
                    hi,
                )
                if changed:
                    experiment.targets[dof_id] = value
                    self._apply_pose(experiment)
                    pose_changed = True
            imgui.end()

        if not self._panel_layout_initialized:
            imgui.set_next_window_pos(1510.0, 40.0)
            imgui.set_next_window_size(395.0, 240.0)
        imgui.begin("Rotation experiment controls")
        if imgui.button("Copy XY pose to YZ/XZ"):
            for experiment in self.robots[1:]:
                experiment.targets[:] = self.robots[0].targets
                experiment.root_euler[:] = self.robots[0].root_euler
                self._apply_pose(experiment)
            pose_changed = True
        if imgui.button("Reset all poses"):
            for experiment in self.robots:
                experiment.targets.fill(0.0)
                experiment.root_euler.fill(0.0)
                self._apply_pose(experiment)
            pose_changed = True
        if imgui.button("Print snapshot to console"):
            self._print_snapshot()
        imgui.text("Use identical poses when comparing projection planes.")
        imgui.end()
        self._panel_layout_initialized = True

        if pose_changed:
            self.visual.sync()
            self._update_heading_arrows()

    def cleanup(self):
        if hasattr(self, "world"):
            self.world.release()


if __name__ == "__main__":
    app = RotationEncodingComparison()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
