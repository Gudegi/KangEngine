"""Batched KangSimWorld example with multiple independent environments.

This is the small ramp example: create many envs, spawn one rigid body per env,
step them together, and read batched state tensors from ``world.state``.

The example also demonstrates runtime PhysX material changes on a sloped static
terrain.  All boxes start with a low-friction material; after a short delay,
half of the envs are switched to a high-friction material while the other half
remains slippery.  The UI shows the average downhill speed of both groups so
friction changes can be checked while the simulation is running.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import kangengine as ke
from kangengine import imgui, keys


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def grid_position(index: int, columns: int, spacing: float, height: float):
    row = index // columns
    col = index % columns
    x = (col - (columns - 1) * 0.5) * spacing
    y = row * spacing
    return [x, y, height]

# TDOO : make easy Cloner API
def grid_positions(num_envs: int, columns: int, spacing: float, height: float):
    env_ids = torch.arange(num_envs, dtype=torch.float32)
    rows = torch.div(env_ids, columns, rounding_mode="floor")
    cols = torch.remainder(env_ids, columns)
    x = (cols - (columns - 1) * 0.5) * spacing
    y = rows * spacing
    z = torch.full_like(x, float(height))
    return torch.stack((x, y, z), dim=1)


def y_axis_quat_xyzw(angle_rad: float):
    half = angle_rad * 0.5
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def friction_group_color(env_id: int):
    # Even envs stay low-friction/slippery. Odd envs switch to high friction at
    # runtime. Keep colors grouped instead of using a gradient so the test reads
    # visually at a glance.
    if env_id % 2 == 0:
        return [0.15, 0.45, 1.0, 1.0]  # low friction: blue
    return [1.0, 0.42, 0.12, 1.0]  # runtime high friction: orange


class MultiEnvSimWorldApp(ke.App):
    def __init__(self, num_envs: int, friction_switch_time: float):
        super().__init__()
        self.num_envs = int(num_envs)
        self.friction_switch_time = float(friction_switch_time)

    def setup(self):
        self.paused = False
        self.sim_time = 0.0
        self.material_update_count = 0
        self.runtime_material_applied = False
        self.spacing = 0.8
        self.ramp_angle_deg = 18.0
        self.ramp_angle = math.radians(self.ramp_angle_deg)
        self.ramp_half_extents = [4.7, 4.0, 0.05]
        self.ramp_pos = [0.0, 0.0, 0.0]
        self.ramp_rot_xyzw = y_axis_quat_xyzw(self.ramp_angle)
        self.box_half_z = 0.12
        self.spawn_x = -2.4
        self.columns = max(1, int(math.ceil(math.sqrt(self.num_envs))))
        self.low_friction_envs = tuple(range(0, self.num_envs, 2))
        self.high_friction_envs = tuple(range(1, self.num_envs, 2))
        if not self.high_friction_envs and self.num_envs > 0:
            self.high_friction_envs = (0,)
            self.low_friction_envs = ()
        self.low_friction_material = ke.physics.PhysicsMaterialDesc([0.05, 0.05, 0.0])
        self.high_friction_material = ke.physics.PhysicsMaterialDesc([3.0, 2.5, 0.0])

        self.shaders = self.create_standard_shaders()
        self.set_camera_view([3.6, -5.0, 2.8], [0.0, 0.0, 0.6])

        self.world = ke.sim.KangSimWorld(
            num_envs=self.num_envs,
            sim_dt=1.0 / 120.0,
            add_ground=False,
        )
        self.world.physics.add_static_box(
            self.ramp_half_extents,
            self.ramp_pos,
            self.ramp_rot_xyzw,
            register_as_ground=True,
        )
        self.visual = ke.visual.sim.SimWorldVisualizer(self, self.world)
        self._add_ramp_visual()
        self.box_xml = package_asset_path("objects", "box.xml")
        box_data = self.world.load_mjcf(self.box_xml)
        for env_id in range(self.num_envs):
            pos = self._ramp_spawn_position(env_id)
            self.world.add_rigid(
                box_data,
                env_id=env_id,
                obj_id=0,
                name=f"box_{env_id}",
                pos=pos,
                density=60.0,
            )

        self.box = self.world.get_rigid_batch(obj_id=0)
        # Runtime API smoke: start every rigid instance with a slippery material.
        # Later, a subset is switched to high friction while the sim is running.
        self.material_update_count = self.box.set_collision_material(
            None, self.low_friction_material
        )
        self.box_visual_batch = self.visual.add(
            self.box,
            self.box_xml,
            prim_base_path="/group/box",
            shader=self.shaders.common,
            color=[friction_group_color(env_id) for env_id in range(self.num_envs)],
        )

        self._reset()
        print(f"Multi-env KangSimWorld example: num_envs={self.num_envs}")
        print(f"root_pos tensor shape: {tuple(self.box.get_root_pos().shape)}")
        print(
            "Runtime friction demo: "
            f"initial low-friction shapes updated={self.material_update_count}; "
            f"switch high-friction envs={self.high_friction_envs} "
            f"at t={self.friction_switch_time:.2f}s; "
            f"ramp={self.ramp_angle_deg:.1f}deg"
        )
        self.check_error()

    def _reset(self):
        self.sim_time = 0.0
        self.runtime_material_applied = False
        self.material_update_count = self.box.set_collision_material(
            None, self.low_friction_material
        )
        positions = torch.tensor(
            [self._ramp_spawn_position(env_id) for env_id in range(self.num_envs)],
            dtype=torch.float32,
        )
        rotations = torch.zeros((self.num_envs, 4), dtype=torch.float32)
        rotations[:, 3] = 1.0 # quat xyzw

        velocities = torch.zeros((self.num_envs, 3), dtype=torch.float32)
        # +X is downhill for the ramp quaternion above.
        velocities[:, 0] = 1.0
        velocities[:, 1] = 0.0
        velocities[:, 2] = 0.0

        self.box.set_root_state(
            None,
            positions,
            rotations,
            linear_velocity=velocities,
            angular_velocity=[0.0, 0.0, 0.0],
        )
        self.world.step(substeps=0, apply_commands=False)
        self.visual.sync()

    def preRender(self):
        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused
        if self.was_key_pressed(keys.R):
            self._reset()
        if self.paused:
            return

        self.world.step(substeps=2)
        self.sim_time += self.world.sim_dt * 2
        if (
            not self.runtime_material_applied
            and self.sim_time >= self.friction_switch_time
        ):
            self.material_update_count += self.box.set_collision_material(
                self.high_friction_envs, self.high_friction_material
            )
            self.runtime_material_applied = True
            print(
                "Runtime friction switched: "
                f"envs={self.high_friction_envs}, "
                f"total_updated_shapes={self.material_update_count}"
            )
        self.visual.sync()
        self.check_error()

    def render(self):
        root_pos = self.box.get_root_pos()
        root_vel = self.box.get_root_vel()
        mean_height = float(root_pos[:, 2].mean().item())
        min_height = float(root_pos[:, 2].min().item())
        max_height = float(root_pos[:, 2].max().item())
        downhill_speed = root_vel[:, 0]
        low_speed = self._mean_for_envs(downhill_speed, self.low_friction_envs)
        high_speed = self._mean_for_envs(downhill_speed, self.high_friction_envs)

        imgui.begin("Multi-Env Sim World")
        imgui.text("KangSimWorld batched state example")
        imgui.text("Space: pause/resume    R: reset")
        imgui.separator()
        imgui.text(f"State: {'paused' if self.paused else 'running'}")
        imgui.text(f"Envs: {self.num_envs}")
        imgui.text(f"root_pos shape: {tuple(root_pos.shape)}")
        imgui.text("height min/mean/max:")
        imgui.same_line()
        imgui.text(f"{min_height: .2f} / {mean_height: .2f} / {max_height: .2f}")
        imgui.separator()
        imgui.text("Runtime friction material test")
        imgui.text(
            f"switch: {'done' if self.runtime_material_applied else 'pending'} "
            f"at t={self.friction_switch_time:.2f}s"
        )
        imgui.text(f"ramp angle: {self.ramp_angle_deg:.1f} deg")
        imgui.text(f"updated shapes: {self.material_update_count}")
        imgui.text(f"low friction envs: {self.low_friction_envs}")
        imgui.text(f"high friction envs: {self.high_friction_envs}")
        imgui.text(
            "downhill velocity low/high: "
            f"{low_speed: .3f} / {high_speed: .3f}"
        )
        imgui.end()

    def _ramp_spawn_position(self, env_id: int):
        row = env_id // self.columns
        col = env_id % self.columns
        y = (col - (self.columns - 1) * 0.5) * self.spacing
        x = self.spawn_x - row * 0.45
        z = self._ramp_top_z(x) + self.box_half_z + 0.04
        return [x, y, z]

    def _ramp_top_z(self, x: float) -> float:
        half_t = self.ramp_half_extents[2]
        return -math.tan(self.ramp_angle) * float(x) + half_t / math.cos(
            self.ramp_angle
        )

    def _add_ramp_visual(self):
        mesh_data = ke.geometry.create_box_data(
            self.ramp_half_extents[0] * 2.0,
            self.ramp_half_extents[1] * 2.0,
            self.ramp_half_extents[2] * 2.0,
        )
        view = self.add_mesh(
            "/terrain/inclined_box",
            mesh_data,
            self.shaders.common,
            color=[0.45, 0.45, 0.5, 1.0],
        )
        view.prim.set_local_translation(ke.vec3(self.ramp_pos))
        # ke.quat constructor is (w, x, y, z), while PhysX binding args above
        # use xyzw lists.
        view.prim.set_local_rotation(
            ke.quat(
                self.ramp_rot_xyzw[3],
                self.ramp_rot_xyzw[0],
                self.ramp_rot_xyzw[1],
                self.ramp_rot_xyzw[2],
            )
        )
        return view

    @staticmethod
    def _mean_for_envs(values: torch.Tensor, env_ids: tuple[int, ...]) -> float:
        if not env_ids:
            return float("nan")
        index = torch.tensor(env_ids, dtype=torch.long, device=values.device)
        return float(values.index_select(0, index).mean().item())

    def cleanup(self):
        if hasattr(self, "world"):
            self.world.release()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--friction-switch-time",
        type=float,
        default=1.0,
        help="Seconds before switching odd envs to high friction at runtime.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = MultiEnvSimWorldApp(args.num_envs, args.friction_switch_time)
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
