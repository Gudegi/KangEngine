"""Batched KangSimWorld example with multiple independent environments.

This is the small "platform" example: create many envs, spawn one rigid body per
env, step them together, and read batched state tensors from ``world.state``.
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


def env_color(index: int, count: int):
    t = index / max(1, count - 1)
    return [0.15 + 0.75 * t, 0.35, 0.95 - 0.55 * t, 1.0]


class MultiEnvSimWorldApp(ke.App):
    def __init__(self, num_envs: int):
        super().__init__()
        self.num_envs = int(num_envs)

    def setup(self):
        self.paused = False
        self.spacing = 0.7
        self.spawn_height = 1.5
        self.columns = max(1, int(math.ceil(math.sqrt(self.num_envs))))

        self.shaders = self.create_standard_shaders()
        self.add_ground(scale=12.0, shader=self.shaders.ground)
        self.set_camera_view([3.8, -5.2, 3.2], [0.0, 1.2, 0.5])

        self.world = ke.KangSimWorld(
            num_envs=self.num_envs,
            sim_dt=1.0 / 120.0,
            add_ground=True,
        )
        self.visual = ke.KangWorldVisualBridge(self, self.world)
        self.ball_xml = package_asset_path("objects", "ball.xml")
        ball_data = self.world.load_mjcf(self.ball_xml)

        for env_id in range(self.num_envs):
            pos = grid_position(env_id, self.columns, self.spacing, self.spawn_height)
            self.world.add_rigid(
                ball_data,
                env_id=env_id,
                obj_id=0,
                name=f"ball_{env_id}",
                pos=pos,
                density=60.0,
            )

            self.visual.add_rigid(
                env_id,
                0,
                self.ball_xml,
                prim_base_path=f"/group/env_{env_id}/ball",
                shader=self.shaders.common,
                color=env_color(env_id, self.num_envs),
            )

        self._reset()
        print(f"Multi-env KangSimWorld example: num_envs={self.num_envs}")
        print(f"root_pos tensor shape: {tuple(self.world.state.get_root_pos(0).shape)}")
        self.check_error()

    def _reset(self):
        env_ids = torch.arange(self.num_envs, dtype=torch.int64)
        positions = grid_positions(
            self.num_envs,
            self.columns,
            self.spacing,
            self.spawn_height,
        )
        rotations = torch.zeros((self.num_envs, 4), dtype=torch.float32)
        rotations[:, 3] = 1.0 # quat xyzw

        velocities = torch.zeros((self.num_envs, 3), dtype=torch.float32)
        velocities[:, 0] = torch.linspace(-2.0, 2.0, self.num_envs)
        velocities[:, 1] = 1.0
        velocities[:, 2] = 5.0

        self.world.set_root_state(
            env_ids,
            0,
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
        self.visual.sync()
        self.check_error()

    def render(self):
        root_pos = self.world.state.get_root_pos(0)
        mean_height = float(root_pos[:, 2].mean().item())
        min_height = float(root_pos[:, 2].min().item())
        max_height = float(root_pos[:, 2].max().item())

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
        imgui.end()

    def cleanup(self):
        if hasattr(self, "world"):
            self.world.release()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main():
    args = parse_args()
    app = MultiEnvSimWorldApp(args.num_envs)
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
