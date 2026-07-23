"""Minimal KangSimWorld example.

Inspired by Newton's basic shape examples: create a small simulation world,
spawn one rigid body, step the world, and sync it to the viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
from kangengine import imgui, keys


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


class MinimalSimWorldApp(ke.App):
    def setup(self):
        self.paused = False
        self.spawn_pos = [0.0, 0.0, 1.8]

        self.standard_materials = self.create_standard_materials()
        self.add_ground()
        self.set_camera_view([3.0, -4.0, 2.2], [0.0, 0.0, 0.7])

        self.world = ke.sim.KangSimWorld(num_envs=1, sim_dt=1.0 / 120.0, add_ground=True)
        self.visual = ke.visual.sim.SimWorldVisualizer(self, self.world)

        self.ball_xml = package_asset_path("objects", "ball.xml")
        ball_data = self.world.load_mjcf(self.ball_xml)
        self.ball = self.world.add_rigid(
            ball_data,
            env_id=0,
            obj_id=0,
            name="ball",
            pos=self.spawn_pos,
            density=600.0,
        )
        self.ball_visual = self.visual.add(
            self.ball,
            self.ball_xml,
            prim_base_path="/ball",
            material=self.standard_materials.common,
            color=[0.95, 0.2, 0.12, 1.0],
        )

        self._reset()
        print("Minimal KangSimWorld example: one dynamic rigid body")
        self.check_error()

    def _reset(self):
        self.ball.set_root_state(
            None,
            self.spawn_pos,
            [0.0, 0.0, 0.0, 1.0],
            linear_velocity=[0.0, 0.0, 0.0],
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
        pos = self.ball.get_root_pos()

        imgui.begin("Minimal Sim World")
        imgui.text("KangSimWorld + SimWorldVisualizer")
        imgui.text("Space: pause/resume    R: reset")
        imgui.separator()
        imgui.text(f"State: {'paused' if self.paused else 'running'}")
        imgui.text(f"Ball root: {pos[0]: .2f}, {pos[1]: .2f}, {pos[2]: .2f}")
        imgui.end()

    def cleanup(self):
        if hasattr(self, "world"):
            self.world.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()
    app = MinimalSimWorldApp()
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
