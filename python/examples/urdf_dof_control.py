"""Unitree G1 URDF articulation DOF position-control viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from mjcf_dof_control import MjcfDofControlApp, package_asset_path


def default_urdf_path() -> Path:
    return Path(package_asset_path("characters", "g1", "g1_29dof.urdf"))


class UrdfDofControlApp(MjcfDofControlApp):
    """Load the G1 URDF and expose every revolute DOF as a slider."""

    window_title = "URDF DOF Control — Unitree G1"
    object_name = "g1"
    prim_base_path = "/g1"
    camera_pos = (3.8, -5.4, 1.4)
    camera_target = (0.0, 0.0, 0.75)
    root_pos = (0.0, 0.0, 1.0)
    drag_force_debug_path = "/debug/urdf_drag_force"
    drag_force_target_debug_path = "/debug/urdf_drag_force_target"

    def __init__(self, urdf_path: str | Path):
        super().__init__(urdf_path)
        self.urdf_path = self.mjcf_path

    def load_articulation(self):
        data = self.world.load_urdf(self.urdf_path, order=self.order)
        self.obj_id = 0
        config = (
            ke.physics.ArticulationConfig.fixed_base()
            if self.fixed_base
            else ke.physics.ArticulationConfig.free_base()
        )
        self.robot = self.world.add_articulation(
            data,
            env_id=0,
            obj_id=self.obj_id,
            name=self.object_name,
            config=config,
        ).articulation

        self.articulation_visual_view = self.visual.add_articulation_scene_graph(
            0,
            self.obj_id,
            self.urdf_path,
            path=self.prim_base_path,
            order=self.order,
            material=self.standard_materials.pbr,
            collision_path=f"{self.prim_base_path}_collision",
            show_collision=self.show_collision,
            color=(
                None
                if self.visual_color is None
                else np.array(self.visual_color, dtype=np.float32)
            ),
        )
        self.visual_body_prims = self.articulation_visual_view.prims

        self.num_dofs = self.robot.num_dofs()
        self.dof_names = self.world.state.get_obj_dof_names(self.obj_id)
        self.dof_limits = np.asarray(
            self.world.state.get_obj_dof_limits(self.obj_id), dtype=np.float32
        )
        if self.dof_limits.shape != (self.num_dofs, 2):
            raise RuntimeError(
                "G1 URDF DOF limits do not match the created articulation"
            )
        self.targets = self.initial_targets()

    def print_summary(self):
        print(
            f"{self.object_name} loaded: "
            f"links={self.robot.num_links()} dofs={self.num_dofs}"
        )
        print("DOFs:", ", ".join(self.dof_names))
        print("URDF:", self.urdf_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "urdf_path",
        nargs="?",
        default=str(default_urdf_path()),
        help="Path to a Unitree G1 URDF file",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        help="Use a fixed root instead of the default free root.",
    )
    parser.add_argument("--order", default="DFS", choices=("DFS", "BFS"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    class CliUrdfDofControlApp(UrdfDofControlApp):
        fixed_base = args.fixed_base
        order = args.order

    app = CliUrdfDofControlApp(args.urdf_path)
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
