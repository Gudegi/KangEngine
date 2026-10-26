"""Inspect rigid stair geometry with independent poses and surface ray hits."""

import math

import numpy as np
import torch

import kangengine as ke
from kangengine import imgui
from kangengine.asset import load_obj_rigid, rigid_from_terrain_mesh
from kangengine.sim import GridCloner, KangSimWorld
from kangengine.terrain import pyramid_stairs_mesh
from kangengine.visual.sim import SimWorldVisualizer


# Set two OBJ paths to replace the built-in procedural stairs.
OBJ_PATHS = ()
GRID_SIZE = 11


class RigidAssetViewer(ke.App):
    def __init__(self):
        super().__init__()
        self.world: KangSimWorld = KangSimWorld(num_envs=2)
        self.cloner = GridCloner(self.world, spacing=10.0, columns=1)
        assets = (
            [load_obj_rigid(path) for path in OBJ_PATHS]
            if OBJ_PATHS
            else [
                rigid_from_terrain_mesh(pyramid_stairs_mesh(step_height=0.1)),
                rigid_from_terrain_mesh(pyramid_stairs_mesh(step_height=0.18)),
            ]
        )
        self.rigids = self.cloner.add_rigid(assets, obj_id=0, kinematic=True)
        self.visualizer = None
        self.height = 0.0
        self.yaw = 0.0
        self.pitch = 0.0

        grid = torch.linspace(-1.0, 1.0, GRID_SIZE)
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        self.directions = torch.stack((x, y, -torch.ones_like(x)), -1).reshape(-1, 3)
        self.starts = self.cloner.env_origins + torch.tensor([1.0, 0.0, 2.5])
        color = ke.ColorLibrary.get(ke.ColorType.CYAN)
        color2 = ke.ColorLibrary.get(ke.ColorType.RED)
        self.ray_color = np.array([[color.r, color.g, color.b, color.a]], np.float32)
        self.ray_color2 = np.array(
            [[color2.r, color2.g, color2.b, color2.a]], np.float32
        )

    def setup(self):
        self.set_light_direction(ke.Vec3(0.5, 0.7, -1.0))
        self.set_light_intensity(0.9)
        self.set_light_ambient(ke.Vec3(0.3, 0.3, 0.3))

        self.visualizer = SimWorldVisualizer(self, self.world)
        self.visualizer.add(self.rigids, path="/rigids")
        self.set_camera_view(position=(-12.0, -16.0, 13.0), target=(0.0, 5.0, 0.4))

    def fixed_update(self, fixed_dt):
        self.world.advance(fixed_dt)

    def pre_render(self):
        self.visualizer.sync()
        hits, distances = self.rigids.raycast(
            self.starts[:, None, :], self.directions[None, :, :], max_distance=8.0
        )
        valid = torch.isfinite(distances)
        starts = self.starts[:, None, :].expand_as(hits)
        self.debug_overlay.lines(
            path="/scan/rays",
            starts=starts[valid].numpy(),
            ends=hits[valid].numpy(),
            colors=self.ray_color,
        )
        self.debug_overlay.points(
            path="/scan/hits",
            points=hits[valid].numpy(),
            colors=self.ray_color2,
            size=6.0,
        )

    def render(self):
        imgui.begin("Environment 1 rigid pose")
        height_changed, self.height = imgui.slider_float(
            "Height (m)", self.height, 0, 1
        )
        yaw_changed, self.yaw = imgui.slider_float("Yaw (degrees)", self.yaw, -90, 90)
        pitch_changed, self.pitch = imgui.slider_float(
            "Pitch (degrees)", self.pitch, -20, 20
        )
        imgui.text(
            "Environment 0 stays fixed. Rays follow each environment's own asset."
        )
        imgui.end()

        if height_changed or yaw_changed or pitch_changed:
            # Reset only environment 1; the collision and query share this rigid pose.
            half_yaw = math.radians(self.yaw) / 2
            half_pitch = math.radians(self.pitch) / 2
            sy, cy = math.sin(half_yaw), math.cos(half_yaw)
            sp, cp = math.sin(half_pitch), math.cos(half_pitch)
            rotation = torch.tensor([[-sy * sp, cy * sp, sy * cp, cy * cp]])
            position = self.cloner.env_origins[1:2].clone()
            position[:, 2] = self.height
            self.rigids.set_root_state([1], position, rotation, immediate=True)

    def cleanup(self):
        if self.visualizer is not None:
            self.visualizer.release()
        self.world.release()


def main():
    app: RigidAssetViewer = RigidAssetViewer()
    app.initialize(width=1280, height=900, up_axis=ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
