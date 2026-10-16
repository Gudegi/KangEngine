"""Visualize a heading-aligned ray fan over static stairs; requires kangengine[terrain]."""

import math

import numpy as np
import torch

import kangengine as ke
from kangengine import imgui


GRID_SIZE = 11
MAX_DISTANCE = 6.0
ROOT_XY = (-2.5, 0.0)


class DiagonalRaycastViewer(ke.App):
    def __init__(self):
        super().__init__()
        self.world: ke.physics.PhysicsWorld = ke.physics.PhysicsWorld(
            ke.physics.PhysicsConfig.z_up()
        )
        self.asset: ke.terrain.TerrainMesh = ke.terrain.pyramid_stairs_mesh()
        self.instance: ke.terrain.TerrainInstance = self.asset.create(world=self.world)
        self.root_height = 1.2
        self.heading_degrees = 0.0
        self.hit_count = 0

        grid = torch.linspace(-1.0, 1.0, GRID_SIZE)
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        directions = torch.stack((x, y, -torch.ones_like(x)), dim=-1).reshape(-1, 3)
        self.local_directions = torch.nn.functional.normalize(directions, dim=-1)

        colors = [
            ke.ColorLibrary.get(preset)
            for preset in (ke.ColorType.CYAN, ke.ColorType.ORANGE, ke.ColorType.YELLOW)
        ]
        self.colors = np.array([[c.r, c.g, c.b, c.a] for c in colors], dtype=np.float32)

    def setup(self):
        materials = self.create_standard_materials()
        self.instance.add_visual(
            scene=self.scene, path="/terrain", material=materials.ground
        )
        self.set_camera_view(position=(-8.0, -8.0, 7.0), target=(0.0, 0.0, 0.5))

    def pre_render(self):
        # Rotate the root-local fan around the world Z axis.
        heading = math.radians(self.heading_degrees)
        c, s = math.cos(heading), math.sin(heading)
        rotation = torch.tensor(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        directions = self.local_directions @ rotation.T
        root = torch.tensor((*ROOT_XY, self.root_height))
        hits, distances = self.instance.raycast(
            ray_starts=root, ray_directions=directions, max_distance=MAX_DISTANCE
        )

        # Clip missed rays for display; never send infinite hit positions to rendering.
        valid = torch.isfinite(distances)
        ends = torch.where(valid[:, None], hits, root + directions * MAX_DISTANCE)
        line_colors = np.where(valid.numpy()[:, None], self.colors[0], self.colors[1])
        self.hit_count = int(valid.sum())

        self.debug_overlay.lines(
            path="/scan/rays",
            starts=root.expand_as(ends).numpy(),
            ends=ends.numpy(),
            colors=line_colors,
        )
        self.debug_overlay.points(
            path="/scan/hits",
            points=hits[valid].numpy(),
            colors=self.colors[2:3],
            size=6.0,
        )
        self.debug_overlay.points(
            path="/scan/root",
            points=root[None].numpy(),
            colors=self.colors[2:3],
            size=12.0,
        )

    def render(self):
        imgui.begin("Diagonal terrain scan")
        _, self.root_height = imgui.slider_float(
            "Root Z (m)", self.root_height, 0.8, 3.0
        )
        _, self.heading_degrees = imgui.slider_float(
            "Heading (degrees)", self.heading_degrees, -180.0, 180.0
        )
        imgui.text(f"Hits: {self.hit_count}/{GRID_SIZE * GRID_SIZE}")
        imgui.text("Cyan: hit rays. Orange: misses. Yellow: surface hits and root.")
        imgui.text("Raise the root to widen the footprint on level ground.")
        imgui.end()

    def cleanup(self):
        self.instance.remove()
        self.instance = None
        self.world = None


def main():
    app: DiagonalRaycastViewer = DiagonalRaycastViewer()
    app.initialize(width=1280, height=900, up_axis=ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
