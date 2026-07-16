"""Generate a procedural terrain height field in Python and render it as mesh."""

from __future__ import annotations

import argparse

import numpy as np

import kangengine as ke
from kangengine import imgui, terrain


TERRAIN_TYPES = ("stairs", "slope", "pyramid", "wave", "random", "obstacles")


class ProceduralTerrainViewer(ke.App):
    def __init__(
        self,
        *,
        terrain_type: str = "mixed",
        rows: int = 3,
        cols: int = 3,
        tile_width: int = 96,
        tile_length: int = 96,
        horizontal_scale: float = 0.05,
        vertical_scale: float = 0.005,
        backend: str = "cpp",
        seed: int = 7,
    ):
        super().__init__()
        self.terrain_type = terrain_type
        self.rows = int(rows)
        self.cols = int(cols)
        self.tile_width = int(tile_width)
        self.tile_length = int(tile_length)
        self.horizontal_scale = float(horizontal_scale)
        self.vertical_scale = float(vertical_scale)
        self.backend = backend
        self.seed = int(seed)

    def setup(self):
        self.shaders = self.create_standard_shaders()
        self.set_light_direction(ke.vec3(-0.35, 0.82, -0.45))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.vec3(0.32, 0.32, 0.32))

        self.rng = np.random.default_rng(self.seed)
        self.grid = terrain.TerrainGrid(
            self.rows,
            self.cols,
            self.tile_width,
            self.tile_length,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
        )
        self.grid.fill(self._generate_tile)
        self.mesh = self.grid.to_mesh(up_axis=ke.UpAxis.Y, backend=self.backend)

        self.material = self.create_phong_material(
            shader=self.shaders.phong,
            ambient=ke.vec3(0.10, 0.12, 0.10),
            diffuse=ke.vec3(0.27, 0.28, 0.27),
            specular=ke.vec3(0.06, 0.06, 0.06),
            shininess=12.0,
        )
        self.view = self.scene.add_mesh("/procedural_terrain", self.mesh, self.material)
        self.view.set_double_sided(False)

        self._setup_camera()
        print(
            "Procedural terrain loaded: "
            f"type={self.terrain_type} tiles={self.rows}x{self.cols} "
            f"grid={self.grid.width}x{self.grid.length} "
            f"vertices={len(self.mesh.vertices)} "
            f"triangles={len(self.mesh.indices) // 3}"
        )

    def render(self):
        imgui.begin("Procedural Terrain")
        imgui.text(f"type: {self.terrain_type}")
        imgui.text(f"tiles: {self.rows} x {self.cols}")
        imgui.text(f"tile: {self.tile_width} x {self.tile_length}")
        imgui.text(f"grid: {self.grid.width} x {self.grid.length}")
        imgui.text(f"backend: {self.backend}")
        imgui.text(f"horizontal scale: {self.horizontal_scale:.3f}")
        imgui.text(f"vertical scale: {self.vertical_scale:.4f}")
        imgui.text(f"vertices: {len(self.mesh.vertices):,}")
        imgui.text(f"triangles: {len(self.mesh.indices) // 3:,}")
        imgui.end()

    def _generate_tile(
        self, t: terrain.SubTerrain, tile_row: int, tile_col: int
    ) -> terrain.SubTerrain:
        kind = self._tile_type(tile_row, tile_col)
        if kind == "stairs":
            terrain.stairs_terrain(t, step_width=0.35, step_height=0.08)
        elif kind == "slope":
            direction = -1.0 if (tile_row + tile_col) % 2 else 1.0
            terrain.sloped_terrain(t, slope=direction * 0.18)
        elif kind == "pyramid":
            terrain.pyramid_sloped_terrain(t, slope=0.28, platform_size=1.2)
        elif kind == "wave":
            terrain.wave_terrain(t, num_waves=3.0, amplitude=0.35)
        elif kind == "random":
            terrain.random_uniform_terrain(
                t, -0.05, 0.05, step=0.01, rng=self.rng
            )
        elif kind == "obstacles":
            terrain.discrete_obstacles_terrain(
                t,
                max_height=0.25,
                min_size=0.2,
                max_size=0.7,
                num_rects=80,
                platform_size=1.0,
                rng=self.rng,
            )
        else:
            raise ValueError(f"unknown terrain type: {kind}")
        return t

    def _tile_type(self, row: int, col: int) -> str:
        kind = self.terrain_type.lower()
        if kind != "mixed":
            return kind
        return TERRAIN_TYPES[(row * self.cols + col) % len(TERRAIN_TYPES)]

    def _setup_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.02)
        camera.set_far_plane(1000.0)
        camera.set_fov(55.0)
        radius = max(self.grid.width, self.grid.length) * self.horizontal_scale
        target = ke.vec3(0.0, 0.0, 0.0)
        camera.set_target_pos(target)
        camera.set_camera_pos(
            target + ke.vec3(radius * 0.45, radius * 0.55, radius * 0.85)
        )
        self.set_camera_move_speed(max(1.0, radius * 0.35))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View Python-generated terrain.")
    parser.add_argument(
        "--type",
        default="mixed",
        choices=("mixed", *TERRAIN_TYPES),
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=96)
    parser.add_argument("--tile-length", type=int, default=96)
    parser.add_argument("--horizontal-scale", type=float, default=0.05)
    parser.add_argument("--vertical-scale", type=float, default=0.005)
    parser.add_argument("--backend", choices=("cpp", "python"), default="cpp")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=1000)
    return parser


def main():
    args = build_parser().parse_args()
    app = ProceduralTerrainViewer(
        terrain_type=args.type,
        rows=args.rows,
        cols=args.cols,
        tile_width=args.tile_width,
        tile_length=args.tile_length,
        horizontal_scale=args.horizontal_scale,
        vertical_scale=args.vertical_scale,
        backend=args.backend,
        seed=args.seed,
    )
    app.initialize(args.window_width, args.window_height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
