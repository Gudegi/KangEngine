"""Generate a procedural terrain height field in Python and render it as mesh."""

from __future__ import annotations

import argparse

import numpy as np

import kangengine as ke
from kangengine import imgui, keys, terrain


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
        collision_test: bool = True,
        test_bodies_per_type: int = 10,
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
        self.collision_test = bool(collision_test)
        self.test_bodies_per_type = int(test_bodies_per_type)

    def setup(self):
        self.standard_materials = self.create_standard_materials()
        self.set_light_direction(ke.Vec3(-0.35, 0.82, -0.45))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.Vec3(0.32, 0.32, 0.32))

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
            ambient=ke.Vec3(0.10, 0.12, 0.10),
            diffuse=ke.Vec3(0.27, 0.28, 0.27),
            specular=ke.Vec3(0.06, 0.06, 0.06),
            shininess=12.0,
        )
        self.view = self.scene.add_mesh("/procedural_terrain", self.mesh, self.material)
        self.view.set_double_sided(False)

        self.physics = None
        self.collision_added = False
        self.test_bodies = []
        if self.collision_test:
            physics_config = ke.physics.PhysicsConfig.y_up()
            self.physics = ke.physics.PhysicsWorld(physics_config)
            self.timing = self.configure_timing(
                ke.SimulationTimingConfig.from_dt(
                    physics_dt=physics_config.dt,
                    fixed_dt=physics_config.dt,
                    render_hz=60.0,
                )
            )
            self.set_simulation_hotkeys_enabled(True)
            heights = np.ascontiguousarray(self.grid.height_meters(), dtype=np.float32)
            self.collision_added = self.physics.add_heightfield(
                heights.reshape(-1),
                self.grid.width,
                self.grid.length,
                horizontal_scale=self.horizontal_scale,
                up_axis=ke.UpAxis.Y,
                center=True,
                register_as_ground=True,
                material=ke.physics.PhysicsMaterialDesc([1.0, 1.0, 0.0]),
            )
            if self.collision_added:
                self._create_collision_test_bodies()

        self._setup_camera()
        print(
            "Procedural terrain loaded: "
            f"type={self.terrain_type} tiles={self.rows}x{self.cols} "
            f"grid={self.grid.width}x{self.grid.length} "
            f"vertices={len(self.mesh.vertices)} "
            f"triangles={len(self.mesh.indices) // 3} "
            f"collision={'yes' if self.collision_added else 'no'} "
            f"test_bodies={len(self.test_bodies)}"
        )

    def pre_update(self):
        if self.was_key_pressed(keys.R):
            self._reset_collision_test_bodies()

    def fixed_update(self, fixed_dt):
        if self.physics:
            self.physics.step()

    def pre_render(self):
        if self.physics:
            self._sync_collision_test_bodies()

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
        imgui.text(
            f"collision test: {'on' if self.collision_test and self.collision_added else 'off'}"
        )
        imgui.text(f"test bodies: {len(self.test_bodies)}")
        if self.test_bodies:
            paused = self.is_simulation_paused()
            changed, paused = imgui.checkbox("pause physics", paused)
            if changed:
                self.set_simulation_paused(paused)
            imgui.text("Enter: play/pause    Space: pause/step")
            if imgui.button("reset test bodies"):
                self._reset_collision_test_bodies()
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
            terrain.random_uniform_terrain(t, -0.05, 0.05, step=0.01, rng=self.rng)
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

    def _create_collision_test_bodies(self):
        if self.physics is None or not self.collision_added:
            return

        count = max(0, self.test_bodies_per_type)
        if count == 0:
            return

        sphere_material = self.create_phong_material(
            ambient=ke.Vec3(0.12, 0.06, 0.03),
            diffuse=ke.Vec3(0.95, 0.32, 0.08),
            specular=ke.Vec3(0.08, 0.08, 0.08),
            shininess=20.0,
        )
        box_material = self.create_phong_material(
            ambient=ke.Vec3(0.03, 0.07, 0.12),
            diffuse=ke.Vec3(0.12, 0.42, 0.95),
            specular=ke.Vec3(0.08, 0.08, 0.08),
            shininess=20.0,
        )

        radius = max(0.08, self.horizontal_scale * 5.0)
        half = max(0.08, self.horizontal_scale * 4.0)
        sphere_mesh_data = ke.geometry.create_sphere_data(radius, 20, 10)
        box_mesh_data = ke.geometry.create_box_data(half * 2.0, half * 2.0, half * 2.0)

        for index in range(count):
            pos = self._random_spawn_position(index, count * 2)
            actor = self.physics.create_dynamic_sphere(
                radius,
                [pos.x, pos.y, pos.z],
                [0.0, 0.0, 0.0, 1.0],
                1.0,
            )
            view = self.scene.add_mesh(
                f"/collision_test/sphere_{index}", sphere_mesh_data, sphere_material
            )
            self.test_bodies.append(("sphere", actor, view, radius))

        for index in range(count):
            pos = self._random_spawn_position(count + index, count * 2)
            actor = self.physics.create_dynamic_box(
                [half, half, half],
                [pos.x, pos.y, pos.z],
                [0.0, 0.0, 0.0, 1.0],
                1.0,
            )
            view = self.scene.add_mesh(
                f"/collision_test/box_{index}", box_mesh_data, box_material
            )
            self.test_bodies.append(("box", actor, view, half))

        self._sync_collision_test_bodies()

    def _random_spawn_position(self, index: int, total: int):
        width = (self.grid.length - 1) * self.horizontal_scale
        length = (self.grid.width - 1) * self.horizontal_scale
        margin = max(width, length) * 0.12
        x = self.rng.uniform(-width * 0.5 + margin, width * 0.5 - margin)
        z = self.rng.uniform(-length * 0.5 + margin, length * 0.5 - margin)
        max_height = float(np.max(self.grid.height_meters()))
        drop_span = max(2.0, max(width, length) * 0.25)
        y = max_height + 1.0 + drop_span * (0.35 + 0.65 * index / max(total - 1, 1))
        return ke.Vec3(float(x), float(y), float(z))

    def _reset_collision_test_bodies(self):
        total = len(self.test_bodies)
        for index, (_kind, actor, _view, _size) in enumerate(self.test_bodies):
            pos = self._random_spawn_position(index, total)
            actor.set_root_state(
                [pos.x, pos.y, pos.z],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            )
        self._sync_collision_test_bodies()

    def _sync_collision_test_bodies(self):
        for _kind, actor, view, _size in self.test_bodies:
            pos = actor.get_root_position()
            rot = actor.get_root_rotation()
            view.prim.set_local_translation(
                ke.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            )
            view.prim.set_local_rotation(
                ke.Quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2]))
            )

    def _setup_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.02)
        camera.set_far_plane(1000.0)
        camera.set_fov(55.0)
        radius = max(self.grid.width, self.grid.length) * self.horizontal_scale
        target = ke.Vec3(0.0, 0.0, 0.0)
        camera.set_target_pos(target)
        camera.set_camera_pos(
            target + ke.Vec3(radius * 0.45, radius * 0.55, radius * 0.85)
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
    parser.add_argument(
        "--no-collision-test",
        action="store_true",
        help="Disable both PhysX heightfield collision and falling test bodies.",
    )
    parser.add_argument("--test-bodies-per-type", type=int, default=10)
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
        collision_test=not args.no_collision_test,
        test_bodies_per_type=args.test_bodies_per_type,
    )
    app.initialize(args.window_width, args.window_height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
