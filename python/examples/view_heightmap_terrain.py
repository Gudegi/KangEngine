"""View a terrain mesh generated from a heightmap image.

Typical usage:

    python ./python/examples/view_heightmap_terrain.py
    python ./python/examples/view_heightmap_terrain.py assets/external/iceland_heightmap.png --sample-stride 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
from kangengine import imgui


DEFAULT_HEIGHTMAP = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "external"
    / "iceland_heightmap.png"
)


def parse_up_axis(value: str):
    value = value.upper()
    if value == "Y":
        return ke.UpAxis.Y
    if value == "Z":
        return ke.UpAxis.Z
    raise ValueError(f"unsupported up axis: {value}")


class HeightmapTerrainViewer(ke.App):
    def __init__(
        self,
        heightmap_path: Path,
        *,
        up_axis=ke.UpAxis.Y,
        horizontal_scale: float = 1.0,
        height_scale: float = 64.0,
        height_offset: float = -16.0,
        sample_stride: int = 8,
        collision_test: bool = False,
        test_radius: float = 2.0,
        title: str | None = None,
    ):
        super().__init__()
        self.heightmap_path = Path(heightmap_path)
        self.up_axis = up_axis
        self.horizontal_scale = float(horizontal_scale)
        self.height_scale = float(height_scale)
        self.height_offset = float(height_offset)
        self.sample_stride = int(sample_stride)
        self.collision_test = bool(collision_test)
        self.test_radius = float(test_radius)
        self.title = title or f"Heightmap Terrain: {self.heightmap_path.name}"

    def setup(self):
        self.standard_materials = self.create_standard_materials()
        self.set_light_direction(ke.Vec3(-0.35, 0.82, -0.45))
        self.set_light_color(ke.Vec3(1.0, 0.96, 0.9))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.Vec3(0.34, 0.34, 0.34))

        self.mesh = ke.asset.load_heightmap_terrain(
            str(self.heightmap_path),
            self.up_axis,
            horizontal_scale=self.horizontal_scale,
            height_scale=self.height_scale,
            height_offset=self.height_offset,
            sample_stride=self.sample_stride,
        )
        self.material = self.create_phong_material(
            ambient=ke.Vec3(0.12, 0.16, 0.12),
            diffuse=ke.Vec3(0.35, 0.55, 0.32),
            specular=ke.Vec3(0.08, 0.08, 0.08),
            shininess=16.0,
        )
        self.terrain = self.scene.add_mesh("/terrain", self.mesh, self.material)
        self.terrain.set_double_sided(True)

        self.bounds_min, self.bounds_max = self._compute_bounds()
        self.physics = None
        self.collision_added = False
        self.test_actor = None
        self.test_view = None
        if self.collision_test:
            config = (
                ke.physics.PhysicsConfig.z_up()
                if self.up_axis == ke.UpAxis.Z
                else ke.physics.PhysicsConfig.y_up()
            )
            self.physics = ke.physics.PhysicsWorld(config)
            self.timing = self.configure_timing(
                ke.SimulationTimingConfig.from_dt(
                    physics_dt=config.dt,
                    fixed_dt=config.dt,
                    render_hz=60.0,
                )
            )
            self.set_simulation_hotkeys_enabled(True)
            self.collision_added = self.physics.add_heightmap_collision(
                str(self.heightmap_path),
                up_axis=self.up_axis,
                horizontal_scale=self.horizontal_scale,
                height_scale=self.height_scale,
                height_offset=self.height_offset,
                sample_stride=self.sample_stride,
                center=True,
                register_as_ground=True,
                material=ke.physics.PhysicsMaterialDesc([1.0, 1.0, 0.0]),
            )
            if self.collision_test:
                self._create_collision_test_sphere()
        self._setup_camera()

        print(
            "Heightmap terrain loaded: "
            f"{self.heightmap_path} vertices={len(self.mesh.vertices)} "
            f"triangles={len(self.mesh.indices) // 3} "
            f"sample_stride={self.sample_stride} "
            f"collision={'yes' if self.collision_added else 'no'}"
        )

    def fixed_update(self, fixed_dt):
        if self.physics:
            self.physics.step()

    def pre_render(self):
        if self.physics:
            self._sync_collision_test_sphere()

    def render(self):
        imgui.begin(self.title)
        imgui.text(str(self.heightmap_path))
        imgui.text(f"vertices: {len(self.mesh.vertices):,}")
        imgui.text(f"triangles: {len(self.mesh.indices) // 3:,}")
        imgui.text(f"sample stride: {self.sample_stride}")
        imgui.text(f"horizontal scale: {self.horizontal_scale:.3f}")
        imgui.text(f"height scale: {self.height_scale:.3f}")
        imgui.text(f"height offset: {self.height_offset:.3f}")
        imgui.text(f"heightfield collision: {'on' if self.collision_added else 'off'}")
        if self.test_actor is not None:
            imgui.text("collision test sphere: active")
            paused = self.is_simulation_paused()
            changed, paused = imgui.checkbox("pause physics", paused)
            if changed:
                self.set_simulation_paused(paused)
            imgui.text("Enter: play/pause    Space: pause/step")
            if imgui.button("reset sphere"):
                self._reset_collision_test_sphere()
        if self.bounds_min is not None:
            size = self.bounds_max - self.bounds_min
            imgui.text(f"bounds: {size.x:.1f}, {size.y:.1f}, {size.z:.1f}")
        imgui.end()

    def _compute_bounds(self):
        vertices = list(self.mesh.vertices)
        if not vertices:
            return None, None
        min_v = ke.Vec3(vertices[0].x, vertices[0].y, vertices[0].z)
        max_v = ke.Vec3(vertices[0].x, vertices[0].y, vertices[0].z)
        for v in vertices[1:]:
            min_v.x = min(min_v.x, v.x)
            min_v.y = min(min_v.y, v.y)
            min_v.z = min(min_v.z, v.z)
            max_v.x = max(max_v.x, v.x)
            max_v.y = max(max_v.y, v.y)
            max_v.z = max(max_v.z, v.z)
        return min_v, max_v

    def _create_collision_test_sphere(self):
        if self.physics is None or self.bounds_min is None:
            return
        spawn = self._collision_test_spawn_position()
        self.test_actor = self.physics.create_dynamic_sphere(
            self.test_radius,
            [spawn.x, spawn.y, spawn.z],
            [0.0, 0.0, 0.0, 1.0],
            1.0,
        )
        test_mat = self.create_phong_material(
            ambient=ke.Vec3(0.18, 0.08, 0.04),
            diffuse=ke.Vec3(0.95, 0.35, 0.08),
            specular=ke.Vec3(0.1, 0.1, 0.1),
            shininess=24.0,
        )
        sphere_mesh_data = ke.geometry.create_sphere_data(self.test_radius, 24, 12)
        self.test_view = self.scene.add_mesh(
            "/collision_test_sphere", sphere_mesh_data, test_mat
        )
        self._sync_collision_test_sphere()

    def _collision_test_spawn_position(self):
        center = (self.bounds_min + self.bounds_max) * 0.5
        if self.up_axis == ke.UpAxis.Z:
            return ke.Vec3(
                center.x, center.y, self.bounds_max.z + self.test_radius * 6.0
            )
        return ke.Vec3(center.x, self.bounds_max.y + self.test_radius * 6.0, center.z)

    def _reset_collision_test_sphere(self):
        if self.test_actor is None:
            return
        spawn = self._collision_test_spawn_position()
        self.test_actor.set_root_state(
            [spawn.x, spawn.y, spawn.z],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        self._sync_collision_test_sphere()

    def _sync_collision_test_sphere(self):
        if self.test_actor is None or self.test_view is None:
            return
        pos = self.test_actor.get_root_position()
        rot = self.test_actor.get_root_rotation()
        self.test_view.prim.set_local_translation(
            ke.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        )
        self.test_view.prim.set_local_rotation(
            ke.Quat(float(rot[3]), float(rot[1]), float(rot[2]), float(rot[0]))
        )

    def _setup_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.1)
        camera.set_far_plane(10000.0)
        camera.set_fov(55.0)
        if self.bounds_min is None:
            camera.set_target_pos(ke.Vec3(0.0, 0.0, 0.0))
            camera.set_camera_pos(ke.Vec3(0.0, 120.0, 240.0))
            return

        center = (self.bounds_min + self.bounds_max) * 0.5
        size = self.bounds_max - self.bounds_min
        radius = max(size.x, size.y, size.z, 1.0)
        if self.up_axis == ke.UpAxis.Z:
            camera.set_target_pos(center)
            camera.set_camera_pos(
                center + ke.Vec3(radius * 0.35, -radius * 0.75, radius * 0.45)
            )
        else:
            camera.set_target_pos(center)
            camera.set_camera_pos(
                center + ke.Vec3(radius * 0.35, radius * 0.45, radius * 0.75)
            )
        self.set_camera_move_speed(max(10.0, radius * 0.2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a heightmap terrain mesh.")
    parser.add_argument("heightmap", nargs="?", type=Path, default=DEFAULT_HEIGHTMAP)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--horizontal-scale", type=float, default=1.0)
    parser.add_argument("--height-scale", type=float, default=64.0)
    parser.add_argument("--height-offset", type=float, default=-16.0)
    parser.add_argument("--up-axis", choices=("Y", "Z"), default="Y")
    parser.add_argument(
        "--collision-test",
        action="store_true",
        help="Drop a dynamic sphere onto the heightfield to visualize collision.",
    )
    parser.add_argument("--test-radius", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--title", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    heightmap = args.heightmap.expanduser().resolve()
    if not heightmap.exists():
        raise FileNotFoundError(heightmap)

    up_axis = parse_up_axis(args.up_axis)
    app = HeightmapTerrainViewer(
        heightmap,
        up_axis=up_axis,
        horizontal_scale=args.horizontal_scale,
        height_scale=args.height_scale,
        height_offset=args.height_offset,
        sample_stride=args.sample_stride,
        collision_test=args.collision_test,
        test_radius=args.test_radius,
        title=args.title,
    )
    app.initialize(args.width, args.height, False, up_axis)
    app.start()


if __name__ == "__main__":
    main()
