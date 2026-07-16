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
    Path(__file__).resolve().parents[2] / "assets" / "external" / "iceland_heightmap.png"
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
        title: str | None = None,
    ):
        super().__init__()
        self.heightmap_path = Path(heightmap_path)
        self.up_axis = up_axis
        self.horizontal_scale = float(horizontal_scale)
        self.height_scale = float(height_scale)
        self.height_offset = float(height_offset)
        self.sample_stride = int(sample_stride)
        self.title = title or f"Heightmap Terrain: {self.heightmap_path.name}"

    def setup(self):
        self.shaders = self.create_standard_shaders()
        self.set_light_direction(ke.vec3(-0.35, 0.82, -0.45))
        self.set_light_color(ke.vec3(1.0, 0.96, 0.9))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.vec3(0.34, 0.34, 0.34))

        self.mesh = ke.asset.load_heightmap_terrain(
            str(self.heightmap_path),
            self.up_axis,
            horizontal_scale=self.horizontal_scale,
            height_scale=self.height_scale,
            height_offset=self.height_offset,
            sample_stride=self.sample_stride,
        )
        self.material = self.create_phong_material(
            shader=self.shaders.phong,
            ambient=ke.vec3(0.12, 0.16, 0.12),
            diffuse=ke.vec3(0.35, 0.55, 0.32),
            specular=ke.vec3(0.08, 0.08, 0.08),
            shininess=16.0,
        )
        self.terrain = self.scene.add_mesh("/terrain", self.mesh, self.material)
        self.terrain.set_double_sided(True)

        self.bounds_min, self.bounds_max = self._compute_bounds()
        self._setup_camera()

        print(
            "Heightmap terrain loaded: "
            f"{self.heightmap_path} vertices={len(self.mesh.vertices)} "
            f"triangles={len(self.mesh.indices) // 3} "
            f"sample_stride={self.sample_stride}"
        )

    def render(self):
        imgui.begin(self.title)
        imgui.text(str(self.heightmap_path))
        imgui.text(f"vertices: {len(self.mesh.vertices):,}")
        imgui.text(f"triangles: {len(self.mesh.indices) // 3:,}")
        imgui.text(f"sample stride: {self.sample_stride}")
        imgui.text(f"horizontal scale: {self.horizontal_scale:.3f}")
        imgui.text(f"height scale: {self.height_scale:.3f}")
        imgui.text(f"height offset: {self.height_offset:.3f}")
        if self.bounds_min is not None:
            size = self.bounds_max - self.bounds_min
            imgui.text(f"bounds: {size.x:.1f}, {size.y:.1f}, {size.z:.1f}")
        imgui.end()

    def _compute_bounds(self):
        vertices = list(self.mesh.vertices)
        if not vertices:
            return None, None
        min_v = ke.vec3(vertices[0].x, vertices[0].y, vertices[0].z)
        max_v = ke.vec3(vertices[0].x, vertices[0].y, vertices[0].z)
        for v in vertices[1:]:
            min_v.x = min(min_v.x, v.x)
            min_v.y = min(min_v.y, v.y)
            min_v.z = min(min_v.z, v.z)
            max_v.x = max(max_v.x, v.x)
            max_v.y = max(max_v.y, v.y)
            max_v.z = max(max_v.z, v.z)
        return min_v, max_v

    def _setup_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.1)
        camera.set_far_plane(10000.0)
        camera.set_fov(55.0)
        if self.bounds_min is None:
            camera.set_target_pos(ke.vec3(0.0, 0.0, 0.0))
            camera.set_camera_pos(ke.vec3(0.0, 120.0, 240.0))
            return

        center = (self.bounds_min + self.bounds_max) * 0.5
        size = self.bounds_max - self.bounds_min
        radius = max(size.x, size.y, size.z, 1.0)
        if self.up_axis == ke.UpAxis.Z:
            camera.set_target_pos(center)
            camera.set_camera_pos(
                center + ke.vec3(radius * 0.35, -radius * 0.75, radius * 0.45)
            )
        else:
            camera.set_target_pos(center)
            camera.set_camera_pos(
                center + ke.vec3(radius * 0.35, radius * 0.45, radius * 0.75)
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
        title=args.title,
    )
    app.initialize(args.width, args.height, False, up_axis)
    app.start()


if __name__ == "__main__":
    main()
