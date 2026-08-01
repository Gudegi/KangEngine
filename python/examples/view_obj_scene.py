"""Generic OBJ/MTL scene viewer through the component/material path.

This example is the reusable version of the Crytek Sponza viewer.  It uses
``SceneContext.add_obj()`` so imported meshes flow through:

OBJ/MTL -> MeshComponent + MaterialBindingComponent + RenderComponent
        -> SceneResourceManager metadata mirrors.

Typical usage:

    python ./python/examples/view_obj_scene.py --obj-file=./assets/external/Scenes/foo/foo.obj
    python ./python/examples/view_obj_scene.py --obj-file=./assets/external/Scenes/CornellBox/CornellBox-Original.obj --root-path /cornell_box --scale 1.0
    python ./python/examples/view_obj_scene.py --preset=CRYTEK_SPONZA
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import kangengine as ke
from kangengine import imgui

CRYTEK_SPONZA_OBJ = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "external"
    / "Scenes"
    / "crytek_sponza"
    / "sponza.obj"
)


def scene_root_from_path(path: Path) -> str:
    """Return a stable, scene-safe root path derived from an OBJ filename."""

    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in path.stem)
    safe = safe.strip("_") or "obj_scene"
    if safe[0].isdigit():
        safe = f"obj_{safe}"
    return f"/{safe}"


def normalize_root_path(path: str | None, obj_file: Path) -> str:
    if not path:
        return scene_root_from_path(obj_file)
    path = path.strip()
    if not path:
        return scene_root_from_path(obj_file)
    return path if path.startswith("/") else f"/{path}"


def parse_up_axis(value: str):
    value = value.upper()
    if value == "Y":
        return ke.UpAxis.Y
    if value == "Z":
        return ke.UpAxis.Z
    raise ValueError(f"unsupported up axis: {value}")


class ObjSceneViewer(ke.App):
    def __init__(
        self,
        obj_file: Path,
        *,
        root_path: str | None = None,
        title: str | None = None,
        scale: float = 1.0,
        double_sided: bool = True,
        show_ground: bool = False,
        ground_size: float | None = None,
        ground_y: float | None = None,
        light_direction=None,
        light_color=None,
        light_intensity: float = 1.15,
        light_ambient=None,
        camera_pos=None,
        camera_fov: float = 58.0,
    ):
        super().__init__()
        obj_file = Path(obj_file)
        self.obj_file = str(obj_file)
        self.root_path = normalize_root_path(root_path, obj_file)
        self.title = title or f"OBJ Scene: {obj_file.name}"
        self.scale = float(scale)
        self.double_sided = bool(double_sided)
        self.show_ground = bool(show_ground)
        self.ground_size = None if ground_size is None else float(ground_size)
        self.ground_y = None if ground_y is None else float(ground_y)
        self.light_direction = light_direction or ke.vec3(-0.35, 0.82, -0.45)
        self.light_color = light_color or ke.vec3(1.0, 0.96, 0.9)
        self.light_intensity = float(light_intensity)
        self.light_ambient = light_ambient or ke.vec3(0.32, 0.32, 0.32)
        self.camera_pos = camera_pos if camera_pos else None
        self.camera_fov = float(camera_fov)

    def setup(self):
        self.shaders = self.create_standard_shaders()
        self.normal_maps_enabled = True
        self.normal_texture_bindings = []
        self.specular_texture_count = 0
        self.alpha_texture_count = 0

        self._configure_lighting()

        self.import_view = self.scene.add_obj(
            self.root_path,
            self.obj_file,
            double_sided=self.double_sided,
        )
        self.import_view.root.set_local_scale(
            ke.vec3(self.scale, self.scale, self.scale)
        )

        for view in self.import_view:
            material = view.prim.get_material()
            normal_map = getattr(material, "normal_map", None)
            if normal_map is not None:
                self.normal_texture_bindings.append((material, normal_map))
            if getattr(material, "specular_map", None) is not None:
                self.specular_texture_count += 1
            if getattr(material, "alpha_map", None) is not None:
                self.alpha_texture_count += 1

        self.bounds_min, self.bounds_max = compute_obj_bounds(self.import_view.info)
        if self.bounds_min is not None and self.bounds_max is not None:
            self.bounds_min *= self.scale
            self.bounds_max *= self.scale
        self._setup_camera()

        if self.show_ground:
            self._add_ground_from_bounds()

        print(
            "OBJ scene loaded: "
            f"{self.obj_file} root={self.root_path} "
            f"subsets={len(self.import_view)} "
            f"materials={self.import_view.info.material_count} "
            f"textures={len(self.textures)} "
            f"alpha_mapped={self.alpha_texture_count} "
            f"normal_mapped={len(self.normal_texture_bindings)} "
            f"specular_mapped={self.specular_texture_count}"
        )

    def render(self):
        imgui.begin(self.title)
        imgui.text(Path(self.obj_file).name)
        imgui.text(f"root: {self.root_path}")
        imgui.text(f"subsets: {len(self.import_view)}")
        imgui.text(f"materials: {self.import_view.info.material_count}")
        imgui.text(f"textures: {len(self.textures)}")
        imgui.text(f"alpha maps: {self.alpha_texture_count}")
        imgui.text(f"normal maps: {len(self.normal_texture_bindings)}")
        imgui.text(f"specular maps: {self.specular_texture_count}")
        changed, self.normal_maps_enabled = imgui.checkbox(
            "normal maps",
            self.normal_maps_enabled,
        )
        if changed:
            self._apply_normal_map_toggle()
        if self.bounds_min is not None and self.bounds_max is not None:
            size = self.bounds_max - self.bounds_min
            imgui.text(f"bounds: {size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}")
        imgui.end()

    def _configure_lighting(self):
        self.set_light_direction(self.light_direction)
        self.set_light_color(self.light_color)
        self.set_light_intensity(self.light_intensity)
        self.set_light_ambient(self.light_ambient)

    def _setup_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.05)
        camera.set_far_plane(5000.0)
        camera.set_fov(self.camera_fov)
        if self.camera_pos:
            camera.set_camera_pos(self.camera_pos)
            camera.set_target_pos(self.camera_pos + ke.vec3(-3.0, 0.0, 0.0))
            self.set_camera_move_speed(max(5.0, 20.0 * self.scale))
        else:
            if self.bounds_min is None or self.bounds_max is None:
                camera.set_camera_pos(ke.vec3(0.0, 4.0, 14.0))
                camera.set_target_pos(ke.vec3(0.0, 2.0, 0.0))
                self.set_camera_move_speed(max(1.0, 20.0 * self.scale))
                return

            center = (self.bounds_min + self.bounds_max) * 0.5
            size = self.bounds_max - self.bounds_min
            radius = max(float(math.sqrt(float((size * size).sum()))) * 0.5, 1.0)
            camera.set_target_pos(
                ke.vec3(float(center[0]), float(center[1]), float(center[2]))
            )
            camera.set_camera_pos(
                ke.vec3(
                    float(center[0] + radius * 0.15),
                    float(center[1] + radius * 0.10),
                    float(center[2] + radius * 0.85),
                )
            )
            self.set_camera_move_speed(max(1.0, radius * 0.75))

    def _add_ground_from_bounds(self):
        size = None
        ground_y = self.ground_y
        if self.bounds_min is not None and self.bounds_max is not None:
            bounds_size = self.bounds_max - self.bounds_min
            size = max(float(bounds_size[0]), float(bounds_size[2]), 1.0) * 1.2
            if ground_y is None:
                ground_y = float(self.bounds_min[1])

        ground_size = self.ground_size if self.ground_size is not None else size
        if ground_size is None:
            ground_size = max(20.0, 20.0 * self.scale)
        if ground_y is None:
            ground_y = 0.0

        ground = self.add_ground(
            "/ground",
            scale=ground_size,
            shader=self.shaders.ground,
        )
        ground.prim.set_local_translation(ke.vec3(0.0, ground_y, 0.0))

    def _apply_normal_map_toggle(self):
        for material, texture in self.normal_texture_bindings:
            material.normal_map = texture if self.normal_maps_enabled else None
        self.resources.invalidate_usage_cache()


def compute_obj_bounds(info):
    import numpy as np

    mins = []
    maxs = []
    subsets = list(info.subsets)
    meshes = [subset.mesh_data for subset in subsets] if subsets else [info.mesh_data]
    for mesh in meshes:
        vertices = mesh.vertices
        if not vertices:
            continue
        arr = np.asarray([[v.x, v.y, v.z] for v in vertices], dtype=np.float32)
        mins.append(arr.min(axis=0))
        maxs.append(arr.max(axis=0))
    if not mins:
        return None, None
    return np.stack(mins).min(axis=0), np.stack(maxs).max(axis=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View an OBJ/MTL scene.")
    parser.add_argument("--obj-file", type=Path, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--single-sided", action="store_true")
    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--ground-size", type=float, default=None)
    parser.add_argument("--ground-y", type=float, default=None)
    parser.add_argument("--up-axis", choices=("Y", "Z"), default="Y")
    parser.add_argument("--title", default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.preset and not args.obj_file:
        raise RuntimeError("Need to specify --obj-file or --preset")

    preset = args.preset.upper() if args.preset else None
    if preset == "CRYTEK_SPONZA":
        args.obj_file = CRYTEK_SPONZA_OBJ
    elif preset is not None:
        raise ValueError(f"unknown OBJ scene preset: {args.preset}")

    obj_file = args.obj_file.expanduser().resolve()
    if not obj_file.exists():
        raise FileNotFoundError(obj_file)

    app = None
    if preset == "CRYTEK_SPONZA":
        scale = 0.01 if args.scale is None else args.scale
        app = ObjSceneViewer(
            obj_file,
            root_path=args.root_path or "/crytek_sponza",
            title=args.title or "Crytek Sponza",
            scale=scale,
            double_sided=not args.single_sided,
            show_ground=args.ground,
            ground_size=(
                1200.0 * scale if args.ground_size is None else args.ground_size
            ),
            ground_y=(-4.0 * scale if args.ground_y is None else args.ground_y),
            light_direction=ke.vec3(-0.32, 0.93, -0.17),
            light_color=ke.vec3(1.0, 0.94, 0.86),
            light_intensity=1.25,
            light_ambient=ke.vec3(0.38, 0.36, 0.32),
            camera_pos=ke.vec3(5.92, 4.87, -0.96),
            camera_fov=58.0,
        )
    else:
        scale = 1.0 if args.scale is None else args.scale
        app = ObjSceneViewer(
            obj_file,
            root_path=args.root_path,
            title=args.title,
            scale=scale,
            double_sided=not args.single_sided,
            show_ground=args.ground,
            ground_size=args.ground_size,
            ground_y=args.ground_y,
        )
    app.initialize(args.width, args.height, False, parse_up_axis(args.up_axis))
    app.start()


if __name__ == "__main__":
    main()
