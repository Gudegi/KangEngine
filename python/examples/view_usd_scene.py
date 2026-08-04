"""View static USD meshes imported through ke.asset.USDLoader."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import kangengine as ke
from kangengine import asset, imgui


SPONZA_USD = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "external"
    / "Scenes"
    / "main_sponza"
    / "NewSponza_Main_USD_Yup_003.usda"
)


class USDSceneViewer(ke.App):
    def __init__(
        self,
        usd_file: Path,
        scale: float,
        show_ground: bool,
        double_sided: bool,
    ):
        super().__init__()
        self.usd_file = str(usd_file)
        self.scale = scale
        self.show_ground = show_ground
        self.double_sided = double_sided

    def setup(self):
        self.mesh_views = []
        self.normal_maps_enabled = True
        self.normal_texture_bindings = []
        self.bounds_min = None
        self.bounds_max = None
        self.standard_materials = self.create_standard_materials()

        self._configure_lighting()

        self.result = _parse_usd_or_explain(self.usd_file, self.scale)
        self.bounds_min, self.bounds_max = _compute_bounds(self.result.meshes)
        self._frame_camera()

        if self.show_ground:
            self.scene.add_mesh(
                "/ground",
                ke.geometry.create_plane_data(50.0, ke.UpAxis.Y),
                self.standard_materials.ground,
            )

        textured_count = 0
        normal_mapped_count = 0
        for i, mesh in enumerate(self.result.meshes):
            prim_path = "/usd_meshes/" + _safe_prim_name(mesh.name, f"mesh_{i}")
            diffuse_texture = self._load_texture(
                getattr(mesh, "diffuse_texture_path", "")
            )
            normal_texture = self._load_texture(
                getattr(mesh, "normal_texture_path", "")
            )
            material = self._create_usd_material(
                i,
                diffuse_texture,
                normal_texture,
            )
            view = self.scene.add_mesh(
                prim_path,
                mesh.mesh_data,
                material,
                color=ke.Vec4(1.0, 1.0, 1.0, 1.0),
                uri=f"{self.usd_file}#{getattr(mesh, 'prim_path', prim_path)}",
            )
            if diffuse_texture is not None:
                textured_count += 1
            if normal_texture is not None:
                self.normal_texture_bindings.append((material, normal_texture))
                normal_mapped_count += 1
            view.set_double_sided(self.double_sided)
            self.mesh_views.append(view)

        print(
            f"USD loaded: {self.usd_file} meshes={len(self.result.meshes)} "
            f"textured={textured_count} "
            f"normal_mapped={normal_mapped_count} "
            f"warnings={len(self.result.diagnostics.warnings)}"
        )

    def pre_render(self):
        pass

    def render(self):
        imgui.begin("USD")
        imgui.text(Path(self.usd_file).name)
        imgui.text(f"meshes: {len(self.mesh_views)}")
        imgui.text(f"textures: {len(self.textures)}")
        imgui.text(f"normal maps: {len(self.normal_texture_bindings)}")
        changed, self.normal_maps_enabled = imgui.checkbox(
            "normal maps", self.normal_maps_enabled
        )
        if changed:
            self._apply_normal_map_toggle()
        imgui.text(f"warnings: {len(self.result.diagnostics.warnings)}")
        if self.bounds_min is not None and self.bounds_max is not None:
            size = self.bounds_max - self.bounds_min
            imgui.text(f"bounds: {size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}")
        imgui.end()

    def post_render(self):
        pass

    def _configure_lighting(self):
        self.set_light_direction(ke.Vec3(-0.35, 0.82, -0.45))
        self.set_light_color(ke.Vec3(1.0, 0.94, 0.86))
        self.set_light_intensity(1.15)
        self.set_light_ambient(ke.Vec3(0.42, 0.40, 0.36))

    def _frame_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.5)
        camera.set_far_plane(4000.0)
        self.set_camera_move_speed(500.0)
        camera.set_fov(58.0)

        if self.bounds_min is None or self.bounds_max is None:
            camera.set_camera_pos(ke.Vec3(0.0, 35.0, 90.0))
            camera.set_target_pos(ke.Vec3(0.0, 8.0, 0.0))
            return

        center = (self.bounds_min + self.bounds_max) * 0.5
        size = self.bounds_max - self.bounds_min
        radius = max(float(math.sqrt(float((size * size).sum()))) * 0.5, 1.0)
        distance = radius * 1.15
        camera.set_target_pos(
            ke.Vec3(float(center[0]), float(center[1]), float(center[2]))
        )
        camera.set_camera_pos(
            ke.Vec3(
                float(center[0] + distance * 0.35),
                float(center[1] + distance * 0.35),
                float(center[2] + distance * 0.95),
            )
        )

    def _load_texture(self, texture_path: str):
        if not texture_path:
            return None
        path = Path(texture_path)
        if not path.exists():
            return None
        return self.load_texture(path, flip=True)

    def _create_usd_material(self, index: int, diffuse_texture, normal_texture):
        """Create a material-backed surface for one imported USD mesh.

        USDLoader currently exposes resolved diffuse/normal texture paths but
        not full USD Preview Surface factors.  Use a neutral Phong material so
        textured meshes show the texture directly, while untextured meshes get a
        restrained stone palette.
        """
        color = _mesh_color(index)
        has_diffuse = diffuse_texture is not None
        diffuse = (
            ke.Vec3(1.0, 1.0, 1.0)
            if has_diffuse
            else ke.Vec3(float(color.x), float(color.y), float(color.z))
        )
        ambient = (
            ke.Vec3(0.72, 0.72, 0.72)
            if has_diffuse
            else ke.Vec3(
                float(color.x) * 0.65,
                float(color.y) * 0.65,
                float(color.z) * 0.65,
            )
        )
        return self.create_phong_material(
            ambient=ambient,
            diffuse=diffuse,
            specular=ke.Vec3(0.08, 0.08, 0.08),
            shininess=24.0,
            diffuse_map=diffuse_texture,
            normal_map=normal_texture if self.normal_maps_enabled else None,
        )

    def _apply_normal_map_toggle(self):
        for material, texture in self.normal_texture_bindings:
            material.normal_map = texture if self.normal_maps_enabled else None


def _safe_prim_name(name: str, fallback: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    clean = clean.strip("_")
    return clean or fallback


def _mesh_color(index: int):
    # USD material import is still TODO, so use a restrained stone palette
    # instead of vivid per-mesh debug colors.
    palette = [
        ke.Vec4(0.74, 0.70, 0.62, 1.0),
        ke.Vec4(0.66, 0.63, 0.56, 1.0),
        ke.Vec4(0.80, 0.75, 0.66, 1.0),
        ke.Vec4(0.58, 0.56, 0.50, 1.0),
    ]
    return palette[index % len(palette)]


def _compute_bounds(meshes):
    import numpy as np

    mins = []
    maxs = []
    for mesh in meshes:
        vertices = mesh.mesh_data.vertices
        if not vertices:
            continue
        arr = np.asarray([[v.x, v.y, v.z] for v in vertices], dtype=np.float32)
        mins.append(arr.min(axis=0))
        maxs.append(arr.max(axis=0))
    if not mins:
        return None, None
    return np.stack(mins).min(axis=0), np.stack(maxs).max(axis=0)


def _parse_usd_or_explain(usd_file: str, scale: float):
    try:
        return asset.USDLoader.parse(usd_file, scale=scale)
    except RuntimeError as exc:
        message = str(exc)
        if "USD support not compiled" in message:
            raise RuntimeError(
                "view_usd_scene.py requires KangEngine built with -DUSE_USD=ON."
            ) from exc
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_file", type=Path, nargs="?", default=SPONZA_USD)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--double-sided", action="store_true")
    args = parser.parse_args()

    usd_file = args.usd_file.expanduser().resolve()
    if not usd_file.exists():
        raise FileNotFoundError(usd_file)

    app = USDSceneViewer(
        usd_file,
        args.scale,
        show_ground=args.ground,
        double_sided=args.double_sided,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
