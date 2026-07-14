"""View the Crytek Sponza OBJ scene through the component/material path.

This example intentionally uses ``SceneContext.add_obj()`` instead of manually
registering renderer handles.  The import path is:

OBJ/MTL -> MeshComponent + MaterialBindingComponent + RenderComponent
        -> SceneResourceManager metadata mirrors.
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


class CrytekSponzaViewer(ke.App):
    def __init__(
        self,
        obj_file: Path,
        scale: float,
        double_sided: bool,
        show_ground: bool,
    ):
        super().__init__()
        self.obj_file = str(obj_file)
        self.scale = float(scale)
        self.double_sided = bool(double_sided)
        self.show_ground = bool(show_ground)

    def setup(self):
        self.shaders = self.create_standard_shaders()
        self.normal_maps_enabled = True
        self.normal_texture_bindings = []

        self._configure_lighting()

        self.import_view = self.scene.add_obj(
            "/crytek_sponza",
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

        self.bounds_min, self.bounds_max = _compute_bounds(self.import_view.info)
        if self.bounds_min is not None and self.bounds_max is not None:
            self.bounds_min *= self.scale
            self.bounds_max *= self.scale
        self._frame_camera()

        if self.show_ground:
            ground = self.add_ground(
                "/ground",
                scale=1200.0 * self.scale,
                shader=self.shaders.ground,
            )
            ground.prim.set_local_translation(ke.vec3(0.0, -4.0 * self.scale, 0.0))

        print(
            "Crytek Sponza loaded: "
            f"{self.obj_file} subsets={len(self.import_view)} "
            f"materials={self.import_view.info.material_count} "
            f"textures={len(self.textures)} "
            f"normal_mapped={len(self.normal_texture_bindings)}"
        )

    def render(self):
        imgui.begin("Crytek Sponza")
        imgui.text(Path(self.obj_file).name)
        imgui.text(f"subsets: {len(self.import_view)}")
        imgui.text(f"materials: {self.import_view.info.material_count}")
        imgui.text(f"textures: {len(self.textures)}")
        imgui.text(f"normal maps: {len(self.normal_texture_bindings)}")
        changed, self.normal_maps_enabled = imgui.checkbox(
            "normal maps",
            self.normal_maps_enabled,
        )
        if changed:
            self._apply_normal_map_toggle()
        if self.bounds_min is not None and self.bounds_max is not None:
            size = self.bounds_max - self.bounds_min
            imgui.text(f"bounds: {size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}")
        imgui.end()

    def _configure_lighting(self):
        self.set_light_direction(ke.vec3(-0.35, 0.82, -0.45))
        self.set_light_color(ke.vec3(1.0, 0.94, 0.86))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.vec3(0.38, 0.36, 0.32))

    def _frame_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.05)
        camera.set_far_plane(5000.0)
        camera.set_fov(58.0)
        self.set_camera_move_speed(max(20.0, 1200.0 * self.scale))

        if self.bounds_min is None or self.bounds_max is None:
            camera.set_camera_pos(ke.vec3(0.0, 4.0, 14.0))
            camera.set_target_pos(ke.vec3(0.0, 2.0, 0.0))
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

    def _apply_normal_map_toggle(self):
        for material, texture in self.normal_texture_bindings:
            material.normal_map = texture if self.normal_maps_enabled else None


def _compute_bounds(info):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_file", type=Path, nargs="?", default=CRYTEK_SPONZA_OBJ)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--single-sided", action="store_true")
    parser.add_argument("--ground", action="store_true")
    args = parser.parse_args()

    obj_file = args.obj_file.expanduser().resolve()
    if not obj_file.exists():
        raise FileNotFoundError(obj_file)

    app = CrytekSponzaViewer(
        obj_file,
        scale=args.scale,
        double_sided=not args.single_sided,
        show_ground=args.ground,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
