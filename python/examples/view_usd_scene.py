"""View static USD meshes imported through ke.asset.USDLoader."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import asset, imgui, render


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
        self.result = _parse_usd_or_explain(self.usd_file, self.scale)
        self.meshes = self.result.meshes
        self.units_per_meter = self.scale / self.result.meters_per_unit

    def setup(self):
        self.mesh_views = []
        self.normal_maps_enabled = True
        self.normal_texture_bindings = []
        self.exposure_compensation = 0.0
        self.camera_exposure = 1.0
        self.light_prims = []
        self._configure_lighting()
        self.camera_keyboard_enabled = False
        self.set_camera_move_speed(1.5 * self.units_per_meter)
        self.bounds_min, self.bounds_max = _compute_bounds(self.meshes)
        self._reset_camera()

        if self.show_ground:
            self.scene.add_mesh(
                "/ground",
                ke.geometry.create_plane_data(
                    50.0 * self.units_per_meter, self.result.up_axis
                ),
                self.create_standard_materials().ground,
            )

        textured_count = 0
        normal_mapped_count = 0
        materials = {}
        for mesh_idx, mesh in enumerate(self.meshes):
            # One USD prim can produce several material subsets. Keep identity
            # in the source URI, without inventing a hierarchy for baked meshes.
            prim_path = f"/usd_meshes/mesh_{mesh_idx}"
            if mesh.material_path not in materials:
                materials[mesh.material_path] = self._create_usd_material(mesh)
            material = materials[mesh.material_path]
            view = self.scene.add_mesh(
                prim_path,
                mesh.mesh_data,
                material,
                color=ke.Vec4(1.0, 1.0, 1.0, 1.0),
                uri=f"{self.usd_file}#{mesh.prim_path}",
            )
            if mesh.diffuse_texture_path:
                textured_count += 1
            if mesh.normal_texture_path:
                normal_mapped_count += 1
            view.set_double_sided(self.double_sided or mesh.double_sided)
            if mesh.opacity_threshold > 0:
                view.set_alpha_mode(render.AlphaMode.MASK, cutoff=mesh.opacity_threshold)
            elif mesh.base_color[3] < 1:
                view.set_alpha_mode(render.AlphaMode.BLEND)
            self.mesh_views.append(view)

        print(
            f"USD loaded: {self.usd_file} meshes={len(self.meshes)} "
            f"textured={textured_count} "
            f"normal_mapped={normal_mapped_count} "
            f"warnings={len(self.result.diagnostics.warnings)}"
        )

    def pre_render(self):
        if not self.keyboard_input_captured:
            self._move_camera(min(self.get_delta_time(), 0.1))

    def _move_camera(self, dt: float):
        camera = self.get_camera()
        forward = _array(camera.get_camera_look_dir())
        up = np.array(
            [0.0, 0.0, 1.0] if self.result.up_axis == ke.UpAxis.Z else [0.0, 1.0, 0.0]
        )
        # Walk parallel to the floor; Q/E provide explicit vertical movement.
        forward -= up * np.dot(forward, up)
        length = np.linalg.norm(forward)
        if length > 1e-6:
            forward /= length
        right = _array(camera.get_camera_right_dir())
        pressed = self.is_key_pressed
        direction = (
            forward * (int(pressed(ke.keys.W)) - int(pressed(ke.keys.S)))
            + right * (int(pressed(ke.keys.D)) - int(pressed(ke.keys.A)))
            + up * (int(pressed(ke.keys.E)) - int(pressed(ke.keys.Q)))
        )
        length = np.linalg.norm(direction)
        if length == 0:
            return
        speed = self.get_camera_move_speed()
        if pressed(ke.keys.LEFT_SHIFT) or pressed(ke.keys.RIGHT_SHIFT):
            speed *= 3
        delta = direction * (speed * dt / length)
        position = _array(camera.get_camera_pos()) + delta
        target = _array(camera.get_target_pos()) + delta
        camera.set_camera_pos(ke.Vec3(*position))
        camera.set_target_pos(ke.Vec3(*target))

    def render(self):
        if self.hide_ui:
            return
        imgui.begin("USD")
        imgui.text(Path(self.usd_file).name)
        imgui.text(f"meshes: {len(self.mesh_views)}")
        imgui.text(f"textures: {len(self.textures)}")
        imgui.text(f"normal maps: {len(self.normal_texture_bindings)}")
        imgui.text(f"USD lights: {len(self.light_prims)}")
        if self.light_prims:
            changed_exposure, self.exposure_compensation = imgui.slider_float(
                "Exposure compensation (stops)", self.exposure_compensation, -8.0, 8.0
            )
            if changed_exposure:
                self._apply_exposure()
        changed, self.normal_maps_enabled = imgui.checkbox(
            "normal maps", self.normal_maps_enabled
        )
        if changed:
            self._apply_normal_map_toggle()
        imgui.text(f"warnings: {len(self.result.diagnostics.warnings)}")
        imgui.text("RMB: look | WASD: walk | Q/E: down/up | Shift: faster")
        if imgui.button("Reset camera"):
            self._reset_camera()
        if imgui.button("Frame scene"):
            self._frame_camera()
        if self.bounds_min is not None and self.bounds_max is not None:
            size = self.bounds_max - self.bounds_min
            imgui.text(f"bounds: {size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}")
        imgui.end()

    def post_render(self):
        pass

    def _configure_lighting(self):
        if self.result.lights:
            self.set_light_intensity(0.0)
            self.set_light_ambient(ke.Vec3(0, 0, 0))
            for source in self.result.lights:
                prim = self.scene.define_prim(
                    path="/lights/usd" + source.prim_path,
                    prim_type=ke.scene.PrimType.LIGHT,
                )
                if source.type == ke.scene.LightType.RECT:
                    prim.set_rect_light(source.rect)
                else:
                    prim.set_dome_light(source.dome)
                prim.set_world_matrix(source.world_transform)
                prim.set_visible(source.visible)
                prim.set_attribute_string("source:usd:primPath", source.prim_path)
                prim.set_attribute_string("source:usd:asset", self.usd_file)
                self.light_prims.append(prim)
            self._apply_exposure()
            return
        direction = (
            (-0.35, -0.45, 0.82)
            if self.result.up_axis == ke.UpAxis.Z
            else (-0.35, 0.82, -0.45)
        )
        self.set_light_direction(ke.Vec3(*direction))
        self.set_light_color(ke.Vec3(1.0, 0.94, 0.86))
        self.set_light_intensity(1.15)
        self.set_light_ambient(ke.Vec3(0.42, 0.40, 0.36))

    def _apply_exposure(self):
        self.set_tone_map(
            mode=render.ToneMapMode.ACES_NARKOWICZ,
            exposure=self.camera_exposure * 2.0 ** self.exposure_compensation,
        )

    def _reset_camera(self):
        if not self.result.cameras:
            self._frame_camera()
            return
        source = self.result.cameras[0]
        if self.result.lights:
            self.camera_exposure = source.linear_exposure_scale
            self.exposure_compensation = 0.0
            self._apply_exposure()
        camera = self.get_camera()
        camera.set_camera_pos(source.position)
        camera.set_target_pos(source.target)
        camera.set_fov(source.vertical_fov)
        camera.set_near_plane(source.near_plane)
        camera.set_far_plane(source.far_plane)

    def _frame_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.02 * self.units_per_meter)
        camera.set_fov(58.0)
        if self.bounds_min is None or self.bounds_max is None:
            camera.set_far_plane(1000.0 * self.units_per_meter)
            return

        center = (self.bounds_min + self.bounds_max) * 0.5
        size = self.bounds_max - self.bounds_min
        radius = max(float(np.linalg.norm(size)) * 0.5, self.units_per_meter)
        distance = radius / math.sin(math.radians(29.0))
        offset = np.array(
            [0.35, -0.95, 0.35]
            if self.result.up_axis == ke.UpAxis.Z
            else [0.35, 0.35, 0.95]
        )
        camera.set_far_plane(max(distance * 4.0, 100.0 * self.units_per_meter))
        camera.set_target_pos(ke.Vec3(*center))
        camera.set_camera_pos(ke.Vec3(*(center + offset * distance)))

    def _load_texture(self, texture_path: str, *, srgb: bool = False):
        if not texture_path:
            return None
        return self.load_texture(texture_path, flip=True, srgb=srgb)

    def _create_usd_material(self, mesh: asset.USDMeshInfo):
        normal_texture = self._load_texture(mesh.normal_texture_path)
        material = self.create_pbr_material(
            base_color=ke.Vec4(*mesh.base_color),
            metallic=mesh.metallic,
            roughness=mesh.roughness,
            emissive_color=ke.Vec3(*mesh.emissive_color),
            emissive_strength=1.0,
            base_color_texture=self._load_texture(
                mesh.diffuse_texture_path, srgb=mesh.diffuse_texture_srgb
            ),
            normal_texture=normal_texture,
            metallic_roughness_texture=self._load_texture(
                mesh.metallic_roughness_texture_path
            ),
            metallic_texture=self._load_texture(mesh.metallic_texture_path),
            roughness_texture=self._load_texture(mesh.roughness_texture_path),
            emissive_texture=self._load_texture(
                mesh.emissive_texture_path, srgb=mesh.emissive_texture_srgb
            ),
        )
        if normal_texture is not None:
            self.normal_texture_bindings.append((material, normal_texture))
        return material

    def _apply_normal_map_toggle(self):
        for material, texture in self.normal_texture_bindings:
            material.normal_texture = texture if self.normal_maps_enabled else None


def _array(vector):
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def _compute_bounds(meshes):
    mins = []
    maxs = []
    for mesh in meshes:
        lower, upper = mesh.bounds
        mins.append(_array(lower))
        maxs.append(_array(upper))
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
    app.initialize(
        width=args.width, height=args.height, hide_ui=False, up_axis=app.result.up_axis
    )
    app.start()


if __name__ == "__main__":
    main()
