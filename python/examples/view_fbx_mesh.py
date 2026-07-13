"""View a static FBX mesh imported through ke.asset.FBXLoader."""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
from kangengine import asset, imgui, scene

FBX_ROOT_PATH = "/fbx_static"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def default_fbx_file() -> Path:
    return repo_root() / "assets" / "external" / "Big Vegas" / "Big Vegas.fbx"


class FBXMeshViewer(ke.App):
    def __init__(self, fbx_file: Path, scale: float, show_ground: bool):
        super().__init__()
        self.fbx_file = str(fbx_file)
        self.scale = scale
        self.show_ground = show_ground

    def setup(self):
        self.mesh_views = []
        self.textures = []
        self.texture_cache = {}
        self.double_sided = True
        self.cast_shadows = True

        device = self.get_renderer().device()
        vs = package_asset_path("shaders", "common.vs")
        fs = package_asset_path("shaders", "common.fs")
        tex_fs = package_asset_path("shaders", "commonTex.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.mesh_shader = device.create_shader_from_file(vs, fs)
        self.mesh_texture_shader = device.create_shader_from_file(vs, tex_fs)
        self.ground_shader = device.create_shader_from_file(vs, checker_fs)

        for shader in (self.mesh_shader, self.mesh_texture_shader, self.ground_shader):
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)
            shader.set_uniform_block_binding("shadowUBO", 2)
            shader.set_int("normalDebugMode", 0)

        self.ground_shader.use()
        self.ground_shader.set_vec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        dark_green = ke.preset_rgba(ke.ColorType.DARK_GREEN)
        self.ground_shader.set_vec4("checkerColor2", ke.vec4(*dark_green))

        self._configure_lighting()
        self._configure_camera()
        self.meshes = asset.FBXLoader.load_meshes(self.fbx_file, self.scale)
        scene.Prim.define_manipulation_group(self.get_scene(), FBX_ROOT_PATH)

        if self.show_ground:
            self.scene.add_mesh(
                "/ground",
                scene.Prim.create_plane_data(4.0, self.up_axis),
                self.ground_shader,
            )

        textured_count = 0
        normal_mapped_count = 0
        used_mesh_names = set()
        for idx, mesh in enumerate(self.meshes):
            mesh_name = _unique_prim_name(
                _safe_prim_name(mesh.name, f"mesh_{idx}"),
                used_mesh_names,
            )
            prim_path = (
                f"{FBX_ROOT_PATH}/{mesh_name}"
            )
            diffuse_path = _material_texture_path(mesh, "diffuse")
            diffuse_texture = self._load_texture(diffuse_path)
            normal_texture = self._load_texture(
                _material_texture_path(mesh, "normal")
            )
            shader = (
                self.mesh_texture_shader
                if diffuse_texture is not None
                else self.mesh_shader
            )
            view = self.scene.add_mesh(
                prim_path,
                mesh.mesh_data,
                shader,
                color=_mesh_material_color(mesh, _mesh_color(idx)),
            )
            if diffuse_texture is not None:
                view.set_texture(diffuse_texture, 0)
                if Path(diffuse_path).suffix.lower() in {".png", ".tga"}:
                    # FBX does not reliably describe alpha semantics. Atlas
                    # textures in these formats commonly use binary cutouts;
                    # Mask keeps depth writes and clips their empty texels.
                    view.set_alpha_mode(ke.AlphaMode.Mask, 0.5)
                textured_count += 1
            if diffuse_texture is not None and normal_texture is not None:
                view.set_texture(normal_texture, 5)
                normal_mapped_count += 1
            view.set_double_sided(self.double_sided)
            view.set_casts_shadow(self.cast_shadows)
            self.mesh_views.append(view)

        print(
            f"FBX mesh loaded: {self.fbx_file} "
            f"meshes={len(self.meshes)} "
            f"textured={textured_count} "
            f"normal_mapped={normal_mapped_count}"
        )
        self.check_error()

    def preRender(self):
        pass

    def render(self):
        imgui.begin("FBX Mesh")
        imgui.text(Path(self.fbx_file).name)
        imgui.text(f"meshes: {len(self.meshes)}")
        imgui.text(f"textures: {len(self.textures)}")
        double_changed, self.double_sided = imgui.checkbox(
            "double sided", self.double_sided
        )
        shadow_changed, self.cast_shadows = imgui.checkbox(
            "cast shadows", self.cast_shadows
        )
        if double_changed:
            for view in self.mesh_views:
                view.set_double_sided(self.double_sided)
        if shadow_changed:
            for view in self.mesh_views:
                view.set_casts_shadow(self.cast_shadows)
        imgui.end()

    def postRender(self):
        pass

    def _configure_lighting(self):
        self.set_light_direction(ke.vec3(-0.35, 0.82, -0.45))
        self.set_light_color(ke.vec3(1.0, 0.94, 0.86))
        self.set_light_intensity(1.1)
        self.set_light_ambient(ke.vec3(0.36, 0.35, 0.32))

    def _configure_camera(self):
        camera = self.get_camera()
        camera.set_near_plane(0.01)
        camera.set_far_plane(100.0)
        camera.set_fov(48.0)
        self.set_camera_move_speed(1.0)
        camera.set_camera_pos(ke.vec3(1.5, 1.0, 2.0))
        camera.set_target_pos(ke.vec3(0.0, 0.4, 0.0))

    def _load_texture(self, texture_path: str):
        if not texture_path:
            return None
        path = Path(texture_path)
        if not path.exists():
            return None
        key = str(path.resolve())
        if key in self.texture_cache:
            return self.texture_cache[key]
        texture = self.load_texture(path, flip=True)
        self.texture_cache[key] = texture
        return texture


def _safe_prim_name(name: str, fallback: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    clean = clean.strip("_")
    return clean or fallback


def _unique_prim_name(name: str, used_names: set[str]) -> str:
    candidate = name
    suffix = 1
    while candidate in used_names:
        candidate = f"{name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _mesh_color(index: int):
    palette = [
        ke.vec4(0.70, 0.58, 0.43, 1.0),
        ke.vec4(0.54, 0.47, 0.38, 1.0),
        ke.vec4(0.76, 0.70, 0.60, 1.0),
    ]
    return palette[index % len(palette)]


def _mesh_material_color(mesh, fallback):
    material = _primary_material(mesh)
    if material is None or material.diffuse_color is None:
        return fallback
    color = material.diffuse_color
    return ke.vec4(
        float(color[0]), float(color[1]), float(color[2]), float(color[3])
    )


def _primary_material(mesh):
    materials = list(mesh.materials)
    primary = int(mesh.primary_material_index)
    if 0 <= primary < len(materials):
        return materials[primary]
    return None


def _material_texture_path(mesh, texture_kind: str) -> str:
    material = _primary_material(mesh)
    if material is None:
        return ""
    if texture_kind == "diffuse" and material.has_diffuse_texture:
        return material.diffuse_texture_path
    if texture_kind == "normal" and material.has_normal_texture:
        return material.normal_texture_path
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx-file", default=str(default_fbx_file()))
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-ground", action="store_true")
    args = parser.parse_args()

    fbx_file = Path(args.fbx_file).expanduser().resolve()
    if not fbx_file.exists():
        raise FileNotFoundError(fbx_file)

    app = FBXMeshViewer(fbx_file, args.scale, not args.no_ground)
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
