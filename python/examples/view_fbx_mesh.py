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
        self.mesh_prims = []
        self.mesh_handles = []
        self.textures = []
        self.texture_cache = {}
        self.double_sided = True
        self.cast_shadows = True

        device = self.getGraphicsDevice()
        vs = package_asset_path("shaders", "common.vs")
        fs = package_asset_path("shaders", "common.fs")
        tex_fs = package_asset_path("shaders", "commonTex.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.mesh_shader = device.createShaderFromFile(vs, fs)
        self.mesh_texture_shader = device.createShaderFromFile(vs, tex_fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)

        for shader in (self.mesh_shader, self.mesh_texture_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)
            shader.setInt("normalDebugMode", 0)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.62, 0.82, 0.68, 1.0))

        self._configure_lighting()
        self._configure_camera()
        self.meshes = asset.FBXLoader.load_meshes(self.fbx_file, self.scale)
        scene.Prim.define_manipulation_group(self.getScene(), FBX_ROOT_PATH)

        if self.show_ground:
            ground = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
            ground.set_mesh_data(scene.Prim.create_plane_data(4.0, self.up_axis))
            self.addShape(self.ground_shader, ground)

        textured_count = 0
        normal_mapped_count = 0
        for idx, mesh in enumerate(self.meshes):
            prim_path = f"{FBX_ROOT_PATH}/{_safe_prim_name(mesh.name, f'mesh_{idx}')}"
            prim = self.getScene().define_prim(prim_path, scene.PrimType.Mesh)
            prim.set_mesh_data(mesh.mesh_data)
            prim.set_display_color_alpha(_mesh_color(idx))
            self.mesh_prims.append(prim)

            diffuse_texture = self._load_texture(_material_texture_path(mesh, "diffuse"))
            normal_texture = self._load_texture(_material_texture_path(mesh, "normal"))
            shader = self.mesh_texture_shader if diffuse_texture is not None else self.mesh_shader
            handle = self.addShape(shader, prim)
            if diffuse_texture is not None:
                self.setShapeTexture(handle, diffuse_texture, 0)
                textured_count += 1
            if diffuse_texture is not None and normal_texture is not None:
                self.setShapeTexture(handle, normal_texture, 5)
                normal_mapped_count += 1
            self.setShapeDoubleSided(handle, self.double_sided)
            self.setShapeCastsShadow(handle, self.cast_shadows)
            self.mesh_handles.append(handle)

        print(
            f"FBX mesh loaded: {self.fbx_file} "
            f"meshes={len(self.meshes)} "
            f"textured={textured_count} "
            f"normal_mapped={normal_mapped_count}"
        )
        self.checkError()

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
            for handle in self.mesh_handles:
                self.setShapeDoubleSided(handle, self.double_sided)
        if shadow_changed:
            for handle in self.mesh_handles:
                self.setShapeCastsShadow(handle, self.cast_shadows)
        imgui.end()

    def postRender(self):
        pass

    def _configure_lighting(self):
        self.setLightDirection(ke.vec3(-0.35, 0.82, -0.45))
        self.setLightColor(ke.vec3(1.0, 0.94, 0.86))
        self.setLightIntensity(1.1)
        self.setLightAmbient(ke.vec3(0.36, 0.35, 0.32))

    def _configure_camera(self):
        camera = self.getCamera()
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
        texture = self.getGraphicsDevice().createTexture(str(path), True)
        self.texture_cache[key] = texture
        self.textures.append(texture)
        return texture


def _safe_prim_name(name: str, fallback: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    clean = clean.strip("_")
    return clean or fallback


def _mesh_color(index: int):
    palette = [
        ke.vec4(0.70, 0.58, 0.43, 1.0),
        ke.vec4(0.54, 0.47, 0.38, 1.0),
        ke.vec4(0.76, 0.70, 0.60, 1.0),
    ]
    return palette[index % len(palette)]


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
