"""FBX character viewer using the C++ SkinnedCharacterBridge binding."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import kangengine as ke
from kangengine import imgui, keys, scene


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def default_fbx_file() -> Path:
    return repo_root() / "assets" / "external" / "fbx/aiming1_subject4.fbx" #"Capoeira.fbx"


def default_bind_file() -> Path:
    return repo_root() / "assets" / "external" / "Geno.fbx"


def resolve_bind_file(fbx_file: Path) -> Path | None:
    candidate = default_bind_file()
    if fbx_file.parent.name == "fbx" and candidate.exists():
        return candidate.resolve()
    return None


class FbxCharacterBridgeViewer(ke.App):
    def __init__(
        self,
        fbx_file: Path,
        bind_file: Path | None,
        clip_index: int,
        fps: float,
        scale: float,
        line_radius: float,
        use_materials: bool,
        material_mode: str,
    ):
        super().__init__()
        self.fbx_file = str(fbx_file)
        self.bind_file = str(bind_file) if bind_file is not None else None
        self.clip_index = clip_index
        self.fps = fps
        self.scale = scale
        self.line_radius = line_radius
        self.use_materials = use_materials
        self.material_mode = material_mode

    def setup(self):
        self.show_mesh = True
        self.show_skeleton = True
        self.cast_shadows = True
        self.animate = True
        self.playback_speed = 1.0
        pg = ke.ColorLibrary.get(ke.ColorType.PASTEL_GREEN)
        self.mesh_color = [pg.r, pg.g, pg.b, pg.a]
        self.time = 0.0
        self.line_handle = None
        self.skeleton_starts = None
        self.skeleton_ends = None
        self.skeleton_colors = None

        device = self.getGraphicsDevice()
        vs = package_asset_path("shaders", "common.vs")
        skinned_vs = package_asset_path("shaders", "skinned_mesh.vs")
        fs = package_asset_path("shaders", "common.fs")
        tex_fs = package_asset_path("shaders", "commonTex.fs")
        debug_checker_fs = package_asset_path("shaders", "debug_checker.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        if self.material_mode == "texture":
            mesh_fs = tex_fs
        elif self.material_mode == "flat":
            mesh_fs = fs
        else:
            mesh_fs = debug_checker_fs

        self.textured_mesh_shader = device.createShaderFromFile(skinned_vs, mesh_fs)
        self.skeleton_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)

        for shader in (
            self.textured_mesh_shader,
            self.skeleton_shader,
            self.ground_shader,
        ):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)
        self.textured_mesh_shader.use()
        self.textured_mesh_shader.setInt("uTexture", 0)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

        ground = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(20.0, self.up_axis))
        self.addShape(self.ground_shader, ground)

        camera = self.getCamera()
        camera.set_camera_pos(ke.vec3(0.0, 1.45, 3.2))
        camera.set_target_pos(ke.vec3(0.0, 0.85, 0.0))

        self.character = ke.SkinnedCharacterBridge.from_fbx(
            self,
            self.textured_mesh_shader,
            self.fbx_file,
            self.bind_file,
            "/fbx_character",
            clip_index=self.clip_index,
            fps=self.fps,
            scale=self.scale,
            use_materials=self.use_materials,
        )
        self.motion = self.character.motion()
        self.parents = self.motion.parent_indices()
        self.names = self.motion.node_names()
        self._update_skeleton_lines(self.motion.sample(0.0, loop=True))
        if not self.use_materials or self.material_mode == "debug_checker":
            self.character.set_color(ke.vec4(*self.mesh_color))
        self._apply_visibility()
        self.character.set_casts_shadow(self.cast_shadows)

        print(
            f"FBX character2 loaded: {Path(self.fbx_file).name} "
            f"bind={Path(self.bind_file).name if self.bind_file else '<same>'} "
            f"meshes={self.character.num_meshes()} "
            f"joints={self.motion.num_joints()} "
            f"frames={self.motion.num_frames()}"
        )
        self.checkError()

    def _update_skeleton_lines(self, state):
        positions = state.compute_global_positions()
        starts = []
        ends = []
        colors = []
        for body_idx, parent_idx in enumerate(self.parents):
            if parent_idx < 0:
                continue
            parent_pos = positions[parent_idx]
            body_pos = positions[body_idx]
            starts.append(
                [float(parent_pos.x), float(parent_pos.y), float(parent_pos.z)]
            )
            ends.append([float(body_pos.x), float(body_pos.y), float(body_pos.z)])

            name = self.names[body_idx].lower()
            if "left" in name:
                colors.append([0.2, 0.45, 1.0, 1.0])
            elif "right" in name:
                colors.append([1.0, 0.25, 0.18, 1.0])
            else:
                colors.append([0.98, 0.98, 0.98, 1.0])

        starts_t = torch.tensor(starts, dtype=torch.float32)
        ends_t = torch.tensor(ends, dtype=torch.float32)
        colors_t = torch.tensor(colors, dtype=torch.float32)
        if not self.show_skeleton:
            colors_t[:, 3] = 0.0

        self.skeleton_starts = starts_t
        self.skeleton_ends = ends_t
        self.skeleton_colors = torch.tensor(colors, dtype=torch.float32)

        if self.line_handle is None:
            self.line_handle = scene.DebugDraw.log_lines(
                self,
                self.skeleton_shader,
                "/debug/fbx_character2_skeleton",
                starts_t,
                ends_t,
                colors_t,
                self.line_radius,
                8,
            )
        else:
            scene.DebugDraw.update_lines(
                self,
                self.line_handle,
                starts_t,
                ends_t,
                colors_t,
            )

    def _apply_visibility(self):
        self.character.set_visible(self.show_mesh)
        if self.line_handle is None:
            return
        colors = self.skeleton_colors.clone()
        colors[:, 3] = 1.0 if self.show_skeleton else 0.0
        scene.DebugDraw.update_lines(
            self,
            self.line_handle,
            self.skeleton_starts,
            self.skeleton_ends,
            colors,
        )

    def preRender(self):
        changed = False
        if self.was_key_pressed(keys.M):
            self.show_mesh = not self.show_mesh
            changed = True
        if self.was_key_pressed(keys.L):
            self.show_skeleton = not self.show_skeleton
            changed = True
        if self.was_key_pressed(keys.H):
            self.cast_shadows = not self.cast_shadows
            self.character.set_casts_shadow(self.cast_shadows)
        if self.was_key_pressed(keys.SPACE):
            self.animate = not self.animate
        if changed:
            self._apply_visibility()
        if self.animate:
            self.time += self.get_delta_time() * max(0.0, self.playback_speed)
            state = self.character.apply_time(self.time)
            self._update_skeleton_lines(state)
        self.checkError()

    def render(self):
        imgui.begin("FBX Character 2")
        imgui.text(f"{Path(self.fbx_file).name}")
        imgui.text(
            f"Meshes: {self.character.num_meshes()}  "
            f"Joints: {self.motion.num_joints()}"
        )
        state = "running" if self.animate else "paused"
        duration = max(self.motion.duration(), 1e-6)
        imgui.text(f"Time: {self.time % duration:.3f}s / {duration:.3f}s  |  {state}")
        imgui.text("Space: pause/resume    M: mesh    L: skeleton    H: shadow")
        mesh_changed, self.show_mesh = imgui.checkbox("show mesh", self.show_mesh)
        skeleton_changed, self.show_skeleton = imgui.checkbox(
            "show skeleton",
            self.show_skeleton,
        )
        shadow_changed, self.cast_shadows = imgui.checkbox(
            "cast shadows",
            self.cast_shadows,
        )
        color_changed = False
        for idx, label in enumerate(("mesh R", "mesh G", "mesh B", "mesh A")):
            changed, value = imgui.slider_float(
                label,
                float(self.mesh_color[idx]),
                0.0,
                1.0,
            )
            if changed:
                self.mesh_color[idx] = value
                color_changed = True
        _, self.animate = imgui.checkbox("animate", self.animate)
        _, self.playback_speed = imgui.slider_float(
            "playback speed",
            self.playback_speed,
            0.0,
            4.0,
        )
        if mesh_changed or skeleton_changed:
            self._apply_visibility()
        if shadow_changed:
            self.character.set_casts_shadow(self.cast_shadows)
        if color_changed:
            self.character.set_color(ke.vec4(*self.mesh_color))
            self._apply_visibility()
        imgui.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx-file", default=str(default_fbx_file()))
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--line-radius", type=float, default=0.008)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--material-mode",
        choices=("debug_checker", "texture", "flat"),
        default="debug_checker",
    )
    parser.add_argument(
        "--no-materials",
        action="store_true",
        help="Ignore FBX material colors/textures and use fallback colors.",
    )
    args = parser.parse_args()

    fbx_file = Path(args.fbx_file).expanduser().resolve()
    if not fbx_file.exists():
        raise FileNotFoundError(fbx_file)
    bind_file = resolve_bind_file(fbx_file)

    app = FbxCharacterBridgeViewer(
        fbx_file,
        bind_file,
        args.clip_index,
        args.fps,
        args.scale,
        args.line_radius,
        not args.no_materials,
        args.material_mode,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
