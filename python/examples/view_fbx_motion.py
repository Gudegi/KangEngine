"""Debug skeleton-line playback for an FBX animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import kangengine as ke
from kangengine import asset, imgui, keys, scene


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def default_fbx_file() -> Path:
    return repo_root() / "assets" / "external" / "Capoeira.fbx"


class FbxMotionViewer(ke.App):
    def __init__(
        self,
        fbx_file: Path,
        clip_index: int,
        fps: float,
        scale: float,
        line_radius: float,
    ):
        super().__init__()
        self.fbx_file = str(fbx_file)
        self.clip_index = clip_index
        self.fps = fps
        self.scale = scale
        self.line_radius = line_radius

    def setup(self):
        self.paused = False
        self.playback_speed = 1.0
        self.time = 0.0
        self.line_handle = None

        self.motion = asset.FBXLoader.load_motion(
            self.fbx_file,
            clip_index=self.clip_index,
            fps=self.fps,
            scale=self.scale,
        )
        self.parents = self.motion.parent_indices()
        self.names = self.motion.node_names()

        device = self.getGraphicsDevice()
        vs = package_asset_path("shaders", "common.vs")
        fs = package_asset_path("shaders", "common.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.skeleton_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)

        for shader in (self.skeleton_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

        ground = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(20.0, self.up_axis))
        self.addShape(self.ground_shader, ground)

        camera = self.getCamera()
        camera.set_camera_pos(ke.vec3(0.0, 1.6, 3.8))
        camera.set_target_pos(ke.vec3(0.0, 0.9, 0.0))

        self._apply_time(0.0)
        print(
            f"FBX motion loaded: {self.motion.num_frames()} frames, "
            f"{self.motion.num_joints()} joints, {self.motion.fps():.3f} fps"
        )
        print(f"motion: {self.motion.motion_name()}")
        print(f"fbx: {self.fbx_file}")
        self.checkError()

    def _apply_time(self, time: float):
        state = self.motion.sample(time, loop=True)
        positions = state.compute_global_positions()
        self._update_skeleton_lines(positions)

    def _update_skeleton_lines(self, positions):
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
                colors.append([0.92, 0.92, 0.92, 1.0])

        starts_t = torch.tensor(starts, dtype=torch.float32)
        ends_t = torch.tensor(ends, dtype=torch.float32)
        colors_t = torch.tensor(colors, dtype=torch.float32)

        if self.line_handle is None:
            self.line_handle = scene.DebugDraw.log_lines(
                self,
                self.skeleton_shader,
                "/debug/fbx_skeleton",
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

    def preRender(self):
        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused
        if self.was_key_pressed(keys.R):
            self.time = 0.0
            self._apply_time(self.time)

        if self.paused:
            return

        self.time += (1.0 / 60.0) * max(0.0, self.playback_speed)
        self._apply_time(self.time)
        self.checkError()

    def render(self):
        state = "PAUSED" if self.paused else "running"
        duration = max(self.motion.duration(), 1e-6)
        imgui.begin("FBX Motion")
        imgui.text(f"{Path(self.fbx_file).name}")
        imgui.text(f"Time: {self.time % duration:.3f}s / {duration:.3f}s  |  {state}")
        imgui.text(f"Joints: {self.motion.num_joints()}  FPS: {self.motion.fps():.3f}")
        imgui.text("Space: pause/resume    R: reset")
        _, self.playback_speed = imgui.slider_float(
            "playback speed",
            self.playback_speed,
            0.0,
            4.0,
        )
        imgui.end()

    def postRender(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx-file", default=str(default_fbx_file()))
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--line-radius", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    fbx_file = Path(args.fbx_file).resolve()
    if not fbx_file.exists():
        raise FileNotFoundError(fbx_file)

    app = FbxMotionViewer(
        fbx_file,
        args.clip_index,
        args.fps,
        args.scale,
        args.line_radius,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
