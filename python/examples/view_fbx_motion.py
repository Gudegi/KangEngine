"""Debug skeleton-line playback for an FBX animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import kangengine as ke
from kangengine import asset, imgui, keys


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
        self.line_view = None

        self.motion = asset.FBXLoader.load_motion(
            self.fbx_file,
            clip_index=self.clip_index,
            fps=self.fps,
            scale=self.scale,
        )
        self.editor = ke.motion_module.MotionEditor(self.motion)
        self.parents = self.motion.parent_indices()
        self.names = self.motion.node_names()

        device = self.get_renderer().device()
        vs = package_asset_path("shaders", "common.vs")
        fs = package_asset_path("shaders", "common.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.skeleton_shader = device.create_shader_from_file(vs, fs)
        self.ground_shader = device.create_shader_from_file(vs, checker_fs)

        for shader in (self.skeleton_shader, self.ground_shader):
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)
            shader.set_uniform_block_binding("shadowUBO", 2)

        self.ground_shader.use()
        self.ground_shader.set_vec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.set_vec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

        self.editor.add_module(
            ke.motion_module.RootTrajectoryModule(
                self,
                "/debug/fbx_motion_root_trajectory",
                line_width=2.0,
                point_size=8.0,
            )
        )
        self.editor.add_module(
            ke.motion_module.TrackingModule(
                self,
                "/debug/fbx_motion_tracking",
                line_width=2.0,
                point_size=8.0,
            )
        )
        self.editor.add_module(
            ke.motion_module.ContactModule(
                self,
                "/debug/fbx_motion_contacts",
                point_size=11.0,
            )
        )

        self.scene.add_ground(scale=20.0, shader=self.ground_shader)

        camera = self.get_camera()
        camera.set_camera_pos(ke.vec3(0.0, 1.6, 3.8))
        camera.set_target_pos(ke.vec3(0.0, 0.9, 0.0))

        self._apply_time(0.0)
        print(
            f"FBX motion loaded: {self.motion.num_frames()} frames, "
            f"{self.motion.num_joints()} joints, {self.motion.fps():.3f} fps"
        )
        print(f"motion: {self.motion.motion_name()}")
        print(f"fbx: {self.fbx_file}")
        self.check_error()

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

        if self.line_view is None:
            self.line_view = self.scene.log_lines(
                "/debug/fbx_skeleton",
                self.skeleton_shader,
                starts_t,
                ends_t,
                colors_t,
                self.line_radius,
                8,
            )
        else:
            self.line_view.update_lines(starts_t, ends_t, colors_t)

    def preRender(self):
        if self.was_key_pressed(keys.SPACE):
            self.editor.player.playing = not self.editor.player.playing
        if self.was_key_pressed(keys.R):
            self.editor.reset()
            self._apply_time(self.editor.player.time)

        if self.editor.update(self.get_delta_time()):
            self._apply_time(self.editor.player.time)
        self.check_error()

    def render(self):
        imgui.begin("FBX Motion")
        imgui.text(f"Joints: {self.motion.num_joints()}  FPS: {self.motion.fps():.3f}")
        imgui.text("Space: pause/resume    R: reset")
        imgui.end()
        if self.editor.render_panel(Path(self.fbx_file).name):
            self._apply_time(self.editor.player.time)

    def postRender(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx-file", default=str(default_fbx_file()))
    parser.add_argument("--clip-index", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=-1.0)
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
