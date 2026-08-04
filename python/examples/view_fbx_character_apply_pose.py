"""Replay an FBX motion by directly applying root/local-quaternion poses."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import kangengine as ke
from kangengine import asset, imgui, keys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_fbx_file() -> Path:
    return repo_root() / "assets" / "external" / "Capoeira.fbx"


def quat_wxyz_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dots = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dots < 0.0, -q1, q1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    close = dots > 0.9995
    lerp = q0 + alpha * (q1 - q0)
    lerp /= np.maximum(np.linalg.norm(lerp, axis=-1, keepdims=True), 1e-8)

    theta_0 = np.arccos(dots)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dots * sin_theta / np.maximum(sin_theta_0, 1e-8)
    s1 = sin_theta / np.maximum(sin_theta_0, 1e-8)
    slerp = s0 * q0 + s1 * q1
    slerp /= np.maximum(np.linalg.norm(slerp, axis=-1, keepdims=True), 1e-8)
    return np.where(close, lerp, slerp).astype(np.float32)


class FbxCharacterApplyPoseViewer(ke.App):
    def __init__(
        self,
        fbx_file: Path,
        bind_file: Path | None,
        clip_index: int,
        fps: float,
        scale: float,
        line_radius: float,
        material_mode: str,
        interpolate: bool,
    ):
        super().__init__()
        self.fbx_file = str(fbx_file)
        self.bind_file = str(bind_file) if bind_file is not None else None
        self.clip_index = clip_index
        self.fps = fps
        self.scale = scale
        self.line_radius = line_radius
        self.material_mode = material_mode
        self.interpolate = interpolate

    def setup(self):
        self.time = 0.0
        self.playing = True
        self.show_mesh = True
        self.show_skeleton = True
        self.line_view = None
        self.skeleton_starts = None
        self.skeleton_ends = None
        self.skeleton_colors = None

        self.standard_materials = self.create_standard_materials()
        self.mesh_shader = (
            self.standard_materials.skinned_texture
            if self.material_mode == "texture"
            else self.standard_materials.skinned_debug_checker
        )
        self.skeleton_shader = self.standard_materials.common

        self.add_ground(material=self.standard_materials.ground)
        self.set_camera_view([0.0, 1.45, 3.2], [0.0, 0.85, 0.0])

        self.character = ke.visual.SkinVisual.from_fbx(
            self,
            self.mesh_shader,
            self.fbx_file,
            self.bind_file,
            "/fbx_pose_character",
            clip_index=self.clip_index,
            fps=self.fps,
            scale=self.scale,
            use_materials=self.material_mode == "texture",
        )
        if self.material_mode != "texture":
            self.character.set_color(
                ke.vec4(*ke.preset_rgba(ke.ColorType.PASTEL_PURPLE))
            )

        self.motion = self.character.motion()
        self.parents = self.motion.parent_indices()
        self.names = self.motion.node_names()
        self.root_translations, self.local_rotations_wxyz = (
            self._extract_motion_pose_arrays()
        )

        self._apply_pose_frame(0)
        self._update_skeleton_lines(self.motion.frame(0))
        self._apply_visibility()
        self._print_import_info()
        self.check_error()

    def _motion_root_translation(self, frame: int) -> np.ndarray:
        root = self.motion.root_translation(frame)
        return np.array([root.x, root.y, root.z], dtype=np.float32)

    def _extract_motion_pose_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.motion.num_frames()
        joints = self.motion.num_joints()
        roots = np.zeros((frames, 3), dtype=np.float32)
        rotations = np.asarray(
            self.motion.local_rotations_wxyz_flat(),
            dtype=np.float32,
        ).reshape(frames, joints, 4)
        for frame in range(frames):
            roots[frame] = self._motion_root_translation(frame)
        return roots, rotations.copy()

    def _looped_frame_float(self, time: float) -> float:
        frames = self.motion.num_frames()
        fps = max(float(self.motion.fps()), 1e-6)
        if frames <= 1:
            return 0.0

        playback_duration = frames / fps
        wrapped = math.fmod(time, playback_duration)
        if wrapped < 0.0:
            wrapped += playback_duration
        return wrapped * fps

    def _sample_pose_nearest(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        frame = min(
            self.motion.num_frames() - 1,
            max(0, int(self._looped_frame_float(time))),
        )
        return self.root_translations[frame], self.local_rotations_wxyz[frame]

    def _sample_pose_interpolated(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        frames = self.motion.num_frames()
        if frames <= 1:
            return self.root_translations[0], self.local_rotations_wxyz[0]

        frame_float = self._looped_frame_float(time)
        i0 = min(frames - 1, max(0, int(math.floor(frame_float))))
        i1 = min(frames - 1, i0 + 1)
        alpha = 0.0 if i0 == i1 else frame_float - float(i0)

        root = (
            self.root_translations[i0] * (1.0 - alpha)
            + self.root_translations[i1] * alpha
        ).astype(np.float32)
        rotations = quat_wxyz_slerp(
            self.local_rotations_wxyz[i0],
            self.local_rotations_wxyz[i1],
            alpha,
        )
        return root, rotations

    def _sample_pose(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        if self.interpolate:
            return self._sample_pose_interpolated(time)
        return self._sample_pose_nearest(time)

    def _apply_pose_at_time(self, time: float):
        root, rotations = self._sample_pose(time)
        return self.character.apply_pose(root, rotations)

    def _apply_pose_frame(self, frame: int):
        return self.character.apply_pose(
            self.root_translations[frame],
            self.local_rotations_wxyz[frame],
        )

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

        if self.line_view is None:
            self.line_view = self.scene.log_lines(
                "/debug/fbx_apply_pose_skeleton",
                self.skeleton_shader,
                starts_t,
                ends_t,
                colors_t,
                self.line_radius,
                8,
            )
        else:
            self.line_view.update_lines(starts_t, ends_t, colors_t)

    def _apply_visibility(self):
        self.character.set_visible(self.show_mesh)
        if self.line_view is None:
            return
        colors = self.skeleton_colors.clone()
        colors[:, 3] = 1.0 if self.show_skeleton else 0.0
        self.line_view.update_lines(
            self.skeleton_starts,
            self.skeleton_ends,
            colors,
        )

    def _print_import_info(self):
        clips = asset.FBXLoader.load_animation_clip_infos(self.fbx_file)
        print(f"FBX apply-pose character loaded: {Path(self.fbx_file).name}")
        print("load: SkinVisual.from_fbx(...)")
        print(
            "pose path: motion -> root_translations/local_rotations_wxyz -> sampled pose -> character.apply_pose(...)"
        )
        print(f"pose sampling: {'interpolated' if self.interpolate else 'nearest'}")
        print("skeleton path: original motion.sample(time)")
        print(f"root_translations shape: {self.root_translations.shape}")
        print(f"local_rotations_wxyz shape: {self.local_rotations_wxyz.shape}")
        print(
            f"motion reference: {self.motion.motion_name()} "
            f"joints={self.motion.num_joints()} frames={self.motion.num_frames()} "
            f"fps={self.motion.fps():g}"
        )

        for idx, clip in enumerate(clips):
            selected = " <- selected" if clip.name == self.motion.motion_name() else ""
            print(
                f"  clip[{idx}] {clip.name} "
                f"duration={clip.end_time - clip.start_time:.3f}s "
                f"frame_rate={clip.frame_rate:g}{selected}"
            )

    def preRender(self):
        changed = False
        if self.was_key_pressed(keys.M):
            self.show_mesh = not self.show_mesh
            changed = True
        if self.was_key_pressed(keys.L):
            self.show_skeleton = not self.show_skeleton
            changed = True
        if self.was_key_pressed(keys.SPACE):
            self.playing = not self.playing
        if self.was_key_pressed(keys.I):
            self.interpolate = not self.interpolate
        if changed:
            self._apply_visibility()

        if self.playing:
            self.time += self.get_delta_time()
            self._apply_pose_at_time(self.time)
            self._update_skeleton_lines(self.motion.sample(self.time, loop=True))
        self.check_error()

    def render(self):
        imgui.begin("FBX Apply Pose")
        imgui.text(f"{Path(self.fbx_file).name}")
        imgui.text(
            f"Meshes: {self.character.num_meshes()}  Joints: {self.motion.num_joints()}"
        )
        state = "running" if self.playing else "paused"
        imgui.text(f"Time: {self.time:.3f}s  |  {state}")
        sampling = "interpolated" if self.interpolate else "nearest"
        imgui.text(f"Pose sampling: {sampling}")
        imgui.text("Space: pause/resume    I: interpolation    M: mesh    L: skeleton")
        mesh_changed, self.show_mesh = imgui.checkbox("show mesh", self.show_mesh)
        skeleton_changed, self.show_skeleton = imgui.checkbox(
            "show skeleton",
            self.show_skeleton,
        )
        interpolate_changed, self.interpolate = imgui.checkbox(
            "interpolate pose",
            self.interpolate,
        )
        if mesh_changed or skeleton_changed:
            self._apply_visibility()
        if interpolate_changed:
            self._apply_pose_at_time(self.time)
        imgui.end()

    def postRender(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx-file", default=str(default_fbx_file()))
    parser.add_argument("--bind-file", default=None)
    parser.add_argument("--clip-index", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--line-radius", type=float, default=0.008)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--nearest",
        action="store_true",
        help="Disable root lerp/quaternion slerp for the apply_pose path.",
    )
    parser.add_argument(
        "--material-mode",
        choices=("debug_checker", "texture"),
        default="debug_checker",
    )
    args = parser.parse_args()

    fbx_file = Path(args.fbx_file).expanduser().resolve()
    if not fbx_file.exists():
        raise FileNotFoundError(fbx_file)
    bind_file = (
        Path(args.bind_file).expanduser().resolve()
        if args.bind_file is not None
        else None
    )
    if bind_file is not None and not bind_file.exists():
        raise FileNotFoundError(bind_file)

    app = FbxCharacterApplyPoseViewer(
        fbx_file,
        bind_file,
        args.clip_index,
        args.fps,
        args.scale,
        args.line_radius,
        args.material_mode,
        not args.nearest,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    # app.set_render_hz(0.0)
    app.start()


if __name__ == "__main__":
    main()
