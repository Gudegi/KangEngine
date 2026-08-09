"""ImGui motion timeline"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .._core import _ke
from ..utils import (
    DEFAULT_CONTACT_SEMANTICS,
    DEFAULT_TRACKING_SEMANTICS,
    JointMapper,
    JointSemantic,
)

imgui = _ke.imgui


def _vec3_to_list(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _vec3_to_array(value) -> np.ndarray:
    return np.asarray(_vec3_to_list(value), dtype=np.float32)


@dataclass
class MotionSampleData:
    positions: np.ndarray
    matrices: np.ndarray
    velocities: np.ndarray
    node_names: list[str]
    fps: float
    mapper: JointMapper

    @classmethod
    def from_motion(cls, motion, profiles=None) -> "MotionSampleData":
        matrices = np.asarray(motion.global_matrices(), dtype=np.float32)
        positions = np.asarray(motion.global_positions(), dtype=np.float32)
        velocities = np.asarray(motion.global_linear_velocities(), dtype=np.float32)
        fps = max(float(motion.fps()), 1e-6)
        mapper = JointMapper.from_motion(motion, profiles=profiles or None)
        return cls(
            positions,
            matrices,
            velocities,
            mapper.joint_names,
            fps,
            mapper,
        )

    def joint_indices(self, names: list[str]) -> list[int]:
        return self.mapper.find_names(names)

    def semantic_indices(self, semantics: list[JointSemantic | str]) -> list[int]:
        return self.mapper.find_many(semantics)


@dataclass
class RootTrajectoryData:
    positions: np.ndarray
    velocities: np.ndarray
    directions: np.ndarray

    @classmethod
    def from_motion(
        cls,
        motion,
        root_index: int = 0,
        samples: Optional[MotionSampleData] = None,
    ) -> "RootTrajectoryData":
        samples = samples or MotionSampleData.from_motion(motion)
        if root_index >= samples.positions.shape[1]:
            empty = np.zeros((0, 3), dtype=np.float32)
            return cls(empty, empty, empty)

        positions = samples.positions[:, root_index, :]
        velocities = samples.velocities[:, root_index, :]
        directions = velocities.copy()
        directions[:, 1] = 0.0
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.divide(
            directions,
            np.maximum(norms, 1e-8),
            out=np.zeros_like(directions),
        )
        return cls(positions, velocities, directions)


@dataclass
class TrackingData:
    positions: np.ndarray
    velocities: np.ndarray
    joint_indices: list[int]
    joint_names: list[str]

    @classmethod
    def from_motion(
        cls,
        motion,
        joint_names: Optional[list[str]] = None,
        joint_indices: Optional[list[int]] = None,
        joint_semantics: Optional[list[JointSemantic | str]] = None,
        samples: Optional[MotionSampleData] = None,
    ) -> "TrackingData":
        samples = samples or MotionSampleData.from_motion(motion)
        indices = list(joint_indices or [])
        if joint_semantics is None and joint_names is None:
            joint_semantics = list(DEFAULT_TRACKING_SEMANTICS)
        indices.extend(samples.semantic_indices(joint_semantics or []))
        indices.extend(samples.joint_indices(joint_names or []))

        unique_indices = []
        for index in indices:
            index = int(index)
            if 0 <= index < samples.positions.shape[1] and index not in unique_indices:
                unique_indices.append(index)
        if not unique_indices and samples.positions.shape[1] > 0:
            unique_indices = [0]

        names = [
            samples.node_names[index]
            if index < len(samples.node_names)
            else f"joint_{index}"
            for index in unique_indices
        ]
        return cls(
            samples.positions[:, unique_indices, :],
            samples.velocities[:, unique_indices, :],
            unique_indices,
            names,
        )


@dataclass
class ContactData:
    positions: np.ndarray
    velocities: np.ndarray
    foot_indices: list[int]
    foot_names: list[str]
    up_axis: int = 1

    @classmethod
    def from_motion(
        cls,
        motion,
        foot_names: Optional[list[str]] = None,
        foot_indices: Optional[list[int]] = None,
        foot_semantics: Optional[list[JointSemantic | str]] = None,
        up_axis: int = 1,
        samples: Optional[MotionSampleData] = None,
    ) -> "ContactData":
        samples = samples or MotionSampleData.from_motion(motion)
        indices = list(foot_indices or [])
        if foot_semantics is None and foot_names is None:
            foot_semantics = list(DEFAULT_CONTACT_SEMANTICS)
        indices.extend(samples.semantic_indices(foot_semantics or []))
        indices.extend(samples.joint_indices(foot_names or []))

        unique_indices = []
        for index in indices:
            index = int(index)
            if 0 <= index < samples.positions.shape[1] and index not in unique_indices:
                unique_indices.append(index)

        names = [
            samples.node_names[index]
            if index < len(samples.node_names)
            else f"joint_{index}"
            for index in unique_indices
        ]
        return cls(
            samples.positions[:, unique_indices, :],
            samples.velocities[:, unique_indices, :],
            unique_indices,
            names,
            up_axis,
        )

    def contacts(
        self,
        height_threshold: float,
        velocity_threshold: float,
    ) -> np.ndarray:
        if self.positions.size == 0:
            return np.zeros((0, 0), dtype=bool)
        axis = int(np.clip(self.up_axis, 0, 2))
        heights = self.positions[:, :, axis]
        floor_heights = np.min(heights, axis=0, keepdims=True)
        height_ok = heights <= floor_heights + float(height_threshold)
        speed = np.linalg.norm(self.velocities, axis=2)
        velocity_ok = speed <= float(velocity_threshold)
        return height_ok & velocity_ok


class MotionCameraFollower:
    def __init__(
        self,
        motion,
        samples: Optional[MotionSampleData] = None,
        target_semantic: JointSemantic | str = JointSemantic.HIPS,
        target_index: Optional[int] = None,
        target_offset=(0.0, 0.85, 0.0),
        camera_offset=(0.0, 0.65, 3.2),
        smoothing: float = 0.12,
    ):
        self.samples = samples or MotionSampleData.from_motion(motion)
        if target_index is None:
            target_index = self.samples.mapper.find(target_semantic)
        if target_index is None:
            target_index = 0
        self.target_index = int(target_index)
        self.target_offset = np.asarray(target_offset, dtype=np.float32)
        self.camera_offset = np.asarray(camera_offset, dtype=np.float32)
        self.smoothing = max(0.0, float(smoothing))

    def target_at(self, frame_index: int) -> np.ndarray:
        if self.samples.positions.size == 0:
            return self.target_offset.copy()
        frame = int(np.clip(frame_index, 0, self.samples.positions.shape[0] - 1))
        joint = int(np.clip(self.target_index, 0, self.samples.positions.shape[1] - 1))
        return self.samples.positions[frame, joint, :] + self.target_offset

    def update(self, camera, frame_index: int, force: bool = False) -> None:
        target = self.target_at(frame_index)
        camera_pos = target + self.camera_offset

        if not force and self.smoothing > 0.0:
            alpha = min(self.smoothing, 1.0)
            old_target = _vec3_to_array(camera.get_target_pos())
            old_camera = _vec3_to_array(camera.get_camera_pos())
            target = old_target * (1.0 - alpha) + target * alpha
            camera_pos = old_camera * (1.0 - alpha) + camera_pos * alpha

        camera.set_target_pos(
            _ke.Vec3(float(target[0]), float(target[1]), float(target[2]))
        )
        camera.set_camera_pos(
            _ke.Vec3(
                float(camera_pos[0]),
                float(camera_pos[1]),
                float(camera_pos[2]),
            )
        )


@dataclass
class MotionPlayer:
    """Track playback time and select frames from a sampled motion."""

    duration: float
    fps: float
    num_frames: int
    time: float = 0.0
    playing: bool = True
    loop: bool = True
    time_scale: float = 1.0

    @classmethod
    def from_motion(cls, motion, playing: bool = True) -> "MotionPlayer":
        return cls(
            duration=max(float(motion.duration()), 0.0),
            fps=max(float(motion.fps()), 1e-6),
            num_frames=max(int(motion.num_frames()), 1),
            playing=playing,
        )

    @property
    def frame_index(self) -> int:
        if self.num_frames <= 1:
            return 0
        return max(0, min(self.num_frames - 1, int(self.time * self.fps)))

    @property
    def playback_duration(self) -> float:
        return self.num_frames / self.fps

    @property
    def normalized_time(self) -> float:
        if self.duration <= 1e-6:
            return 0.0
        return max(0.0, min(1.0, self.time / self.duration))

    def set_frame(self, frame: int) -> None:
        frame = max(0, min(self.num_frames - 1, int(frame)))
        self.time = frame / self.fps

    def reset(self) -> None:
        self.time = 0.0

    def update(self, dt: float) -> bool:
        if not self.playing:
            return False
        old_time = self.time
        self.time += max(0.0, float(dt)) * self.time_scale
        self._wrap_or_clamp()
        return abs(self.time - old_time) > 1e-9

    def _wrap_or_clamp(self) -> None:
        if self.playback_duration <= 1e-6:
            self.time = 0.0
            return
        if self.loop:
            self.time = self.time % self.playback_duration
        else:
            self.time = max(0.0, min(self.time, self.duration))


class MotionEditor:
    """Motion playback, sequencer UI, and pluggable motion modules."""

    def __init__(
        self,
        motion,
        motion_name: str = "Motion",
        panel_name: str = "MotionSequencer",
        fit_to_content: bool = True,
    ) -> None:
        self.motion = motion
        self.player = MotionPlayer.from_motion(motion)
        self.panel_name = panel_name
        self.motion_name = motion_name
        self.fit_to_content = fit_to_content
        self.first_frame = 0
        self.legend_width = 200.0
        self.is_expanded = True  # Show progress bar
        self.selected_track_idx = -1
        self.modules = []
        self._motion_samples: Optional[MotionSampleData] = None
        self.panel = _ke.MotionSequencerPanel()
        self.panel.set_motion(
            self.motion_name,
            self.player.num_frames,
            self.player.fps,
        )

    def set_playing(self, playing: bool):
        self.player.playing = playing

    def update(self, dt: float) -> bool:
        is_changed = self.player.update(dt)
        if is_changed:
            self.update_modules()
        return is_changed

    def reset(self) -> None:
        self.player.reset()
        self.update_modules()

    def motion_samples(self) -> MotionSampleData:
        if self._motion_samples is None:
            self._motion_samples = MotionSampleData.from_motion(self.motion)
        return self._motion_samples

    def global_positions(self) -> np.ndarray:
        return self.motion_samples().positions

    def add_module(self, module):
        module.initialize(self)
        self.modules.append(module)
        module.update(self)
        return module

    def update_modules(self, state=None) -> None:
        for module in self.modules:
            if module.enabled:
                module.update(self, state)

    def render_modules_ui(self) -> bool:
        is_changed = False
        if not self.modules:
            return False

        imgui.separator()
        imgui.text("Modules")
        for module in self.modules:
            changed, module.enabled = imgui.checkbox(module.name, module.enabled)
            is_changed = is_changed or changed
            imgui.same_line()
            visible_changed, module.visible = imgui.checkbox(
                f"visible##{module.name}",
                module.visible,
            )
            is_changed = is_changed or visible_changed
            if module.enabled:
                is_changed = module.ui(self) or is_changed
            if changed or visible_changed:
                module.update(self)
        return is_changed

    def render_panel_py(self, motion_name: Optional[str] = None) -> bool:
        if motion_name is not None:
            self.motion_name = str(motion_name)
        is_changed = False
        player = self.player
        state = "running" if player.playing else "paused"
        current_frame = player.frame_index

        imgui.text(
            f"Frame {current_frame + 1}/{player.num_frames}  {player.time:.3f}s / {player.duration:.3f}s  |  {state}"
        )

        if imgui.button("Pause" if player.playing else "Play"):
            player.playing = not player.playing
        imgui.same_line()
        if imgui.button("Reset"):
            player.reset()
            is_changed = True
        imgui.same_line()
        _, player.loop = imgui.checkbox("loop", player.loop)

        _, player.time_scale = imgui.slider_float(
            "playback speed",
            player.time_scale,
            0.0,
            4.0,
        )

        (
            seq_is_changed,
            current_frame,
            self.first_frame,
            self.is_expanded,
            self.selected_track_idx,
            self.legend_width,
        ) = imgui.motion_sequencer_resizable(
            self.panel_name,
            current_frame,
            0,
            max(player.num_frames - 1, 0),
            self.first_frame,
            self.is_expanded,
            self.selected_track_idx,
            self.motion_name,
            self.fit_to_content,
            self.legend_width,
        )
        if seq_is_changed:
            player.set_frame(current_frame)
            is_changed = True
            self.update_modules()

        imgui.progress_bar(
            player.normalized_time,
            -1.0,
            0.0,
            f"{player.normalized_time * 100.0:.1f}%",
        )
        module_changed = self.render_modules_ui()
        if module_changed:
            self.update_modules()
        is_changed = module_changed or is_changed
        return is_changed

    def render_panel(self, motion_name: Optional[str] = None) -> bool:
        if motion_name is not None and self.motion_name != str(motion_name):
            self.motion_name = str(motion_name)
            self.panel.set_motion(
                self.motion_name,
                self.player.num_frames,
                self.player.fps,
            )

        player = self.player
        previous_time = player.time
        self.panel.set_current_time(player.time)
        self.panel.set_playing(player.playing)
        self.panel.set_loop(player.loop)
        self.panel.set_time_scale(player.time_scale)
        self.panel.set_legend_width(self.legend_width)
        self.panel.build_panel()

        player.time = self.panel.current_time()
        player.playing = self.panel.is_playing()
        player.loop = self.panel.loop()
        player.time_scale = self.panel.time_scale()
        self.legend_width = self.panel.legend_width()
        is_changed = abs(player.time - previous_time) > 1e-9
        if is_changed:
            self.update_modules()
        return is_changed

    def render(self) -> bool:
        return self.render_panel()
