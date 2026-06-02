""" ImGui motion timeline"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ._core import _ke
from .utils import preset_rgba
from .utils.math import normalize_vector
imgui = _ke.imgui


@dataclass
class MotionPlayer:
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
        self.is_expanded = True  # Show progress bar
        self.selected_track_idx = -1
        self.modules = []
        self.panel = _ke.MotionSequencerPanel()
        self.panel.set_motion(self.motion_name, self.player.num_frames, self.player.fps)

    def update(self, dt: float) -> bool:
        is_changed = self.player.update(dt)
        if is_changed:
            self.update_modules()
        return is_changed

    def reset(self) -> None:
        self.player.reset()
        self.update_modules()

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
            f"Frame {current_frame + 1}/{player.num_frames}  "
            f"{player.time:.3f}s / {player.duration:.3f}s  |  {state}"
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

        seq_is_changed, current_frame, self.first_frame, self.is_expanded, self.selected_track_idx = (
            imgui.motion_sequencer(
                self.panel_name,
                current_frame,
                0,
                max(player.num_frames - 1, 0),
                self.first_frame,
                self.is_expanded,
                self.selected_track_idx,
                self.motion_name,
                self.fit_to_content,
            )
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
        self.panel.build_panel()

        player.time = self.panel.current_time()
        player.playing = self.panel.is_playing()
        player.loop = self.panel.loop()
        player.time_scale = self.panel.time_scale()
        is_changed = abs(player.time - previous_time) > 1e-9
        if is_changed:
            self.update_modules()
        return is_changed

    def render(self) -> bool:
        return self.render_panel()


class MotionModule:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.visible = True

    def initialize(self, editor: MotionEditor) -> None:
        pass

    def update(self, editor: MotionEditor, state=None) -> None:
        pass

    def ui(self, editor: MotionEditor) -> bool:
        return False


class RootTrajectoryModule(MotionModule):
    def __init__(
        self,
        app,
        path: str = "/debug/root_trajectory",
        root_index: int = 0,
        line_width: float = 2.0,
        point_size: float = 10.0,
    ):
        super().__init__("Root Trajectory")
        self.app = app
        self.path = path
        self.root_index = root_index
        self.line_width = line_width
        self.point_size = point_size
        self.window_frames = 120
        self.stride = 4
        self.direction_length = 0.22
        self.velocity_scale = 0.08
        self.draw_path = True
        self.draw_points = True
        self.draw_directions = True
        self.draw_velocities = True
        self._positions = []
        self._velocities = []
        self._directions = []
        self._last_key = None

    def _clear_debug_draw(self) -> None:
        self.app.clear_debug_lines(f"{self.path}/path")
        self.app.clear_debug_points(f"{self.path}/points")
        self.app.clear_debug_points(f"{self.path}/current")
        self.app.clear_debug_lines(f"{self.path}/directions")
        self.app.clear_debug_lines(f"{self.path}/velocities")

    def initialize(self, editor: MotionEditor) -> None:
        self._positions = []
        self._velocities = []
        self._directions = []
        motion = editor.motion
        frame_count = max(int(motion.num_frames()), 1)
        fps = max(float(motion.fps()), 1e-6)
        for frame in range(frame_count):
            state = motion.sample(frame / fps, loop=False)
            positions = state.compute_global_positions()
            if self.root_index >= len(positions):
                break
            pos = positions[self.root_index]
            self._positions.append([float(pos.x), float(pos.y), float(pos.z)])
        for i, pos in enumerate(self._positions):
            prev_pos = self._positions[max(i - 1, 0)]
            next_pos = self._positions[min(i + 1, len(self._positions) - 1)]
            velocity = [
                (next_pos[0] - prev_pos[0]) * fps * 0.5,
                (next_pos[1] - prev_pos[1]) * fps * 0.5,
                (next_pos[2] - prev_pos[2]) * fps * 0.5,
            ]
            direction = [velocity[0], 0.0, velocity[2]]
            direction = normalize_vector(direction).tolist()
            self._velocities.append(velocity)
            self._directions.append(direction)
        self._last_key = None

    def update(self, editor: MotionEditor, state=None) -> None:
        if not self._positions:
            return
        if not self.enabled or not self.visible:
            self._clear_debug_draw()
            self._last_key = None
            return

        current = editor.player.frame_index
        stride = max(1, int(self.stride))
        window = max(1, int(self.window_frames))
        key = (
            current,
            window,
            stride,
            self.enabled,
            self.visible,
            self.draw_path,
            self.draw_points,
            self.draw_directions,
            self.draw_velocities,
            float(self.direction_length),
            float(self.velocity_scale),
        )
        if key == self._last_key:
            return
        self._last_key = key

        lo = max(0, current - window)
        hi = min(len(self._positions) - 1, current + window)
        indices = list(range(lo, hi + 1, stride))
        if indices[-1] != hi:
            indices.append(hi)

        alpha = 1.0
        black = preset_rgba(_ke.ColorType.BLACK)
        orange = preset_rgba(_ke.ColorType.ORANGE)
        lime_green = preset_rgba(_ke.ColorType.LIME_GREEN)
        royal_blue = preset_rgba(_ke.ColorType.ROYAL_BLUE)
        path_starts = []
        path_ends = []
        path_colors = []
        for a, b in zip(indices[:-1], indices[1:]):
            path_starts.append(self._positions[a])
            path_ends.append(self._positions[b])
            path_colors.append(black)

        points = [self._positions[i] for i in indices]
        point_colors = []
        direction_starts = []
        direction_ends = []
        direction_colors = []
        velocity_starts = []
        velocity_ends = []
        velocity_colors = []
        current_pos = self._positions[current]
        for i in indices:
            ratio = 0.0 if hi == lo else (i - lo) / float(hi - lo)
            if i < current:
                color = [*royal_blue[:3], alpha * (0.35 + 0.45 * ratio)]
            elif i > current:
                color = [*lime_green[:3], alpha * (0.85 - 0.45 * ratio)]
            else:
                color = orange
            point_colors.append(color)

            pos = self._positions[i]
            direction = self._directions[i]
            velocity = self._velocities[i]
            direction_starts.append(pos)
            direction_ends.append(
                [
                    pos[0] + direction[0] * self.direction_length,
                    pos[1] + 0.02,
                    pos[2] + direction[2] * self.direction_length,
                ]
            )
            direction_colors.append([*orange[:3], alpha * 0.9])
            velocity_starts.append([pos[0], pos[1] + 0.04, pos[2]])
            velocity_ends.append(
                [
                    pos[0] + velocity[0] * self.velocity_scale,
                    pos[1] + velocity[1] * self.velocity_scale + 0.04,
                    pos[2] + velocity[2] * self.velocity_scale,
                ]
            )
            velocity_colors.append([*lime_green[:3], alpha * 0.55])

        if self.draw_path and path_starts:
            self.app.log_debug_lines(
                f"{self.path}/path",
                np.asarray(path_starts, dtype=np.float32),
                np.asarray(path_ends, dtype=np.float32),
                np.asarray(path_colors, dtype=np.float32),
                self.line_width,
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/path")

        if self.draw_points and points:
            self.app.log_debug_points(
                f"{self.path}/points",
                np.asarray(points, dtype=np.float32),
                np.asarray(point_colors, dtype=np.float32),
                self.point_size,
            )
            self.app.log_debug_points(
                f"{self.path}/current",
                np.asarray([current_pos], dtype=np.float32),
                np.asarray([black], dtype=np.float32),
                self.point_size * 2.0,
            )
        else:
            self.app.clear_debug_points(f"{self.path}/points")
            self.app.clear_debug_points(f"{self.path}/current")

        if self.draw_directions and direction_starts:
            self.app.log_debug_lines(
                f"{self.path}/directions",
                np.asarray(direction_starts, dtype=np.float32),
                np.asarray(direction_ends, dtype=np.float32),
                np.asarray(direction_colors, dtype=np.float32),
                max(self.line_width * 0.75, 1.0),
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/directions")

        if self.draw_velocities and velocity_starts:
            self.app.log_debug_lines(
                f"{self.path}/velocities",
                np.asarray(velocity_starts, dtype=np.float32),
                np.asarray(velocity_ends, dtype=np.float32),
                np.asarray(velocity_colors, dtype=np.float32),
                max(self.line_width * 0.5, 1.0),
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/velocities")

    def ui(self, editor: MotionEditor) -> bool:
        is_changed = False
        changed, value = imgui.slider_float(
            f"window frames##{self.name}",
            float(self.window_frames),
            1.0,
            max(float(editor.player.num_frames), 1.0),
        )
        if changed:
            self.window_frames = int(value)
            self._last_key = None
            is_changed = True

        changed, value = imgui.slider_float(
            f"stride##{self.name}",
            float(self.stride),
            1.0,
            20.0,
        )
        if changed:
            self.stride = int(value)
            self._last_key = None
            is_changed = True

        changed, self.draw_path = imgui.checkbox(
            f"path##{self.name}",
            self.draw_path,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_points = imgui.checkbox(
            f"points##{self.name}",
            self.draw_points,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_directions = imgui.checkbox(
            f"directions##{self.name}",
            self.draw_directions,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_velocities = imgui.checkbox(
            f"velocities##{self.name}",
            self.draw_velocities,
        )
        is_changed = changed or is_changed

        changed, value = imgui.slider_float(
            f"point size##{self.name}",
            float(self.point_size),
            1.0,
            24.0,
        )
        if changed:
            self.point_size = value
            self._last_key = None
            is_changed = True

        changed, value = imgui.slider_float(
            f"direction length##{self.name}",
            float(self.direction_length),
            0.0,
            1.0,
        )
        if changed:
            self.direction_length = value
            self._last_key = None
            is_changed = True

        return is_changed
