"""Small ImGui motion timeline helpers built on KangEngine's ImSequencer binding."""

from __future__ import annotations

from dataclasses import dataclass

from ._core import _ke
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
        return max(0, min(self.num_frames - 1, int(round(self.time * self.fps))))

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
        if self.duration <= 1e-6:
            self.time = 0.0
            return
        if self.loop:
            self.time = self.time % self.duration
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
        self.player = MotionPlayer.from_motion(motion)
        self.panel_name = panel_name
        self.motion_name = motion_name
        self.fit_to_content = fit_to_content
        self.first_frame = 0
        self.is_expanded = True  # Show progress bar
        self.selected_track_idx = -1

    def update(self, dt: float) -> bool:
        return self.player.update(dt)

    def reset(self) -> None:
        self.player.reset()

    def render_ui(self) -> bool:
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

        imgui.progress_bar(
            player.normalized_time,
            -1.0,
            0.0,
            f"{player.normalized_time * 100.0:.1f}%",
        )
        return is_changed

    def render(self) -> bool:
        return self.render_ui()
