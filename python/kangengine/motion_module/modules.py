from __future__ import annotations

from typing import Optional

import numpy as np

from .._core import _ke
from .editor import (
    ContactData,
    MotionEditor,
    MotionSampleData,
    RootTrajectoryData,
    TrackingData,
)
from ..utils import (
    JointSemantic, 
    log_debug_axes, 
    preset_rgba, 
    DEFAULT_CONTACT_SEMANTICS, 
    DEFAULT_TRACKING_SEMANTICS
)

imgui = _ke.imgui


def _rotation_extrinsic_xyz_degrees(rotation_deg) -> np.ndarray:
    """Extrinsic XYZ Euler angles in degrees, composed as Rz * Ry * Rx."""
    x, y, z = np.radians(np.asarray(rotation_deg, dtype=np.float32))
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rz = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return rz @ ry @ rx


def _transform_from_translation_rotation(translation, rotation_deg) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _rotation_extrinsic_xyz_degrees(rotation_deg)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


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
        data = RootTrajectoryData.from_motion(
            editor.motion,
            self.root_index,
            editor.motion_samples(),
        )
        self._positions = data.positions.tolist()
        self._velocities = data.velocities.tolist()
        self._directions = data.directions.tolist()
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

class TrackingModule(MotionModule):
    def __init__(
        self,
        app,
        path: str = "/debug/motion_tracking",
        joint_names: Optional[list[str]] = None,
        joint_indices: Optional[list[int]] = None,
        joint_semantics: Optional[list[JointSemantic | str]] = None,
        line_width: float = 2.0,
        point_size: float = 9.0,
    ):
        super().__init__("Tracking")
        self.app = app
        self.path = path
        self.joint_names = list(joint_names or [])
        self.joint_indices = list(joint_indices or [])
        self.joint_semantics = list(
            joint_semantics
            if joint_semantics is not None
            else DEFAULT_TRACKING_SEMANTICS
        )
        self.line_width = line_width
        self.point_size = point_size
        self.window_frames = 90
        self.stride = 3
        self.velocity_scale = 0.08
        self.draw_trails = True
        self.draw_points = True
        self.draw_velocities = True
        self.use_per_joint_colors = False
        self._positions = np.zeros((0, 0, 3), dtype=np.float32)
        self._velocities = np.zeros((0, 0, 3), dtype=np.float32)
        self._tracked_indices: list[int] = []
        self._tracked_names: list[str] = []
        self._last_key = None

    def _clear_debug_draw(self) -> None:
        self.app.clear_debug_lines(f"{self.path}/trails")
        self.app.clear_debug_lines(f"{self.path}/velocities")
        self.app.clear_debug_points(f"{self.path}/points")
        self.app.clear_debug_points(f"{self.path}/current")

    def initialize(self, editor: MotionEditor) -> None:
        data = TrackingData.from_motion(
            editor.motion,
            self.joint_names,
            self.joint_indices,
            self.joint_semantics,
            editor.motion_samples(),
        )
        self._positions = data.positions
        self._velocities = data.velocities
        self._tracked_indices = data.joint_indices
        self._tracked_names = data.joint_names
        self._last_key = None

    def _joint_color(self, joint_slot: int, alpha: float = 1.0) -> list[float]:
        if not self.use_per_joint_colors:
            return preset_rgba(_ke.ColorType.PASTEL_SKY, alpha)

        palette = [
            _ke.ColorType.PASTEL_SKY,
            _ke.ColorType.PASTEL_CORAL,
            _ke.ColorType.PASTEL_GREEN,
            _ke.ColorType.ORANGE,
            _ke.ColorType.ROYAL_BLUE,
            _ke.ColorType.LIME_GREEN,
        ]
        color = preset_rgba(palette[joint_slot % len(palette)], alpha)
        return color

    def update(self, editor: MotionEditor, state=None) -> None:
        if self._positions.size == 0:
            return
        if not self.enabled or not self.visible:
            self._clear_debug_draw()
            self._last_key = None
            return

        current = min(editor.player.frame_index, self._positions.shape[0] - 1)
        stride = max(1, int(self.stride))
        window = max(1, int(self.window_frames))
        key = (
            current,
            window,
            stride,
            self.enabled,
            self.visible,
            self.draw_trails,
            self.draw_points,
            self.draw_velocities,
            self.use_per_joint_colors,
            float(self.velocity_scale),
            tuple(self._tracked_indices),
        )
        if key == self._last_key:
            return
        self._last_key = key

        lo = max(0, current - window)
        hi = min(self._positions.shape[0] - 1, current + window)
        indices = list(range(lo, hi + 1, stride))
        if indices[-1] != hi:
            indices.append(hi)

        trail_starts = []
        trail_ends = []
        trail_colors = []
        points = []
        point_colors = []
        current_points = []
        current_colors = []
        velocity_starts = []
        velocity_ends = []
        velocity_colors = []

        for joint_slot in range(self._positions.shape[1]):
            base_color = self._joint_color(joint_slot)
            for a, b in zip(indices[:-1], indices[1:]):
                ratio = 0.0 if hi == lo else (a - lo) / float(hi - lo)
                alpha = 0.25 + 0.65 * ratio
                trail_starts.append(self._positions[a, joint_slot].tolist())
                trail_ends.append(self._positions[b, joint_slot].tolist())
                trail_colors.append([*base_color[:3], alpha])

            for index in indices:
                points.append(self._positions[index, joint_slot].tolist())
                point_colors.append([*base_color[:3], 0.55])

            current_pos = self._positions[current, joint_slot]
            current_points.append(current_pos.tolist())
            current_colors.append([*base_color[:3], 1.0])

            velocity = self._velocities[current, joint_slot]
            vertical_offset = np.array([0.0, 0.035, 0.0], dtype=np.float32)
            velocity_starts.append((current_pos + vertical_offset).tolist())
            velocity_ends.append(
                (
                    current_pos
                    + vertical_offset
                    + velocity * self.velocity_scale
                ).tolist()
            )
            velocity_colors.append([*base_color[:3], 0.8])

        if self.draw_trails and trail_starts:
            self.app.log_debug_lines(
                f"{self.path}/trails",
                np.asarray(trail_starts, dtype=np.float32),
                np.asarray(trail_ends, dtype=np.float32),
                np.asarray(trail_colors, dtype=np.float32),
                self.line_width,
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/trails")

        if self.draw_points and points:
            self.app.log_debug_points(
                f"{self.path}/points",
                np.asarray(points, dtype=np.float32),
                np.asarray(point_colors, dtype=np.float32),
                self.point_size,
            )
            self.app.log_debug_points(
                f"{self.path}/current",
                np.asarray(current_points, dtype=np.float32),
                np.asarray(current_colors, dtype=np.float32),
                self.point_size * 1.6,
            )
        else:
            self.app.clear_debug_points(f"{self.path}/points")
            self.app.clear_debug_points(f"{self.path}/current")

        if self.draw_velocities and velocity_starts:
            self.app.log_debug_lines(
                f"{self.path}/velocities",
                np.asarray(velocity_starts, dtype=np.float32),
                np.asarray(velocity_ends, dtype=np.float32),
                np.asarray(velocity_colors, dtype=np.float32),
                max(self.line_width * 0.7, 1.0),
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/velocities")

    def ui(self, editor: MotionEditor) -> bool:
        is_changed = False
        tracked = ", ".join(self._tracked_names) if self._tracked_names else "none"
        imgui.text(f"tracked joints: {tracked}")

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

        changed, self.draw_trails = imgui.checkbox(
            f"trails##{self.name}",
            self.draw_trails,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_points = imgui.checkbox(
            f"points##{self.name}",
            self.draw_points,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_velocities = imgui.checkbox(
            f"velocities##{self.name}",
            self.draw_velocities,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.use_per_joint_colors = imgui.checkbox(
            f"per joint colors##{self.name}",
            self.use_per_joint_colors,
        )
        if changed:
            self._last_key = None
            is_changed = True

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
            f"velocity scale##{self.name}",
            float(self.velocity_scale),
            0.0,
            0.3,
        )
        if changed:
            self.velocity_scale = value
            self._last_key = None
            is_changed = True

        return is_changed

class ContactModule(MotionModule):
    def __init__(
        self,
        app,
        path: str = "/debug/motion_contacts",
        foot_names: Optional[list[str]] = None,
        foot_indices: Optional[list[int]] = None,
        foot_semantics: Optional[list[JointSemantic | str]] = None,
        up_axis: int = 1,
        point_size: float = 20.0,
    ):
        super().__init__("Contact")
        self.app = app
        self.path = path
        self.foot_names = list(foot_names or [])
        self.foot_indices = list(foot_indices or [])
        self.foot_semantics = list(
            foot_semantics
            if foot_semantics is not None
            else DEFAULT_CONTACT_SEMANTICS
        )
        self.up_axis = int(up_axis)
        self.point_size = point_size
        self.window_frames = 80
        self.stride = 2
        self.height_threshold = 0.045
        self.velocity_threshold = 0.35
        self.non_contact_alpha = 1.0
        self.draw_window_points = True
        self.draw_current = True
        self.draw_velocity = False
        self._data: Optional[ContactData] = None
        self._positions = np.zeros((0, 0, 3), dtype=np.float32)
        self._velocities = np.zeros((0, 0, 3), dtype=np.float32)
        self._contacts = np.zeros((0, 0), dtype=bool)
        self._foot_indices: list[int] = []
        self._foot_names: list[str] = []
        self._last_key = None

    def _clear_debug_draw(self) -> None:
        self.app.clear_debug_points(f"{self.path}/window")
        self.app.clear_debug_points(f"{self.path}/current")
        self.app.clear_debug_lines(f"{self.path}/velocity")

    def initialize(self, editor: MotionEditor) -> None:
        self._data = ContactData.from_motion(
            editor.motion,
            self.foot_names,
            self.foot_indices,
            self.foot_semantics,
            self.up_axis,
            editor.motion_samples(),
        )
        self._positions = self._data.positions
        self._velocities = self._data.velocities
        self._foot_indices = self._data.foot_indices
        self._foot_names = self._data.foot_names
        self._recompute_contacts()
        self._last_key = None

    def _recompute_contacts(self) -> None:
        if self._data is None or self._positions.size == 0:
            self._contacts = np.zeros((0, 0), dtype=bool)
            return
        self._contacts = self._data.contacts(
            self.height_threshold,
            self.velocity_threshold,
        )

    def _contact_color(self, is_contact: bool, alpha: float = 1.0) -> list[float]:
        if is_contact:
            return preset_rgba(_ke.ColorType.LIME_GREEN, alpha)
        return [0.02, 0.02, 0.02, min(alpha, self.non_contact_alpha)]

    def update(self, editor: MotionEditor, state=None) -> None:
        if self._positions.size == 0:
            return
        if not self.enabled or not self.visible:
            self._clear_debug_draw()
            self._last_key = None
            return

        current = min(editor.player.frame_index, self._positions.shape[0] - 1)
        stride = max(1, int(self.stride))
        window = max(1, int(self.window_frames))
        key = (
            current,
            window,
            stride,
            self.enabled,
            self.visible,
            self.draw_window_points,
            self.draw_current,
            self.draw_velocity,
            float(self.height_threshold),
            float(self.velocity_threshold),
            float(self.non_contact_alpha),
            float(self.point_size),
            tuple(self._foot_indices),
        )
        if key == self._last_key:
            return
        self._last_key = key
        self._recompute_contacts()

        lo = max(0, current - window)
        hi = min(self._positions.shape[0] - 1, current + window)
        indices = list(range(lo, hi + 1, stride))
        if indices[-1] != hi:
            indices.append(hi)

        window_points = []
        window_colors = []
        current_points = []
        current_colors = []
        velocity_starts = []
        velocity_ends = []
        velocity_colors = []

        for foot_slot in range(self._positions.shape[1]):
            for index in indices:
                contact = bool(self._contacts[index, foot_slot])
                ratio = 0.0 if hi == lo else (index - lo) / float(hi - lo)
                alpha = 0.25 + 0.55 * ratio
                window_points.append(self._positions[index, foot_slot].tolist())
                window_colors.append(self._contact_color(contact, alpha))

            current_contact = bool(self._contacts[current, foot_slot])
            current_pos = self._positions[current, foot_slot]
            current_points.append(current_pos.tolist())
            current_colors.append(self._contact_color(current_contact, 1.0))

            vertical_offset = np.array([0.0, 0.035, 0.0], dtype=np.float32)
            velocity = self._velocities[current, foot_slot] * 0.08
            velocity_starts.append((current_pos + vertical_offset).tolist())
            velocity_ends.append(
                (current_pos + vertical_offset + velocity).tolist()
            )
            velocity_colors.append(self._contact_color(current_contact, 0.75))

        if self.draw_window_points and window_points:
            self.app.log_debug_points(
                f"{self.path}/window",
                np.asarray(window_points, dtype=np.float32),
                np.asarray(window_colors, dtype=np.float32),
                max(self.point_size * 0.65, 1.0),
            )
        else:
            self.app.clear_debug_points(f"{self.path}/window")

        if self.draw_current and current_points:
            self.app.log_debug_points(
                f"{self.path}/current",
                np.asarray(current_points, dtype=np.float32),
                np.asarray(current_colors, dtype=np.float32),
                self.point_size,
            )
        else:
            self.app.clear_debug_points(f"{self.path}/current")

        if self.draw_velocity and velocity_starts:
            self.app.log_debug_lines(
                f"{self.path}/velocity",
                np.asarray(velocity_starts, dtype=np.float32),
                np.asarray(velocity_ends, dtype=np.float32),
                np.asarray(velocity_colors, dtype=np.float32),
                1.5,
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/velocity")

    def ui(self, editor: MotionEditor) -> bool:
        is_changed = False
        tracked = ", ".join(self._foot_names) if self._foot_names else "none"
        imgui.text(f"contact joints: {tracked}")

        changed, value = imgui.slider_float(
            f"height threshold##{self.name}",
            float(self.height_threshold),
            0.0,
            0.2,
        )
        if changed:
            self.height_threshold = value
            self._last_key = None
            is_changed = True

        changed, value = imgui.slider_float(
            f"velocity threshold##{self.name}",
            float(self.velocity_threshold),
            0.0,
            2.0,
        )
        if changed:
            self.velocity_threshold = value
            self._last_key = None
            is_changed = True

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

        changed, self.draw_window_points = imgui.checkbox(
            f"window points##{self.name}",
            self.draw_window_points,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_current = imgui.checkbox(
            f"current##{self.name}",
            self.draw_current,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_velocity = imgui.checkbox(
            f"velocity##{self.name}",
            self.draw_velocity,
        )
        is_changed = changed or is_changed

        changed, value = imgui.slider_float(
            f"non-contact alpha##{self.name}",
            float(self.non_contact_alpha),
            0.0,
            1.0,
        )
        if changed:
            self.non_contact_alpha = value
            self._last_key = None
            is_changed = True

        changed, value = imgui.slider_float(
            f"point size##{self.name}",
            float(self.point_size),
            1.0,
            30.0,
        )
        if changed:
            self.point_size = value
            self._last_key = None
            is_changed = True

        return is_changed

class TargetModule(MotionModule):
    OFFSET_FRAMES = ("world", "root", "source")

    def __init__(
        self,
        app,
        path: str = "/debug/motion_target",
        source_semantic: JointSemantic | str = JointSemantic.RIGHT_HAND,
        source_index: Optional[int] = None,
        target_offset=(0.25, 0.15, 0.0),
        target_euler_extrinsic_xyz_deg=(0.0, 0.0, 0.0),
        target_transform=None,
        offset_frame: str = "root",
        point_size: float = 18.0,
        line_width: float = 2.0,
    ):
        super().__init__("Target")
        self.app = app
        self.path = path
        self.source_semantic = source_semantic
        self.source_index = source_index
        self.target_euler_extrinsic_xyz_deg = np.asarray(
            target_euler_extrinsic_xyz_deg,
            dtype=np.float32,
        )
        self.target_transform = (
            np.asarray(target_transform, dtype=np.float32).reshape(4, 4)
            if target_transform is not None
            else _transform_from_translation_rotation(
                target_offset,
                self.target_euler_extrinsic_xyz_deg,
            )
        )
        self.offset_frame = (
            offset_frame
            if offset_frame in self.OFFSET_FRAMES
            else "root"
        )
        self.point_size = float(point_size)
        self.line_width = float(line_width)
        self.draw_source = True
        self.draw_target = True
        self.draw_error = True
        self.draw_axes = True
        self.draw_source_axes = True
        self.axis_length = 0.18
        self._samples: Optional[MotionSampleData] = None
        self._resolved_index = 0
        self._last_key = None

    def _clear_debug_draw(self) -> None:
        self.app.clear_debug_points(f"{self.path}/source")
        self.app.clear_debug_points(f"{self.path}/target")
        self.app.clear_debug_lines(f"{self.path}/error")
        self.app.clear_debug_lines(f"{self.path}/axes")
        self.app.clear_debug_lines(f"{self.path}/source_axes")

    def initialize(self, editor: MotionEditor) -> None:
        self._samples = editor.motion_samples()
        if self.source_index is not None:
            index = int(self.source_index)
        else:
            found = self._samples.mapper.find(self.source_semantic)
            index = 0 if found is None else int(found)
        joint_count = self._samples.positions.shape[1]
        self._resolved_index = int(np.clip(index, 0, max(joint_count - 1, 0)))
        self._last_key = None

    def source_position(self, frame_index: int) -> np.ndarray:
        if self._samples is None or self._samples.positions.size == 0:
            return np.zeros(3, dtype=np.float32)
        frame = int(np.clip(frame_index, 0, self._samples.positions.shape[0] - 1))
        return self._samples.positions[frame, self._resolved_index, :]

    def target_position(self, frame_index: int) -> np.ndarray:
        return self.target_world_transform(frame_index)[:3, 3]

    def target_world_transform(self, frame_index: int) -> np.ndarray:
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = self.offset_frame_rotation(frame_index)
        transform[:3, 3] = self.source_position(frame_index)
        return transform @ self.target_transform

    def offset_frame_rotation(self, frame_index: int) -> np.ndarray:
        if self._samples is None or self._samples.matrices.size == 0:
            return np.eye(3, dtype=np.float32)
        if self.offset_frame == "world":
            return np.eye(3, dtype=np.float32)

        frame = int(np.clip(frame_index, 0, self._samples.matrices.shape[0] - 1))
        joint = 0 if self.offset_frame == "root" else self._resolved_index
        joint = int(np.clip(joint, 0, self._samples.matrices.shape[1] - 1))
        return self._samples.matrices[frame, joint, :3, :3]

    def target_rotation(self, frame_index: int) -> np.ndarray:
        return self.target_world_transform(frame_index)[:3, :3]

    def update(self, editor: MotionEditor, state=None) -> None:
        if self._samples is None or self._samples.positions.size == 0:
            return
        if not self.enabled or not self.visible:
            self._clear_debug_draw()
            self._last_key = None
            return

        frame = editor.player.frame_index
        source = self.source_position(frame)
        target = self.target_position(frame)
        key = (
            frame,
            self.enabled,
            self.visible,
            self.draw_source,
            self.draw_target,
            self.draw_error,
            self.draw_axes,
            self.draw_source_axes,
            float(self.point_size),
            float(self.line_width),
            float(self.axis_length),
            tuple(float(v) for v in self.target_transform.reshape(-1)),
            self.offset_frame,
            self._resolved_index,
        )
        if key == self._last_key:
            return
        self._last_key = key

        if self.draw_source:
            self.app.log_debug_points(
                f"{self.path}/source",
                np.asarray([source], dtype=np.float32),
                np.asarray(
                    [preset_rgba(_ke.ColorType.PASTEL_SKY, 1.0)],
                    dtype=np.float32,
                ),
                max(self.point_size * 0.65, 1.0),
            )
        else:
            self.app.clear_debug_points(f"{self.path}/source")

        if self.draw_target:
            self.app.log_debug_points(
                f"{self.path}/target",
                np.asarray([target], dtype=np.float32),
                np.asarray(
                    [preset_rgba(_ke.ColorType.PASTEL_CORAL, 1.0)],
                    dtype=np.float32,
                ),
                self.point_size,
            )
        else:
            self.app.clear_debug_points(f"{self.path}/target")

        if self.draw_error:
            self.app.log_debug_lines(
                f"{self.path}/error",
                np.asarray([source], dtype=np.float32),
                np.asarray([target], dtype=np.float32),
                np.asarray(
                    [preset_rgba(_ke.ColorType.ORANGE, 0.9)],
                    dtype=np.float32,
                ),
                self.line_width,
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/error")

        if self.draw_axes:
            log_debug_axes(
                self.app,
                f"{self.path}/axes",
                target,
                self.target_rotation(frame),
                self.axis_length,
                max(self.line_width * 0.75, 1.0),
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/axes")

        if self.draw_source_axes:
            log_debug_axes(
                self.app,
                f"{self.path}/source_axes",
                source,
                self.source_rotation(frame),
                self.axis_length,
                max(self.line_width * 0.75, 1.0),
            )
        else:
            self.app.clear_debug_lines(f"{self.path}/source_axes")

    def source_rotation(self, frame_index: int) -> np.ndarray:
        if self._samples is None or self._samples.matrices.size == 0:
            return np.eye(3, dtype=np.float32)

        frame = int(np.clip(frame_index, 0, self._samples.matrices.shape[0] - 1))
        joint = int(
            np.clip(self._resolved_index, 0, self._samples.matrices.shape[1] - 1)
        )
        return self._samples.matrices[frame, joint, :3, :3]

    def ui(self, editor: MotionEditor) -> bool:
        is_changed = False
        if self._samples is not None and self._samples.node_names:
            name = self._samples.node_names[self._resolved_index]
        else:
            name = f"joint_{self._resolved_index}"
        imgui.text(f"source joint: {name}")
        imgui.text(f"offset frame: {self.offset_frame}")
        imgui.same_line()
        if imgui.button(f"next frame##{self.name}"):
            index = self.OFFSET_FRAMES.index(self.offset_frame)
            self.offset_frame = self.OFFSET_FRAMES[
                (index + 1) % len(self.OFFSET_FRAMES)
            ]
            self._last_key = None
            is_changed = True

        for axis, label in enumerate(
            ("target pos X", "target pos Y", "target pos Z")
        ):
            changed, value = imgui.slider_float(
                f"{label}##{self.name}",
                float(self.target_transform[axis, 3]),
                -2.0,
                2.0,
            )
            if changed:
                self.target_transform[axis, 3] = value
                self._last_key = None
                is_changed = True

        for axis, label in enumerate(
            (
                "target euler extrinsic XYZ X",
                "target euler extrinsic XYZ Y",
                "target euler extrinsic XYZ Z",
            )
        ):
            changed, value = imgui.slider_float(
                f"{label}##{self.name}",
                float(self.target_euler_extrinsic_xyz_deg[axis]),
                -180.0,
                180.0,
            )
            if changed:
                self.target_euler_extrinsic_xyz_deg[axis] = value
                self.target_transform[:3, :3] = (
                    _rotation_extrinsic_xyz_degrees(
                        self.target_euler_extrinsic_xyz_deg
                    )
                )
                self._last_key = None
                is_changed = True

        changed, self.draw_source = imgui.checkbox(
            f"source##{self.name}",
            self.draw_source,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_target = imgui.checkbox(
            f"target##{self.name}",
            self.draw_target,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_error = imgui.checkbox(
            f"error##{self.name}",
            self.draw_error,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_axes = imgui.checkbox(
            f"target axes##{self.name}",
            self.draw_axes,
        )
        is_changed = changed or is_changed
        imgui.same_line()
        changed, self.draw_source_axes = imgui.checkbox(
            f"source axes##{self.name}",
            self.draw_source_axes,
        )
        is_changed = changed or is_changed

        changed, value = imgui.slider_float(
            f"point size##{self.name}",
            float(self.point_size),
            1.0,
            40.0,
        )
        if changed:
            self.point_size = value
            self._last_key = None
            is_changed = True

        changed, value = imgui.slider_float(
            f"line width##{self.name}",
            float(self.line_width),
            1.0,
            8.0,
        )
        if changed:
            self.line_width = value
            self._last_key = None
            is_changed = True

        changed, value = imgui.slider_float(
            f"axis length##{self.name}",
            float(self.axis_length),
            0.02,
            1.0,
        )
        if changed:
            self.axis_length = value
            self._last_key = None
            is_changed = True

        return is_changed
