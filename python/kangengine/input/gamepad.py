"""Standard gamepad state and input helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .._core import _ke


@dataclass(frozen=True)
class GamepadState:
    connected: bool = False
    name: str = ""
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    left_bumper: bool = False
    right_bumper: bool = False
    back: bool = False
    start: bool = False
    guide: bool = False
    left_thumb: bool = False
    right_thumb: bool = False
    dpad_up: bool = False
    dpad_right: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = -1.0
    right_trigger: float = -1.0


class Gamepad:
    def __init__(self, app, index: int):
        self._app = app
        self.index = int(index)

    @property
    def connected(self) -> bool:
        return bool(self._app._is_gamepad_connected(self.index))

    def state(self) -> GamepadState:
        return GamepadState(**self._app._get_gamepad_state(self.index))

    def get_left_joystick(
        self,
        *,
        camera_relative: bool = False,
        drift_threshold: float = 0.1,
        state: GamepadState | None = None,
    ):
        """Return ``(direction, strength)`` above the drift threshold."""
        if state is None:
            state = self.state()
        magnitude = math.hypot(state.left_x, state.left_y)
        drift_threshold = min(0.99, max(0.0, float(drift_threshold)))
        if not state.connected or magnitude <= drift_threshold:
            return None

        stick_right = state.left_x / magnitude
        stick_forward = -state.left_y / magnitude
        strength = min(
            1.0,
            (magnitude - drift_threshold) / (1.0 - drift_threshold),
        )

        if self._app.up_axis == _ke.UpAxis.Z:
            direction = np.array([stick_right, stick_forward, 0.0])
            up_index = 2
        elif self._app.up_axis == _ke.UpAxis.Y:
            direction = np.array([stick_right, 0.0, stick_forward])
            up_index = 1
        else:
            raise ValueError("Gamepad navigation requires Y-up or Z-up.")

        if not camera_relative:
            return direction, strength

        camera = self._app.get_camera()
        forward = np.array(camera.get_camera_look_dir(), dtype=float)
        right = np.array(camera.get_camera_right_dir(), dtype=float)
        forward[up_index] = 0.0
        right[up_index] = 0.0
        forward_length = np.linalg.norm(forward)
        right_length = np.linalg.norm(right)
        if forward_length == 0.0 or right_length == 0.0:
            return None

        direction = (
            right / right_length * stick_right
            + forward / forward_length * stick_forward
        )
        return direction, strength

    def get_right_joystick(
        self,
        *,
        orbit: bool = False,
        drift_threshold: float = 0.1,
        state: GamepadState | None = None,
    ):
        """Return ``(yaw_pitch, strength)`` for look or orbit controls."""
        if state is None:
            state = self.state()
        magnitude = math.hypot(state.right_x, state.right_y)
        drift_threshold = min(0.99, max(0.0, float(drift_threshold)))
        if not state.connected or magnitude <= drift_threshold:
            return None

        yaw = state.right_x / magnitude
        pitch = -state.right_y / magnitude
        if orbit:
            pitch = -pitch

        strength = min(
            1.0,
            (magnitude - drift_threshold) / (1.0 - drift_threshold),
        )
        return np.array([yaw, pitch]), strength


__all__ = ["Gamepad", "GamepadState"]
