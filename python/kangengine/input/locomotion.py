"""Map keyboard/gamepad input to world-space movement goals, independently of MM."""

from __future__ import annotations

import math
import numpy as np

from .._core import _ke


class LocomotionInput:
    """WASD or sticks -> (world velocity, world facing), on a configured plane.

    W and stick-up point along forward_axis; right is cross(forward, up).
    Shift / right bumper selects run speed. The right stick can set facing
    independently of movement. Without directional input, facing is retained.
    In auto mode, held WASD takes precedence over the gamepad. Keyboard input
    captured by UI is ignored. Disconnected controllers produce no input.
    """

    def __init__(
        self, app, *, up_axis="Z", forward_axis="X", gamepad_index=0, deadzone=0.15
    ):
        if (
            up_axis not in ("X", "Y", "Z")
            or forward_axis not in ("X", "Y", "Z")
            or up_axis == forward_axis
        ):
            raise ValueError(
                "up_axis and forward_axis must be distinct X, Y, or Z axes"
            )
        if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
            raise ValueError("deadzone must be finite and in [0, 1)")
        self._app = app
        self._keyboard = app.input.keyboard
        self._gamepad = app.input.gamepad(gamepad_index)
        self._up = np.eye(3)["XYZ".index(up_axis)]
        self._forward = np.eye(3)["XYZ".index(forward_axis)]
        self._right = np.cross(self._forward, self._up)
        self._deadzone = float(deadzone)

    def _stick(self, x, y):
        magnitude = math.hypot(x, y)
        if magnitude <= self._deadzone:
            return np.zeros(3)
        strength = min(1.0, (magnitude - self._deadzone) / (1 - self._deadzone))
        return (x * self._right - y * self._forward) * (strength / magnitude)

    def sample(
        self,
        *,
        facing,
        walk_speed=1.4,
        run_speed=3.5,
        source="auto",
        face_movement=True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Poll once. source is 'auto', 'keyboard', or 'gamepad'. Speeds are units/s."""
        if source not in ("auto", "keyboard", "gamepad"):
            raise ValueError("source must be auto, keyboard, or gamepad")
        if any(not math.isfinite(s) or s < 0 for s in (walk_speed, run_speed)):
            raise ValueError("movement speeds must be finite and nonnegative")
        facing = np.asarray(facing, dtype=float)
        if facing.shape != (3,) or not np.isfinite(facing).all():
            raise ValueError("facing must contain three finite values")
        facing = facing - self._up * np.dot(facing, self._up)
        length = np.linalg.norm(facing)
        if length < 1e-8:
            raise ValueError(
                "facing must have a nonzero direction in the movement plane"
            )
        facing = facing / length
        keys = _ke.keys
        held = (False,) * 4
        if source != "gamepad" and not self._app.keyboard_input_captured:
            held = tuple(
                self._keyboard.down(key) for key in (keys.W, keys.A, keys.S, keys.D)
            )
        if source == "keyboard" or any(held):
            w, a, s, d = held
            move = (int(w) - int(s)) * self._forward + (int(d) - int(a)) * self._right
            move /= max(1.0, np.linalg.norm(move))
            running = self._keyboard.down(keys.LEFT_SHIFT) or self._keyboard.down(
                keys.RIGHT_SHIFT
            )
            look = np.zeros(3)
        else:
            state = self._gamepad.state()
            move = (
                self._stick(state.left_x, state.left_y)
                if state.connected
                else np.zeros(3)
            )
            look = (
                self._stick(state.right_x, state.right_y)
                if state.connected
                else np.zeros(3)
            )
            running = state.connected and state.right_bumper
        if np.linalg.norm(look) > 0:
            facing = look / np.linalg.norm(look)
        elif face_movement and np.linalg.norm(move) > 0:
            facing = move / np.linalg.norm(move)
        return move * (run_speed if running else walk_speed), facing
