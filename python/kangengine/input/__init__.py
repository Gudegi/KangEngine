"""Keyboard and standard gamepad input facades."""

from .gamepad import Gamepad, GamepadState
from .locomotion import LocomotionInput


class Keyboard:
    def __init__(self, app):
        self._app = app

    def down(self, key: int) -> bool:
        return self._app.is_key_down(key)

    def pressed(self, key: int) -> bool:
        return self._app.was_key_pressed(key)

    def released(self, key: int) -> bool:
        return self._app.was_key_released(key)


class Input:
    def __init__(self, app):
        self.keyboard = Keyboard(app)
        self._app = app
        self._gamepads = {}

    def gamepad(self, index: int = 0) -> Gamepad:
        index = int(index)
        if index not in self._gamepads:
            self._gamepads[index] = Gamepad(self._app, index)
        return self._gamepads[index]


from .gamepad_visualizer import GamepadVisualizer
from .keyboard_visualizer import KeyboardVisualizer


__all__ = [
    "Gamepad",
    "GamepadState",
    "GamepadVisualizer",
    "KeyboardVisualizer",
    "Input",
    "Keyboard",
    "LocomotionInput",
]
