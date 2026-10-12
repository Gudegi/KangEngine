"""Device-independent movement goals; no window or physical controller needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from kangengine import keys
from kangengine.input import GamepadState, Input, LocomotionInput


class FakeApp:
    keyboard_input_captured = False

    def __init__(self):
        self.held = set()
        self.pad = GamepadState()
        self.polls = 0
        self.input = Input(self)
        self.input.gamepad().state = self.poll_pad

    def is_key_down(self, key):
        return key in self.held

    def poll_pad(self):
        self.polls += 1
        return self.pad


@pytest.mark.parametrize("up,forward", (("Z", "X"), ("Y", "Z")))
def test_wasd_and_stick_agree_on_axes_and_diagonal_speed(up, forward):
    app = FakeApp()
    controls = LocomotionInput(app, up_axis=up, forward_axis=forward)
    facing = np.eye(3)["XYZ".index(forward)]
    app.held = {keys.W, keys.D}
    keyboard, direction = controls.sample(facing=facing)
    assert np.linalg.norm(keyboard) == pytest.approx(1.4)
    expected_right = np.cross(facing, np.eye(3)["XYZ".index(up)])
    np.testing.assert_allclose(direction, (facing + expected_right) / np.sqrt(2))
    app.held.clear()
    app.pad = GamepadState(connected=True, left_x=1, left_y=-1)
    gamepad, pad_direction = controls.sample(facing=facing)
    np.testing.assert_allclose(gamepad, keyboard)
    np.testing.assert_allclose(pad_direction, direction)


def test_run_analog_deadzone_and_release():
    app = FakeApp()
    controls = LocomotionInput(app, deadzone=0.2)
    app.held = {keys.W, keys.RIGHT_SHIFT}
    np.testing.assert_allclose(controls.sample(facing=(1, 0, 0))[0], (3.5, 0, 0))
    app.held.clear()
    app.pad = GamepadState(connected=True, left_y=-0.6, right_bumper=True)
    velocity, facing = controls.sample(facing=(1, 0, 0))
    np.testing.assert_allclose(velocity, (1.75, 0, 0))
    assert app.polls == 1  # One coherent snapshot per sample.
    app.pad = GamepadState(connected=True, left_x=0.1, left_y=-0.1)
    np.testing.assert_array_equal(controls.sample(facing=facing)[0], 0)
    app.pad = GamepadState(connected=False, left_y=-1, right_bumper=True)
    stopped, retained = controls.sample(facing=facing)
    np.testing.assert_array_equal(stopped, 0)
    np.testing.assert_array_equal(retained, facing)


def test_keyboard_priority_ui_capture_and_explicit_sources():
    app = FakeApp()
    controls = LocomotionInput(app)
    app.held = {keys.W, keys.S}
    app.pad = GamepadState(connected=True, left_x=1)
    np.testing.assert_array_equal(controls.sample(facing=(1, 0, 0))[0], 0)
    np.testing.assert_allclose(
        controls.sample(facing=(1, 0, 0), source="gamepad")[0], (0, -1.4, 0)
    )
    app.held = {keys.W}
    app.keyboard_input_captured = True
    np.testing.assert_array_equal(
        controls.sample(facing=(1, 0, 0), source="keyboard")[0], 0
    )
    np.testing.assert_allclose(controls.sample(facing=(1, 0, 0))[0], (0, -1.4, 0))
    app.keyboard_input_captured = False
    np.testing.assert_allclose(controls.sample(facing=(1, 0, 0))[0], (1.4, 0, 0))


def test_strafe_and_right_stick_facing_are_independent_of_velocity():
    app = FakeApp()
    controls = LocomotionInput(app)
    app.held = {keys.D}
    velocity, facing = controls.sample(facing=(1, 0, 0), face_movement=False)
    np.testing.assert_allclose(velocity, (0, -1.4, 0))
    np.testing.assert_array_equal(facing, (1, 0, 0))
    app.held.clear()
    app.pad = GamepadState(connected=True, left_y=-1, right_x=1)
    velocity, facing = controls.sample(facing=(1, 0, 0))
    np.testing.assert_allclose(velocity, (1.4, 0, 0))
    np.testing.assert_allclose(facing, (0, -1, 0))


@pytest.mark.parametrize(
    "options",
    ({"deadzone": 1}, {"deadzone": -1}, {"deadzone": np.nan}, {"forward_axis": "Z"}),
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        LocomotionInput(FakeApp(), **options)


def test_follow_camera_preserves_user_orbit_and_zoom(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/animation"))
    from motion_matching_common import follow_character

    position, target = np.array((8.0, 3.0, 5.0)), np.array((2.0, 1.0, 0.0))
    camera = SimpleNamespace(
        get_camera_pos=lambda: position, get_target_pos=lambda: target
    )
    result = []
    app = SimpleNamespace(
        controller=SimpleNamespace(root_position=np.array((3.0, 2.0, 1.0))),
        follow_camera=True,
        _camera_follow_root=np.array((1.0, 2.0, 1.0)),
        get_camera=lambda: camera,
        set_camera_view=lambda p, t: result.append((p, t)),
    )
    follow_character(app)
    np.testing.assert_allclose(result[0][0], position + (2, 0, 0))
    np.testing.assert_allclose(result[0][0] - result[0][1], position - target)
    app.follow_camera = False
    app.controller.root_position += (1, 0, 0)
    follow_character(app)
    assert len(result) == 1
