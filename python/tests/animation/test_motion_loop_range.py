"""Check inclusive loop ranges without a graphics context or physics reset."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kangengine as ke
from kangengine.app.playback_controller import PlaybackController


def make_editor():
    motion = SimpleNamespace(duration=lambda: 9.9, fps=lambda: 10, num_frames=lambda: 100)
    return ke.motion_module.MotionEditor(motion)


def test_panel_includes_last_frame_and_preserves_overshoot():
    panel = make_editor().panel
    panel.set_loop_range(start_frame=10, end_frame=12)
    assert panel.loop_range() == (10, 12)
    panel.set_current_time(1.2)
    assert panel.current_frame() == 12
    panel.set_current_time(1.3)
    assert panel.current_time() == pytest.approx(1.0)
    panel.set_current_time(2.05)
    assert panel.current_time() == pytest.approx(1.15)
    panel.set_current_time(0.1)
    assert panel.current_frame() == 10


def test_paused_seek_and_disabled_loop_do_not_enforce_range():
    panel = make_editor().panel
    panel.set_loop_range(10, 12)
    panel.set_current_frame(frame=80)
    panel.set_current_time(8.0)
    assert panel.current_frame() == 80
    panel.set_loop(False)
    panel.set_playing(True)
    panel.set_current_time(8.1)
    assert panel.current_frame() == 81
    assert panel.loop_range() == (10, 12)
    panel.set_loop(True)
    panel.clear_loop_range()
    panel.set_current_time(10.1)
    assert panel.current_time() == pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("start,end", [(-1, 12), (13, 12), (0, 100)])
def test_invalid_ranges_leave_existing_bounds_unchanged(start, end):
    panel = make_editor().panel
    panel.set_loop_range(10, 12)
    with pytest.raises(ValueError, match="Loop range"):
        panel.set_loop_range(start, end)
    assert panel.loop_range() == (10, 12)


def test_single_frame_range_and_new_motion_reset():
    panel = make_editor().panel
    panel.set_loop_range(10, 10)
    panel.set_current_time(1.15)
    assert panel.current_frame() == 10
    panel.set_motion("new", 5, 10.0)
    assert panel.loop_range() is None
    panel.set_loop_range(0, 0)
    panel.clear_motions()
    assert panel.loop_range() is None


def test_editor_loops_without_rendering_or_playback_state_changes():
    editor = make_editor()
    app = Mock()
    app.is_simulation_paused.return_value = False
    controller = PlaybackController(app)
    controller.add_target(editor)
    editor.panel.set_loop_range(10, 12)
    editor.player.time = 1.2
    assert editor.update(0.1)
    assert editor.player.time == pytest.approx(1.0)
    assert editor.update(1.05)
    assert editor.player.time == pytest.approx(1.15)
    assert controller.playing and editor.player.playing
    app.set_simulation_paused.assert_not_called()
    editor.set_playing(False)
    editor.player.set_frame(80)
    assert not editor.update(1.0)
    assert editor.player.time == 8.0


def test_editor_range_uses_panel_frame_rate_for_mixed_tracks():
    editor = make_editor()
    editor.panel.set_motions(["slow", "fast"], [100, 200], [10.0, 20.0])
    editor.panel.set_loop_range(20, 25)
    editor.player.time = 1.25
    editor.update(0.1)
    assert editor.player.time == pytest.approx(1.05)


def test_player_whole_motion_loop_and_non_looping_behavior_are_preserved():
    editor = make_editor()
    editor.player.time = 9.9
    editor.update(0.2)
    assert editor.player.time == pytest.approx(0.1)
    editor.panel.set_loop_range(10, 12)
    editor.player.loop = False
    editor.player.time = 9.8
    editor.update(1.0)
    assert editor.player.time == 9.9
