"""Check that panel controls, motion players, and the simulation clock agree."""

import gc
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import kangengine as ke
from kangengine.app.playback_controller import PlaybackController


def panel():
    result = ke.MotionSequencerPanel()
    result.set_motion(motion_name="test", num_frames=100, fps=30.0)
    return result


def test_frame_seek_pauses_all_targets_and_next_toggle_resumes():
    controller = PlaybackController()
    first, second = panel(), panel()
    controller.add_target(first)
    controller.add_target(second)
    first.set_current_frame(frame=37)
    assert not first.is_playing() and not second.is_playing()
    assert not controller.playing
    controller.toggle_play()
    assert first.is_playing() and second.is_playing() and controller.playing
    assert first.current_frame() == 37


def test_editor_and_panel_changes_propagate_in_both_directions():
    motion = SimpleNamespace(duration=lambda: 3.3, fps=lambda: 30, num_frames=lambda: 100)
    editor = ke.motion_module.MotionEditor(motion)
    other = panel()
    controller = PlaybackController()
    controller.add_target(editor)
    controller.add_target(other)
    editor.panel.set_current_frame(frame=20)
    assert not editor.player.playing and not other.is_playing() and not controller.playing
    controller.toggle_play()
    assert editor.player.playing and editor.panel.is_playing()
    editor.set_playing(False)
    assert not controller.playing and not other.is_playing()
    other.set_playing(True)
    assert controller.playing and editor.player.playing and editor.panel.is_playing()


def test_panel_pause_and_resume_update_the_app_clock():
    app = Mock()
    app.is_simulation_paused.return_value = False
    controller = PlaybackController(app)
    target = panel()
    controller.add_target(target)
    target.set_current_frame(frame=15)
    app.set_simulation_paused.assert_called_once_with(True)
    toggle = app._set_playback_toggle_callback.call_args.args[0]
    toggle()
    assert target.is_playing() and controller.playing
    app.set_simulation_paused.assert_called_with(False)
    target.set_playing(False)
    assert not controller.playing
    app.set_simulation_paused.assert_called_with(True)


def test_first_target_preserves_an_initially_paused_app():
    app = Mock()
    app.is_simulation_paused.return_value = True
    controller = PlaybackController(app)
    target = panel()
    controller.add_target(target)
    assert not target.is_playing() and not controller.playing
    app._set_playback_toggle_callback.call_args.args[0]()
    assert target.is_playing() and controller.playing
    app.set_simulation_paused.assert_called_once_with(False)


def test_duplicate_registration_preserves_paused_state():
    controller = PlaybackController()
    target = panel()
    controller.add_target(target)
    target.set_playing(False)
    controller.add_target(target)
    assert controller.get_targets() == [target]
    assert not target.is_playing() and not controller.playing


def test_retained_panel_does_not_keep_controller_alive():
    target = panel()
    controller = PlaybackController()
    controller.add_target(target)
    owner = weakref.ref(controller)
    del controller
    gc.collect()
    assert owner() is None
    target.set_playing(False)
