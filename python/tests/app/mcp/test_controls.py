"""Verify App-side MCP camera, visibility and recording controls."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kangengine as ke
from kangengine.app.mcp import control
from kangengine.app.capture import AutomationError


def test_visibility_keeps_active_and_children_and_honors_parent():
    root = ke.scene.Prim("", ke.scene.PrimType.ROOT)
    parent = root.add_child("parent", ke.scene.PrimType.XFORM)
    child = parent.add_child("child", ke.scene.PrimType.XFORM)
    parent.set_active(False)
    child.set_visible(False)
    app = SimpleNamespace(
        scene=SimpleNamespace(
            get_root_prim=lambda: root, get_prim_at_path=root.get_prim_at_path
        )
    )
    result = control.scene_set_visibility(app, path="/parent", visible=False)
    assert not result["active"]
    assert not child.is_visible()
    result = control.scene_set_visibility(app, path="/parent/child", visible=True)
    assert result["visible"] and not result["effective_visible"]
    assert not result["visible_in_hierarchy"]
    with pytest.raises(AutomationError):
        control.scene_set_visibility(app, path="/parent", visible="false")


@pytest.mark.parametrize(
    "position,target",
    [
        ([0, 0, 0], [0, 0, 0]),
        ([1, 2], [0, 0, 0]),
        ([float("nan"), 1, 2], [0, 0, 0]),
        ([1e100, 0, 0], [0, 0, 0]),
        ([1e30, 0, 0], [0, 0, 0]),
        ([1e-5, 0, 0], [0, 0, 0]),
    ],
)
def test_invalid_camera_cannot_partially_mutate(position, target):
    app = SimpleNamespace(has_active_scene_camera=lambda: False, set_camera_view=Mock())
    with pytest.raises(AutomationError):
        control.camera_set(app, position=position, target=target)
    app.set_camera_view.assert_not_called()


def test_scene_camera_is_not_silently_replaced():
    app = SimpleNamespace(has_active_scene_camera=lambda: True, set_camera_view=Mock())
    with pytest.raises(AutomationError, match="active scene camera"):
        control.camera_set(app, position=[1, 1, 1], target=[0, 0, 0])
    app.set_camera_view.assert_not_called()


def test_recording_confines_files_and_rejects_busy_unready_and_headless(tmp_path):
    app = SimpleNamespace(
        run_config=None,
        frame_index=1,
        is_video_recording=lambda: False,
        start_video_recording=Mock(),
        stop_video_recording=Mock(),
        get_video_recording_path=lambda: tmp_path / "new.mp4",
    )
    existing = tmp_path / "exists.mp4"
    existing.write_bytes(b"keep")
    for path in ("../escape.mp4", "bad.gif", "exists.mp4"):
        with pytest.raises(AutomationError):
            control.recording_start(app, output_dir=tmp_path, output_path=path)
    for fps in (0, 241, float("nan"), True):
        with pytest.raises(AutomationError):
            control.recording_start(
                app, output_dir=tmp_path, output_path="new.mp4", fps=fps
            )
    assert existing.read_bytes() == b"keep"
    app.start_video_recording.assert_not_called()
    for state, code in (
        ("busy", "BUSY"),
        ("unready", "NOT_READY"),
        ("headless", "UNSUPPORTED"),
    ):
        app.is_video_recording = lambda: state == "busy"
        app.frame_index = 0 if state == "unready" else 1
        app.run_config = SimpleNamespace(render_enabled=state != "headless")
        with pytest.raises(AutomationError) as error:
            control.recording_start(app, output_dir=tmp_path, output_path="new.mp4")
        assert error.value.code == code


def test_failed_recording_start_releases_common_controller(tmp_path):
    app = SimpleNamespace(
        run_config=None,
        frame_index=1,
        is_video_recording=lambda: False,
        start_video_recording=Mock(side_effect=RuntimeError("encoder unavailable")),
        stop_video_recording=Mock(),
    )
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        control.recording_start(app, output_dir=tmp_path, output_path="new.mp4")
    app.stop_video_recording.assert_called_once()
