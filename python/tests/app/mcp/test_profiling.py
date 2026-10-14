"""Verify App-side MCP profiling jobs, restoration and bounded results."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kangengine as ke
from kangengine.app.automation import AutomationService
from kangengine.app.capture import AutomationError
from kangengine.app.mcp.profiling import ProfileRunner
from profile_helpers import _frame


class ProfileApp:
    def __init__(self, enabled=False):
        self.frame_index = 10
        self.run_config = None
        self.requested = False
        self.frames = []
        self.renderer = SimpleNamespace(
            profiler_enabled=enabled,
            frame_profile_history=lambda: self.frames,
            set_profiler_enabled=lambda value: setattr(
                self.renderer, "profiler_enabled", value
            ),
        )

    def get_renderer(self):
        return self.renderer

    def _request_automation_frame(self):
        self.requested = True

    def frame(self, service):
        requested, self.requested = self.requested, False
        if requested:
            service.on_frame_started()
            service.on_frame_rendered()
        if self.renderer.profiler_enabled:
            self.frames.append(
                _frame(self.frame_index, ke.render.ProfileSampleStatus.PENDING)
            )
        self.frame_index += 1


@pytest.mark.parametrize("enabled", [False, True])
def test_exact_frame_range_restore_paging_and_idle(monkeypatch, enabled):
    monkeypatch.setattr("kangengine.app.mcp.inspection.engine_status", lambda app: {})
    app = ProfileApp(enabled)
    app.frames.append(_frame(9, ke.render.ProfileSampleStatus.UNSUPPORTED))
    service = AutomationService(app)
    try:
        future = service.submit("profile_start", frame_count=3)
        app.frame(service)
        job_id = future.result()["job_id"]
        for _ in range(3):
            app.frame(service)
        runner = service._profile_runner
        assert runner.status(job_id)["state"] == "completed"
        assert app.renderer.profiler_enabled is enabled
        assert not app.requested
        first = runner.result(job_id, max_frames=2)
        assert [f["frame_index"] for f in first["frames"]] == [10, 11]
        assert first["next_offset"] == 2
        last = runner.result(job_id, offset=2)
        assert [f["frame_index"] for f in last["frames"]] == [12]
        assert last["next_offset"] is None
        assert last["frames"][0]["samples"][0]["duration_ms"] is None
        last["frames"].clear()
        assert len(runner.result(job_id)["frames"]) == 3
    finally:
        service.close()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ending", ["cancel", "close", "missing", "error"])
def test_profile_restores_setting_on_every_terminal_path(enabled, ending):
    app = ProfileApp(enabled)
    runner = ProfileRunner()
    job_id = runner.start(app, frame_count=2)["job_id"]
    app.frame_index += 1
    app.frames.append(_frame(10, ke.render.ProfileSampleStatus.PENDING))
    if ending == "cancel":
        runner.cancel(app, job_id=job_id)
    elif ending == "close":
        runner.close(app)
    else:
        app.frame_index += 1
        if ending == "error":
            app.renderer.frame_profile_history = Mock(
                side_effect=RuntimeError("read failed")
            )
        runner.tick(app)
    assert app.renderer.profiler_enabled is enabled
    assert runner.active is None
    assert runner.status(job_id)["state"] == (
        "failed" if ending in {"missing", "error"} else "cancelled"
    )


def test_profile_bounds_busy_eviction_and_cancel_before_execution(monkeypatch):
    app = ProfileApp()
    runner = ProfileRunner()
    for count in (0, 241, True, 1.5):
        with pytest.raises(AutomationError, match="frame_count"):
            runner.start(app, frame_count=count)
    first = runner.start(app)["job_id"]
    with pytest.raises(AutomationError, match="already running"):
        runner.start(app)
    with pytest.raises(AutomationError, match="still running"):
        runner.result(first)
    runner.cancel(app, job_id=first)
    for _ in range(4):
        runner.start(app)
        runner.close(app)
    with pytest.raises(AutomationError, match="not retained"):
        runner.status(first)
    monkeypatch.setattr("kangengine.app.mcp.inspection.engine_status", lambda app: {})
    service = AutomationService(app)
    try:
        pending = service.submit("profile_start")
        pending.cancel()
        app.frame(service)
        assert not app.renderer.profiler_enabled
        assert service._profile_runner is None
    finally:
        service.close()
