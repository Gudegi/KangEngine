"""Check fixed logical time, cancellation and capture ownership without GL."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from kangengine.app.automation import AutomationService
from kangengine.app.capture import AutomationError
from kangengine.app.mcp.scenario import McpScenarioRunner, MAX_RETAINED_JOBS


class Target:
    def __init__(self):
        self.owned = False
        self.reset()

    def describe(self):
        return {"scenarios": ["demo"], "fixture_version": 1}

    def begin(self, scenario):
        self.owned = True

    def reset(self):
        self.time = 0.0
        self.steps = []

    def step(self, dt):
        assert self.owned
        self.time += dt
        self.steps.append(dt)

    def sync_visuals(self):
        pass

    def snapshot(self):
        return {"time": self.time, "steps": len(self.steps)}

    def end(self):
        self.owned = False


@pytest.mark.parametrize("duration,steps", [(0, 0), (13, 780), (24, 1440), (0.025, 2)])
def test_fixed_steps_and_capture_hold(duration, steps):
    target = Target()
    runner = McpScenarioRunner({"fixture": target})
    job = runner.start(target="fixture", scenario="demo", duration=duration)
    future = Future()
    while runner.status(job["job_id"])["state"] == "running":
        runner.tick(lambda **kwargs: future)
    assert len(target.steps) == steps
    assert target.time == pytest.approx(duration)
    assert target.owned
    for _ in range(3):
        runner.tick(lambda **kwargs: pytest.fail("duplicate capture"))
    assert len(target.steps) == steps
    result = SimpleNamespace(frame_id=42, width=5, height=3, output_path=None)
    future.set_result(result)
    runner.tick(None)
    assert not target.owned
    assert runner.status(job["job_id"])["state"] == "completed"
    assert runner.capture(job["job_id"]) is result
    assert runner.status(job["job_id"])["snapshot"]["time"] == pytest.approx(duration)


def test_errors_limits_cancellation_and_retention():
    target = Target()
    runner = McpScenarioRunner({"fixture": target})
    for duration in (-1, float("nan"), float("inf"), 121, True):
        with pytest.raises(ValueError):
            runner.start(target="fixture", scenario="demo", duration=duration)
    with pytest.raises(AutomationError, match="Unknown"):
        runner.start(target="unknown", scenario="demo", duration=1)
    with pytest.raises(AutomationError, match="Unsupported"):
        runner.start(target="fixture", scenario="other", duration=1)
    first = runner.start(target="fixture", scenario="demo", duration=24)
    with pytest.raises(AutomationError, match="already active"):
        runner.start(target="fixture", scenario="demo", duration=1)
    runner.tick(None)
    assert 0 < len(target.steps) <= 8
    cancelled = runner.cancel(first["job_id"])
    assert cancelled["state"] == "cancelled" and not target.owned
    assert cancelled["snapshot"]["time"] == cancelled["elapsed_seconds"]
    for _ in range(MAX_RETAINED_JOBS):
        job = runner.start(target="fixture", scenario="demo", duration=24)
        runner.cancel(job["job_id"])
    with pytest.raises(AutomationError, match="evicted"):
        runner.status(first["job_id"])


def test_cancel_during_capture_and_committed_save():
    target = Target()
    runner = McpScenarioRunner({"fixture": target})
    for commit in (False, True):
        job = runner.start(target="fixture", scenario="demo", duration=0)
        future = Future()
        runner.tick(lambda **kwargs: future)
        if commit:
            future.set_running_or_notify_cancel()
        result = runner.cancel(job["job_id"])
        assert result["state"] == ("capturing" if commit else "cancelled")
        assert target.owned == commit
        if commit:
            future.set_result(
                SimpleNamespace(frame_id=1, width=1, height=1, output_path="saved.png")
            )
            runner.close()
            assert runner.status(job["job_id"])["state"] == "completed"
        assert not target.owned


@pytest.mark.parametrize(
    "method", ["begin", "reset", "step", "sync_visuals", "snapshot"]
)
def test_target_failures_release_ownership(method):
    target = Target()
    runner = McpScenarioRunner({"fixture": target})
    with mock.patch.object(target, method, side_effect=RuntimeError("target failed")):
        job = runner.start(target="fixture", scenario="demo", duration=1)
        if method == "snapshot":
            runner.cancel(job["job_id"])
        else:
            runner.tick(None)
    assert runner.status(job["job_id"])["state"] == "failed"
    assert not target.owned


def test_service_schedules_capture_and_returns_retained_image(tmp_path):
    target = Target()
    app = SimpleNamespace(
        frame_index=0,
        run_config=None,
        _mcp_scenario_targets={"fixture": target},
        get_width=lambda: 5,
        get_height=lambda: 3,
        read_rgb_pixels=lambda **kwargs: np.zeros((3, 5, 3), dtype=np.uint8),
        _request_automation_frame=lambda: None,
    )

    # AutomationService intentionally requires a weak-referenceable App.
    class App:
        pass

    owner = App()
    owner.__dict__.update(vars(app))
    owner.requested = False
    owner._request_automation_frame = lambda: setattr(owner, "requested", True)
    service = AutomationService(owner, output_dir=tmp_path)

    def frame():
        if owner.requested:
            owner.requested = False
            with mock.patch("kangengine.app.mcp.inspection.engine_status", return_value={}):
                service.on_frame_started()
            service.on_frame_rendered()
        owner.frame_index += 1

    try:
        target.time = 7
        cancelled = service.submit(
            "scenario_start", target="fixture", scenario="demo", duration=1
        )
        assert cancelled.cancel()
        frame()
        assert target.time == 7 and service._mcp_scenario_runner is None
        request = service.submit(
            "scenario_start", target="fixture", scenario="demo", duration=0.025
        )
        frame()
        job_id = request.result()["job_id"]
        frame()
        capture_future = service._mcp_scenario_runner.active[3]
        capture = capture_future.result(timeout=5)
        frame()
        result = service.submit("scenario_capture", job_id=job_id)
        frame()
        assert result.result() is capture
        assert target.time == pytest.approx(0.025)
        assert not target.owned
        assert not owner.requested
        for _ in range(100):
            frame()
        assert not owner.requested and target.time == pytest.approx(0.025)
        pending = service.submit(
            "scenario_start", target="fixture", scenario="demo", duration=24
        )
        frame()
        pending.result()
    finally:
        service.close()
    assert not target.owned
