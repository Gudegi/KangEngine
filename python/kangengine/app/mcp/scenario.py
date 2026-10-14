"""Run optional MCP scenario targets in bounded jobs on the App thread."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import time
from typing import Protocol
import uuid

from ..capture import AutomationError

STEP_SECONDS = 1.0 / 60.0
MAX_DURATION = 120.0
MAX_RETAINED_JOBS = 4
STEPS_PER_FRAME = 8
FRAME_BUDGET_SECONDS = 0.004


class McpScenarioTarget(Protocol):
    """MCP extension contract for target-owned playback and capture.

    All methods run on the App owner thread. ``begin`` must suppress live input
    and automatic advancement until ``end``, including rendering-only frames.
    ``end`` restores input ownership, not the pose before the job. ``describe``
    and ``snapshot`` return bounded JSON-compatible copies, never native views.
    """

    def describe(self) -> dict:
        """Return supported ``scenarios`` and data/configuration identifiers."""
        ...

    def begin(self, scenario: str) -> None:
        """Acquire exclusive playback control for the named input policy."""
        ...

    def reset(self) -> None:
        """Reset logical time and all state affecting playback."""
        ...

    def step(self, dt: float) -> None:
        """Advance by dt seconds; the final step may be shorter than 1/60 s."""
        ...

    def sync_visuals(self) -> None:
        """Apply the current pose without advancing logical time."""
        ...

    def snapshot(self) -> dict:
        """Return the current pose and domain-specific validation results."""
        ...

    def end(self) -> None:
        """Release playback control, including after failure or cancellation."""
        ...


class McpScenarioRunner:
    def __init__(self, targets: dict[str, McpScenarioTarget]):
        self.targets = targets
        self.jobs = OrderedDict()
        self.active = None

    def list_targets(self):
        return {
            "targets": [
                {
                    "name": name,
                    **deepcopy(target.describe()),
                    "capabilities": {
                        "reset": True,
                        "fixed_step": True,
                        "capture": True,
                    },
                }
                for name, target in self.targets.items()
            ]
        }

    def start(self, *, target: str, scenario: str, duration: float, output_path=None):
        if self.active is not None:
            raise AutomationError("BUSY", "A scenario job is already active")
        if target not in self.targets:
            raise AutomationError("NOT_FOUND", f"Unknown scenario target: {target}")
        if (
            isinstance(duration, bool)
            or not math.isfinite(duration)
            or not 0 <= duration <= MAX_DURATION
        ):
            raise ValueError(
                f"duration must be finite and between 0 and {MAX_DURATION} seconds"
            )
        instance = self.targets[target]
        description = deepcopy(instance.describe())
        if scenario not in description["scenarios"]:
            raise AutomationError("UNSUPPORTED", f"Unsupported scenario: {scenario}")
        # Round only values within floating-point noise of a whole step.
        ratio = duration / STEP_SECONDS
        nearest = round(ratio)
        full_steps = (
            nearest
            if math.isclose(ratio, nearest, abs_tol=1e-10, rel_tol=0)
            else math.floor(ratio)
        )
        remainder = max(0.0, duration - full_steps * STEP_SECONDS)
        if math.isclose(ratio, nearest, abs_tol=1e-10, rel_tol=0):
            remainder = 0.0
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "target": target,
            "scenario": scenario,
            "state": "running",
            "requested_duration": duration,
            "elapsed_seconds": 0.0,
            "step_seconds": STEP_SECONDS,
            "steps_completed": 0,
            "steps_total": full_steps + int(remainder > 0),
            "remainder_seconds": remainder,
            "description": description,
            "snapshot": None,
            "capture": None,
            "error": None,
        }
        while len(self.jobs) >= MAX_RETAINED_JOBS:
            self.jobs.popitem(last=False)
        self.jobs[job_id] = (job, None)
        self.active = (job, instance, output_path, None)
        try:
            instance.begin(scenario)
            instance.reset()
        except Exception as error:
            self._finish("failed", error)
        return deepcopy(job)

    def status(self, job_id: str):
        if job_id not in self.jobs:
            raise AutomationError("NOT_FOUND", "Unknown or evicted scenario job")
        return deepcopy(self.jobs[job_id][0])

    def capture(self, job_id: str):
        job = self.status(job_id)
        if job["state"] != "completed":
            raise AutomationError("NOT_READY", "Scenario capture is not complete")
        return self.jobs[job_id][1]

    def _finish(self, state, error=None, capture=None):
        job, target, _, _ = self.active
        try:
            job["snapshot"] = deepcopy(target.snapshot())
        except Exception as snapshot_error:
            error = error or snapshot_error
        try:
            target.end()
        except Exception as end_error:
            error = error or end_error
        job["state"] = "failed" if error else state
        if error:
            job["error"] = {
                "code": getattr(error, "code", "INTERNAL_ERROR"),
                "message": str(error),
            }
        if capture is not None:
            job["capture"] = {
                "frame_id": capture.frame_id,
                "width": capture.width,
                "height": capture.height,
                "output_path": capture.output_path,
            }
        self.jobs[job["job_id"]] = (job, capture)
        self.active = None

    def cancel(self, job_id: str):
        self.status(job_id)
        if self.active is not None and self.active[0]["job_id"] == job_id:
            future = self.active[3]
            # A PNG file commit already in progress must finish truthfully.
            if future is None or future.cancel():
                self._finish("cancelled")
        return self.status(job_id)

    def tick(self, submit_capture):
        if self.active is None:
            return
        job, target, output_path, future = self.active
        try:
            if future is not None:
                if future.done():
                    self._finish("completed", capture=future.result())
                return
            started = time.monotonic()
            for _ in range(STEPS_PER_FRAME):
                if job["steps_completed"] == job["steps_total"]:
                    break
                final_step = job["steps_completed"] + 1 == job["steps_total"]
                dt = (
                    job["remainder_seconds"]
                    if final_step and job["remainder_seconds"]
                    else STEP_SECONDS
                )
                target.step(dt)
                job["steps_completed"] += 1
                job["elapsed_seconds"] = (
                    job["requested_duration"]
                    if final_step
                    else job["steps_completed"] * STEP_SECONDS
                )
                if time.monotonic() - started >= FRAME_BUDGET_SECONDS:
                    break
            target.sync_visuals()
            if job["steps_completed"] == job["steps_total"]:
                job["state"] = "capturing"
                future = submit_capture(output_path=output_path)
                self.active = (job, target, output_path, future)
        except Exception as error:
            self._finish("failed", error)

    def close(self):
        if self.active is not None:
            future = self.active[3]
            if future is not None:
                future.cancel()
                if (
                    future.done()
                    and not future.cancelled()
                    and future.exception() is None
                ):
                    self._finish("completed", capture=future.result())
                    return
            self._finish("cancelled")
