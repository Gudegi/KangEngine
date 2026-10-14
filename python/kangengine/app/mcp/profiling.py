"""Bounded frame profiling jobs advanced on the App owner thread."""

from collections import OrderedDict
from copy import deepcopy
import uuid

from ..capture import AutomationError
from .inspection import _bounded_integer, _frame_snapshot


class ProfileRunner:
    def __init__(self):
        self.jobs = OrderedDict()
        self.active = None

    def start(self, app, *, frame_count=60):
        _bounded_integer(frame_count, "frame_count", 1, 240)
        if self.active is not None:
            raise AutomationError("BUSY", "A profiling job is already running")
        if app.run_config is not None and not app.run_config.render_enabled:
            raise AutomationError("UNSUPPORTED", "Profiling requires rendering")
        renderer = app.get_renderer()
        job = {
            "job_id": uuid.uuid4().hex,
            "state": "running",
            "frame_count": frame_count,
            "frames_collected": 0,
            "first_frame_index": int(app.frame_index),
            "previously_enabled": bool(renderer.profiler_enabled),
            "error": None,
            "frames": [],
        }
        renderer.set_profiler_enabled(True)
        while len(self.jobs) >= 4:
            self.jobs.popitem(last=False)
        self.jobs[job["job_id"]] = job
        self.active = job
        return self.status(job["job_id"])

    def _job(self, job_id):
        try:
            return self.jobs[job_id]
        except KeyError:
            raise AutomationError(
                "NOT_FOUND", "Profiling job is not retained"
            ) from None

    def status(self, job_id):
        return deepcopy({k: v for k, v in self._job(job_id).items() if k != "frames"})

    def result(self, job_id, *, offset=0, max_frames=30):
        _bounded_integer(offset, "offset", 0, 240)
        _bounded_integer(max_frames, "max_frames", 1, 60)
        job = self._job(job_id)
        if job["state"] == "running":
            raise AutomationError("NOT_READY", "Profiling job is still running")
        end = min(offset + max_frames, len(job["frames"]))
        return {
            **self.status(job_id),
            "offset": offset,
            "frames": deepcopy(job["frames"][offset:end]),
            "next_offset": end if end < len(job["frames"]) else None,
            "gpu_results": "snapshot_at_job_end_no_wait",
        }

    def tick(self, app):
        if self.active is None:
            return
        job = self.active
        job["frames_collected"] = min(
            job["frame_count"], int(app.frame_index) - job["first_frame_index"]
        )
        if job["frames_collected"] == job["frame_count"]:
            self._finish(app, "completed")

    def _finish(self, app, state):
        job = self.active
        renderer = app.get_renderer()
        try:
            first = job["first_frame_index"]
            end = min(int(app.frame_index), first + job["frame_count"])
            job["frames"] = [
                _frame_snapshot(frame)
                for frame in renderer.frame_profile_history()
                if first <= frame.frame_index < end
            ]
            job["frames_collected"] = len(job["frames"])
            job["state"] = state
            if state == "completed" and len(job["frames"]) != job["frame_count"]:
                job["state"] = "failed"
                job["error"] = "Requested frames are missing from profiler history"
        except Exception as exc:
            job["state"] = "failed"
            job["error"] = str(exc)
        finally:
            renderer.set_profiler_enabled(job["previously_enabled"])
            self.active = None

    def cancel(self, app, *, job_id):
        if self._job(job_id) is self.active:
            self._finish(app, "cancelled")
        return self.status(job_id)

    def close(self, app):
        if self.active is not None:
            self._finish(app, "cancelled")
