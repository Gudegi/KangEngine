"""Exercise public profiler snapshots, export and real renderer counters."""
import json
from pathlib import Path
import tempfile
import time
import math
import statistics

import kangengine as ke
import numpy as np


def main(*, ui=False):
    app: ke.App = ke.App()
    app.initialize(width=64, height=64, hide_ui=not ui, headless=True)
    app.set_vsync(enabled=False)
    renderer: ke.render.Renderer = app.get_renderer()
    assert not renderer.profiler_enabled
    assert renderer.latest_frame_profile() is None
    assert renderer.frame_profile_history() == ()
    assert renderer.profile_summary().frame_count == 0
    mesh = ke.scene.Prim.create_rectangle_data(1.0, 1.0, 1.0)
    material = app.create_standard_materials().common
    batch = app.scene.add_mesh("/profile/batch", mesh, material=material,
                              transform_source=ke.render.TransformSource.EXTERNAL_BUFFER)
    transforms = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    batch.set_transform_buffer(transforms)
    app.render_frame_once()
    assert renderer.latest_frame_profile() is None
    renderer.set_profiler_enabled(enabled=True)
    app.render_frame_once()
    frame: ke.render.FrameProfile = renderer.latest_frame_profile()
    supported = renderer.profiler_capabilities.pass_timestamps
    assert frame is not None and frame.revision == 0
    assert frame.finalized == (not supported)
    assert frame.backend == ke.render.BackendType.OPENGL
    assert math.isfinite(float(frame.metadata["frame_interval_ms"]))
    assert float(frame.metadata["frame_interval_ms"]) > 0
    assert frame.gpu_latency_frames is None
    assert frame.counters.draw_calls > 0
    assert frame.counters.indexed_draw_calls > 0
    assert frame.counters.buffer_upload_bytes > 0
    samples = {(s.path, s.domain): s for s in frame.samples}
    external = renderer.profiler_capabilities.external_timestamps
    assert samples["frame", ke.render.ProfileTimingDomain.GPU].status == (
        ke.render.ProfileSampleStatus.PENDING if external else ke.render.ProfileSampleStatus.UNSUPPORTED)
    for path in ("frame", "scene_sync", "render/shadow", "render/scene/opaque",
                 "command_record", "submit", "present", "event_poll"):
        sample = samples[path, ke.render.ProfileTimingDomain.CPU]
        assert sample.available and sample.duration_ms >= 0, path
    try:
        frame.counters.draw_calls = 0
        raise AssertionError("snapshot is mutable")
    except AttributeError:
        pass
    original = frame.frame_index
    pending_summary = renderer.profile_summary()
    if supported:
        gpu_summary = next(s for s in pending_summary.scopes
                           if s.path == "render/scene/opaque" and s.domain == ke.render.ProfileTimingDomain.GPU)
        assert gpu_summary.pending_frames == 1 and gpu_summary.mean_ms is None
    if supported:
        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / "pending.json"
            renderer.export_profile_json(path=str(pending_path))
            pending = json.loads(pending_path.read_text())["frames"][0]
            assert not pending["finalized"] and pending["revision"] == 0
            assert any(s["status"] == "pending" and s["duration_ms"] is None for s in pending["samples"])
    for _ in range(3):
        app.render_frame_once()
    assert frame.frame_index == original
    assert renderer.latest_frame_profile().frame_index > original
    assert isinstance(frame.samples, tuple)
    renderer.set_profiler_enabled(enabled=False)
    retained = renderer.latest_frame_profile().frame_index
    app.render_frame_once()
    assert renderer.latest_frame_profile().frame_index == retained
    for _ in range(120):
        if all(f.finalized for f in renderer.frame_profile_history()):
            break
        app.render_frame_once()
        time.sleep(0.001)
    completed = renderer.frame_profile_history()[0]
    assert completed.finalized
    if supported:
        assert not frame.finalized and frame.revision == 0  # Previously returned object is immutable.
        assert completed.revision >= 1 and completed.gpu_latency_frames >= 3
        gpu = [s for s in completed.samples if s.domain == ke.render.ProfileTimingDomain.GPU and s.available]
        assert {"render/scene/opaque", "render/post/tone_map", "render/final_resolve"} <= {s.path for s in gpu}
        assert all(s.duration_ms >= 0 for s in gpu)
        if external:
            timings = {s.path: s.duration_ms for s in gpu}
            assert "frame" in timings
            assert sum(s.path == "frame" for s in gpu) == 1
            assert timings["frame"] >= max(v for p, v in timings.items() if p != "frame")
            if ui:
                assert "render/ui" in timings
            else:
                assert "render/native_blit" in timings and "render/ui" not in timings
            assert "present" in completed.metadata["gpu_timing_exclusions"]
    assert renderer.latest_frame_profile().frame_index == retained
    summary = renderer.profile_summary(max_frames=3)
    assert summary.frame_count == 3 and isinstance(summary.scopes, tuple)
    assert pending_summary.frame_count == 1
    for scope in summary.scopes:
        totals = []
        for captured in renderer.frame_profile_history()[-3:]:
            matches = [s for s in captured.samples if (s.path, s.domain) == (scope.path, scope.domain)]
            if matches and all(s.available for s in matches) and not captured.dropped_samples:
                totals.append(sum(s.duration_ms for s in matches))
        assert scope.ready_frames == len(totals)
        if totals:
            assert math.isclose(scope.mean_ms, statistics.mean(totals))
            assert math.isclose(scope.median_ms, statistics.median(totals))
            assert scope.p95_ms == sorted(totals)[math.ceil(len(totals) * .95) - 1]
            assert scope.max_ms == max(totals)
        else:
            assert scope.mean_ms is None
    try:
        summary.scopes[0].mean_ms = 0
        raise AssertionError("summary is mutable")
    except AttributeError:
        pass
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        renderer.export_profile_json(path=str(path))
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1 and len(data["frames"]) == 4
        assert data["frames"][0]["counters"]["draw_calls"] == frame.counters.draw_calls
        exported_gpu_frame = [s for s in data["frames"][0]["samples"]
                              if s["domain"] == "gpu" and s["path"] == "frame"]
        assert len(exported_gpu_frame) == 1
        assert (exported_gpu_frame[0]["duration_ms"] is not None) == external
        renderer.export_profile_json(path=str(path), max_frames=3)
        window = json.loads(path.read_text())
        assert len(window["frames"]) == window["summary"]["frame_count"] == 3
        assert window["summary"]["first_frame_index"] == summary.first_frame_index
        for actual, exported in zip(summary.scopes, window["summary"]["scopes"]):
            assert actual.path == exported["path"]
            assert actual.ready_frames == exported["ready_frames"]
            if actual.mean_ms is None:
                assert exported["mean_ms"] is None
            else:
                assert math.isclose(actual.mean_ms, exported["mean_ms"], rel_tol=1e-9)
    renderer.set_profiler_enabled(enabled=True)
    app.render_frame_once()
    assert renderer.latest_frame_profile().capture_id == frame.capture_id + 1
    assert renderer.profile_summary().frame_count == 1
    print("PASS: renderer profiler snapshots, asynchronous GPU passes, toggle and JSON export")


if __name__ == "__main__":
    import sys
    main(ui="--ui" in sys.argv)
