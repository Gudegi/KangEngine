"""Fixed renderer workloads with optional native CPU/GPU pass capture.

Run with PYTHONPATH=python python/.venv/bin/python python/tools/benchmark_renderer.py.
Headless uses a hidden GLFW window and still requires a working display.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import kangengine as ke
import numpy as np


class FixtureApp(ke.App):
    update_fixture = None

    def pre_render(self):
        if self.update_fixture is not None:
            self.update_fixture()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("static", "external_cpu", "dynamic_geometry"), default="static")
    parser.add_argument("--instances", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.instances, args.samples, args.width, args.height) <= 0 or args.warmup < 0:
        parser.error("counts and dimensions must be positive; warmup must be nonnegative")
    if args.samples > 240 and args.profiler:
        parser.error("native history currently retains at most 240 frames")

    app: FixtureApp = FixtureApp()
    app.initialize(width=args.width, height=args.height, hide_ui=not args.ui,
                   headless=not args.windowed)
    if app.should_close():
        raise RuntimeError("renderer fixture requires a valid GLFW display/context")
    app.set_vsync(enabled=args.vsync)
    renderer: ke.render.Renderer = app.get_renderer()
    material: ke.material.VertexColorMaterial = app.create_standard_materials().common
    mesh: ke.scene.MeshData = ke.scene.Prim.create_rectangle_data(0.1, 0.1, 0.1)
    side = math.ceil(math.sqrt(args.instances))
    transforms = np.tile(np.eye(4, dtype=np.float32), (args.instances, 1, 1))
    transforms[:, 3, 0] = (np.arange(args.instances) % side - side / 2) * 0.15
    transforms[:, 3, 2] = (np.arange(args.instances) // side - side / 2) * 0.15
    batch = None
    if args.fixture in ("static", "dynamic_geometry"):
        views = []
        for i in range(args.instances):
            view = app.scene.add_mesh(f"/fixture/mesh_{i}", mesh, material=material)
            view.prim.set_local_translation(ke.Vec3(*transforms[i, 3, :3]))
            views.append(view)
        if args.fixture == "dynamic_geometry":
            # Shared mesh stays the bind/source resource; only GPU geometry changes.
            source = np.array([[v.x, v.y, v.z] for v in mesh.vertices], dtype=np.float32)
            positions = source.copy()
            tick = 0

            def update_geometry():
                nonlocal tick
                positions[:, 1] = source[:, 1] * (1 + 0.1 * math.sin(tick * 0.03))
                views[0].update_geometry(positions=positions)
                tick += 1

            app.update_fixture = update_geometry
    else:
        batch = app.scene.add_mesh(
            "/fixture/batch", mesh, material=material,
            transform_source=ke.render.TransformSource.EXTERNAL_BUFFER)
        batch.set_transform_buffer(transforms)

    if args.profiler:
        renderer.set_profiler_enabled(enabled=True)
    elapsed = []
    for i in range(args.warmup + args.samples):
        start = time.perf_counter()
        if batch is not None:
            transforms[:, 3, 1] = math.sin(i * 0.03) * 0.1
            batch.set_transform_buffer(transforms)
        app.render_frame_once()
        if i >= args.warmup:
            elapsed.append((time.perf_counter() - start) * 1000)

    result = {"schema_version": 1, "metadata": vars(args) | {
        "output": str(args.output), "python": platform.python_version(),
        "headless_mode": "hidden_glfw" if not args.windowed else "windowed",
        "pacing": "direct_render_frame_once", "renderer_settings": "engine_defaults",
        "gpu_timing": "pass_timestamps" if renderer.profiler_capabilities.pass_timestamps else "unsupported",
        "gpu_external_timestamps": renderer.profiler_capabilities.external_timestamps,
        "workload": "cube_grid",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "wall_coverage": "fixture_update_and_render_frame_once",
        "external_cpu_generation": "outside_native_frame",
    }, "wall_ms": {"mean": statistics.mean(elapsed), "median": statistics.median(elapsed),
                   "p95": sorted(elapsed)[math.ceil(len(elapsed) * 0.95) - 1],
                   "min": min(elapsed), "max": max(elapsed)}, "raw_wall_ms": elapsed}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    result["metadata"]["source_revision"] = revision.stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    result["metadata"]["tracked_worktree_dirty"] = bool(dirty.stdout.strip())
    cache = Path(__file__).resolve().parents[2] / "build/release/CMakeCache.txt"
    if cache.is_file():
        settings = {}
        keys = {"CMAKE_BUILD_TYPE", "USE_USD", "USE_PHYSX", "USE_CUDA_INTEROP", "IS_PYTHON_LIB"}
        for line in cache.read_text().splitlines():
            if ":" in line and "=" in line and line.split(":", 1)[0] in keys:
                settings[line.split(":", 1)[0]] = line.split("=", 1)[1]
        result["metadata"]["local_release_cache"] = settings
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        result["metadata"]["host_gpu_inventory"] = gpu.stdout.strip() or "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        result["metadata"]["host_gpu_inventory"] = "unavailable"
    if args.profiler:
        renderer.set_profiler_enabled(enabled=False)
        deadline = time.monotonic() + 2.0
        drain_frames = 0
        while drain_frames < 240 and time.monotonic() < deadline:
            if all(f.finalized for f in renderer.frame_profile_history()):
                break
            app.render_frame_once()
            drain_frames += 1
            time.sleep(0.001)
        result["metadata"]["gpu_drain_frames"] = drain_frames
        with tempfile.TemporaryDirectory(prefix="kang-profile-") as directory:
            raw = Path(directory) / "native.json"
            renderer.export_profile_json(path=str(raw), max_frames=args.samples)
            result["native_profile"] = json.loads(raw.read_text())
            result["metadata"]["pending_gpu_frames"] = sum(
                not f["finalized"] for f in result["native_profile"]["frames"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["wall_ms"]))


if __name__ == "__main__":
    main()
