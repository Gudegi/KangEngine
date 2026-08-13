"""Measure Newton-to-KangEngine CPU viewer update and render costs."""

from __future__ import annotations

import argparse
import json
import platform
from statistics import mean
from time import perf_counter

from kangengine.adapters.newton import NewtonViewer


def _build_model(world_count: int):
    import newton

    newton.use_coord_layout_targets = True
    world = newton.ModelBuilder()
    body = world.add_body(label="benchmark_box")
    world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)

    builder = newton.ModelBuilder()
    builder.replicate(world, world_count=world_count)
    return builder.finalize()


def benchmark(world_count: int, frames: int, warmup: int, width: int, height: int):
    model = _build_model(world_count)
    state = model.state()
    viewer = NewtonViewer(width=width, height=height, headless=True)
    try:
        viewer.set_model(model)
        viewer.set_world_offsets((2.0, 2.0, 0.0))
        for frame in range(warmup):
            viewer.begin_frame(float(frame))
            viewer.log_state(state)
            viewer.end_frame()

        update_times = []
        render_times = []
        for frame in range(frames):
            viewer.begin_frame(float(frame))
            start = perf_counter()
            viewer.log_state(state)
            update_end = perf_counter()
            viewer.end_frame()
            render_end = perf_counter()
            update_times.append((update_end - start) * 1_000.0)
            render_times.append((render_end - update_end) * 1_000.0)

        return {
            "worlds": world_count,
            "instances": world_count,
            "frames": frames,
            "update_ms": mean(update_times),
            "render_ms": mean(render_times),
            "total_ms": mean(update_times) + mean(render_times),
        }
    finally:
        viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="1,64,1024")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    world_counts = [int(value) for value in args.worlds.split(",")]
    if any(count < 1 for count in world_counts):
        raise ValueError("world counts must be positive")
    if args.frames < 1 or args.warmup < 0:
        raise ValueError("frames must be positive and warmup cannot be negative")

    results = [
        benchmark(count, args.frames, args.warmup, args.width, args.height)
        for count in world_counts
    ]
    print(
        json.dumps(
            {
                "device": "cpu",
                "platform": platform.platform(),
                "resolution": [args.width, args.height],
                "warmup": args.warmup,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
