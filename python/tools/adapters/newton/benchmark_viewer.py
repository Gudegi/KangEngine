"""Compare KangEngine NewtonViewer and Newton ViewerGL update costs.

This microbenchmark does not step a Newton solver. By default it measures two
paths in KangEngine's ``NewtonViewer``:

- CPU: Newton CPU transforms -> NumPy -> CPU instance upload -> OpenGL
- CUDA: Newton CUDA transforms -> Torch CUDA matrices -> CUDA/OpenGL D2D
  ExternalBuffer copy -> OpenGL

RTX 4090 update results recorded on 2026-08-15 at 1280x720, with 10 warmup
and 30 measured frames (milliseconds, mean):

Worlds | KE CPU | KE CUDA | ViewerGL CPU | ViewerGL CUDA
------ | ------ | ------- | ------------ | -------------
1      | 0.249  | 0.845   | 0.138        | 0.479
64     | 0.315  | 0.610   | 0.156        | 0.274
1024   | 0.473  | 0.767   | 0.160        | 0.284
4096   | 0.789  | 0.716   | 0.294        | 0.610
16384  | 2.317  | 0.653   | 1.002        | 1.138

``update`` measures steady-state ``viewer.log_state(state)``. KangEngine CUDA
overtook KangEngine CPU at 4096 instances and, at 16384, took 0.653 ms versus
ViewerGL CUDA's 1.138 ms (about 1.74x faster). Nsight capture at 16384 showed
that ViewerGL performed 120 D2H copies and 30 context synchronizations over 30
frames; KangEngine's corresponding steady-state capture used one D2D copy per
frame with no D2H or device-wide synchronization.

Use ``--viewer newton`` to run the same model/state/frame loop with Newton's
original ``ViewerGL``. Update costs are the direct comparison. Do not compare
the reported headless render/total values between viewers: ViewerGL's headless
``end_frame()`` mostly measures asynchronous GL submission, while KangEngine's
frame path includes different synchronization and framebuffer work. A renderer
throughput comparison requires GPU timestamp queries and matched render
features.
"""

from __future__ import annotations

import argparse
import json
import platform
from statistics import mean
from time import perf_counter

from kangengine.adapters.newton import NewtonViewer


def _build_model(world_count: int, device: str):
    import newton

    newton.use_coord_layout_targets = True
    world = newton.ModelBuilder()
    body = world.add_body(label="benchmark_box")
    world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)

    builder = newton.ModelBuilder()
    builder.replicate(world, world_count=world_count)
    return builder.finalize(device=device)


def benchmark(
    world_count: int,
    frames: int,
    warmup: int,
    width: int,
    height: int,
    device: str,
    cuda_profile_range: bool,
    viewer_backend: str,
):
    import newton

    model = _build_model(world_count, device)
    state = model.state()
    if viewer_backend == "kangengine":
        viewer = NewtonViewer(
            width=width,
            height=height,
            headless=True,
            allow_cuda_readback=False,
        )
    else:
        viewer = newton.viewer.ViewerGL(
            width=width,
            height=height,
            vsync=False,
            headless=True,
        )
    try:
        setup_start = perf_counter()
        viewer.set_model(model)
        viewer.set_world_offsets((2.0, 2.0, 0.0))
        setup_ms = (perf_counter() - setup_start) * 1_000.0

        registration_start = perf_counter()
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        registration_update_end = perf_counter()
        viewer.end_frame()
        registration_end = perf_counter()
        for frame in range(warmup):
            viewer.begin_frame(float(frame))
            viewer.log_state(state)
            viewer.end_frame()

        update_times = []
        render_times = []
        profiler_api = None
        if cuda_profile_range:
            import torch

            if not str(state.body_q.device).startswith("cuda"):
                raise ValueError("--cuda-profile-range requires a CUDA model")
            profiler_api = torch.cuda.cudart()
            profiler_api.cudaProfilerStart()
        try:
            for frame in range(frames):
                viewer.begin_frame(float(frame))
                start = perf_counter()
                viewer.log_state(state)
                update_end = perf_counter()
                viewer.end_frame()
                render_end = perf_counter()
                update_times.append((update_end - start) * 1_000.0)
                render_times.append((render_end - update_end) * 1_000.0)
        finally:
            if profiler_api is not None:
                profiler_api.cudaProfilerStop()

        return {
            "viewer": viewer_backend,
            "requested_device": device,
            "model_device": str(model.device),
            "transform_device": str(state.body_q.device),
            "worlds": world_count,
            "instances": world_count,
            "frames": frames,
            "setup_ms": setup_ms,
            "registration_update_ms": (registration_update_end - registration_start)
            * 1_000.0,
            "registration_render_ms": (registration_end - registration_update_end)
            * 1_000.0,
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
    parser.add_argument(
        "--viewer",
        choices=("kangengine", "newton"),
        default="kangengine",
        help="Viewer backend to benchmark",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Newton/Warp model device, for example cpu or cuda:0",
    )
    parser.add_argument(
        "--cuda-profile-range",
        action="store_true",
        help="Bracket steady-state frames with cudaProfilerStart/Stop",
    )
    args = parser.parse_args()

    world_counts = [int(value) for value in args.worlds.split(",")]
    if any(count < 1 for count in world_counts):
        raise ValueError("world counts must be positive")
    if args.frames < 1 or args.warmup < 0:
        raise ValueError("frames must be positive and warmup cannot be negative")

    results = [
        benchmark(
            count,
            args.frames,
            args.warmup,
            args.width,
            args.height,
            args.device,
            args.cuda_profile_range,
            args.viewer,
        )
        for count in world_counts
    ]
    print(
        json.dumps(
            {
                "viewer": args.viewer,
                "requested_device": args.device,
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
