"""Stress fixed-update catch-up behavior with a PhysX GPU simulation.

The app deliberately stalls ``preRender()`` so the next rendered frame receives
a large wall-clock delta. Physics still advances only from ``fixedUpdate()``.

Example:

    python python/examples/smoke/fixed_update_gpu_stress.py \
        --num-envs 64 --frames 12 --stall-ms 200 --max-catch-up-steps 8
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import kangengine as ke
import torch


def asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


class FixedUpdateGpuStressApp(ke.App):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.render_frames = 0
        self.fixed_updates = 0
        self._previous_fixed_updates = 0
        self.fixed_updates_per_frame = []
        self.frame_deltas = []
        self.final_sim_time = 0.0
        self.final_z = 0.0
        self._gpu_ready = False

    def setup(self):
        self.set_vsync(False)
        self.timing = self.configure_timing(
            ke.SimulationTimingConfig(
                render_hz=0.0,
                physics_hz=self.args.physics_hz,
                fixed_update_hz=self.args.fixed_hz,
                max_catch_up_steps=self.args.max_catch_up_steps,
            )
        )

        self.world = ke.sim.KangSimWorld(
            num_envs=self.args.num_envs,
            sim_device=f"cuda:{self.args.cuda_device}",
            sim_dt=self.timing.physics_dt,
            add_ground=True,
        )
        ball_data = self.world.load_mjcf(asset_path("objects", "ball.xml"))
        cloner = ke.sim.GridCloner(
            self.world,
            spacing=(1.0, 1.0),
        )
        self.balls = cloner.add_rigid(
            ball_data,
            obj_id=0,
            name="ball",
            density=60.0,
            initial_root_pos=(0.0, 0.0, 2.0),
        )
        self.world.init_gpu_system(cuda_device_id=self.args.cuda_device)
        self._gpu_ready = True
        self.world.step(substeps=0, refresh=False, apply_commands=False)

    def fixedUpdate(self, fixed_dt):
        self.world.advance(fixed_dt, refresh=False)
        self.fixed_updates += 1

    def preRender(self):
        updates_this_frame = self.fixed_updates - self._previous_fixed_updates
        self._previous_fixed_updates = self.fixed_updates
        self.fixed_updates_per_frame.append(updates_this_frame)
        self.frame_deltas.append(float(self.get_delta_time()))
        self.render_frames += 1

        if self.render_frames >= self.args.frames:
            self.request_close()
            return
        time.sleep(self.args.stall_ms / 1000.0)

    def cleanup(self):
        if not hasattr(self, "world"):
            return
        self.final_sim_time = float(self.world.sim_time)
        if self._gpu_ready:
            rigid_state = self.world.get_gpu_rigid_data()
            torch.cuda.synchronize(self.args.cuda_device)
            row = self.world.rigid_gpu_row(0, 0)
            self.final_z = float(rigid_state[row, 2].item())
        self.world.release()

    def validate(self):
        max_updates = max(self.fixed_updates_per_frame, default=0)
        if max_updates > self.args.max_catch_up_steps:
            raise AssertionError(
                f"catch-up limit exceeded: {max_updates} > "
                f"{self.args.max_catch_up_steps}"
            )
        if self.args.expect_dropped_time and self.get_dropped_wall_time() <= 0.0:
            raise AssertionError("expected delayed frames to drop wall time")
        requested_sim_time = self.fixed_updates / self.args.fixed_hz
        physics_step_ratio = requested_sim_time * self.args.physics_hz
        nearest_steps = round(physics_step_ratio)
        if math.isclose(
            physics_step_ratio,
            nearest_steps,
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        ):
            physics_steps = int(nearest_steps)
        else:
            physics_steps = math.floor(physics_step_ratio)
        expected_sim_time = physics_steps / self.args.physics_hz
        if abs(self.final_sim_time - expected_sim_time) > 1.0e-5:
            raise AssertionError(
                f"simulation time mismatch: {self.final_sim_time} != "
                f"{expected_sim_time}"
            )

    def report(self):
        print("Fixed-update GPU stress")
        print(f"  device             : cuda:{self.args.cuda_device}")
        print(f"  environments       : {self.args.num_envs}")
        print(f"  render frames      : {self.render_frames}")
        print(f"  artificial stall   : {self.args.stall_ms:.1f} ms/frame")
        print(f"  fixed update rate  : {self.args.fixed_hz:.1f} Hz")
        print(f"  physics rate       : {self.args.physics_hz:.1f} Hz")
        print(f"  catch-up limit     : {self.args.max_catch_up_steps}")
        print(f"  fixed updates      : {self.fixed_updates}")
        print(f"  updates per frame  : {self.fixed_updates_per_frame}")
        print(f"  max frame delta    : {max(self.frame_deltas, default=0.0):.6f} s")
        print(f"  dropped wall time  : {self.get_dropped_wall_time():.6f} s")
        print(f"  simulation time    : {self.final_sim_time:.6f} s")
        print(f"  first ball z       : {self.final_z:.6f}")
        print("PASS: fixed-update catch-up stayed bounded")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--stall-ms", type=float, default=200.0)
    parser.add_argument("--fixed-hz", type=float, default=60.0)
    parser.add_argument("--physics-hz", type=float, default=120.0)
    parser.add_argument("--max-catch-up-steps", type=int, default=8)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--expect-dropped-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.stall_ms < 0.0:
        parser.error("--stall-ms must be non-negative")
    if args.fixed_hz <= 0.0:
        parser.error("--fixed-hz must be positive")
    if args.physics_hz <= 0.0:
        parser.error("--physics-hz must be positive")
    if args.max_catch_up_steps < 1:
        parser.error("--max-catch-up-steps must be positive")
    return args


def main():
    args = parse_args()
    app = FixedUpdateGpuStressApp(args)
    app.initialize(64, 64, True, ke.UpAxis.Z, headless=True)
    app.start()
    app.validate()
    app.report()


if __name__ == "__main__":
    main()
