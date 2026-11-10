"""Steer a batch of kinematic G1 robots in Newton's native ViewerGL.

Run from the repository root:
    PYTHONPATH=python python python/examples/animation/motion_matching_g1_newton.py

WASD moves relative to the camera, Shift runs, Space pauses, and R resets.
Mouse camera controls remain available; WASD camera translation is disabled.
MotionMatchingBatchController drives native joint coordinates through Newton FK.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter, sleep

import newton
import numpy as np
import torch
import warp as wp
from newton.viewer import ViewerGL

import kangengine as ke
from kangengine.adapters.newton import NewtonMotionAdapter
from kangengine.utils.batched_rotations import quat_wxyz_conjugate, quat_wxyz_rotate
from motion_matching_g1 import (
    CONFIG,
    DEFAULT_CHARACTER,
    DEFAULT_DATASET,
    POSITION_FILTER_WINDOW,
    ROTATION_FILTER_WINDOW,
    _build_motions,
    _load_mapper,
)


DT = 1 / 60
SEARCH_STEPS = 6
WALK_SPEED = 1.0
RUN_SPEED = 3.0
TRAJECTORY_HALF_LIFE = 0.2
TRANSITION_HALF_LIFE = 0.1
START_SEARCH_DELAY = 4 * TRANSITION_HALF_LIFE
MIN_STOP_TIME = 4 * TRANSITION_HALF_LIFE


class G1Batch:
    def __init__(self, database, *, num_envs=4, device="cpu"):
        self.database = database
        mapper = _load_mapper(DEFAULT_CHARACTER)
        adapter = NewtonMotionAdapter(mapper.layout)
        molib = ke.animation.MotionLibrary(
            database.motions, adapter=adapter, device=device
        )
        self.search = ke.animation.MotionMatchingBatchSearch(database, device=device)
        self.controller = ke.animation.MotionMatchingBatchController(
            self.search,
            molib,
            num_envs=num_envs,
            transition_half_life=TRANSITION_HALF_LIFE,
        )
        self.fsm = ke.utils.BatchFSM(
            # Stable state ID -> display label; each environment has its own state.
            {"idle": "Idle", "moving": "Moving", "stopping": "Stopping"},
            # Transition ID, source state, target state, signal supplied to step().
            [
                ke.utils.FSMTransition("start", "idle", "moving", "move_requested"),
                ke.utils.FSMTransition("stop", "moving", "stopping", "stop_requested"),
                # A new movement request wins if stopped is also true this step.
                ke.utils.FSMTransition(
                    "restart", "stopping", "moving", "move_requested", priority=1
                ),
                # No input requests stopping; stopped confirms it is safe to hold.
                ke.utils.FSMTransition("stand", "stopping", "idle", "stopped"),
            ],
            initial_state="idle",
            num_envs=num_envs,
        )

        self.idle_idx = self.fsm.state_idx("idle")
        self.moving_idx = self.fsm.state_idx("moving")
        self.stopping_idx = self.fsm.state_idx("stopping")
        clip_names = [motion.motion_name() for motion in database.motions]
        self.start_motion_mask = torch.tensor(
            ["_start_" in name for name in clip_names], device=device
        )
        self.stop_motion_mask = torch.tensor(
            ["_stop_" in name for name in clip_names], device=device
        )
        self.moving_motion_mask = torch.tensor(
            ["idle" not in name and "_stop_" not in name for name in clip_names],
            device=device,
        )
        # CPU scheduling metadata; copy selected match clocks only on state entry.
        self.clip_durations = torch.tensor(
            [(motion.num_frames() - 1) / motion.fps() for motion in database.motions]
        )
        self.search_delay = torch.zeros(num_envs)
        self.stop_remaining = torch.full((num_envs,), torch.inf)

        robot = newton.ModelBuilder()
        robot.add_mjcf(
            str(DEFAULT_CHARACTER),
            floating=True,
            ignore_names=["floor", "ground"],
            enable_self_collisions=False,
        )
        adapter.validate_model(robot.finalize(device=device))
        builder = newton.ModelBuilder()
        builder.replicate(robot, num_envs)
        builder.add_ground_plane()
        self.model = builder.finalize(device=device)
        self.state = self.model.state()

        names = list(database.motions[0].node_names())
        self.position_ids = [names.index(name) for name in CONFIG.position_joints]
        self.velocity_ids = [names.index(name) for name in CONFIG.velocity_joints]
        self.horizons = torch.tensor(CONFIG.trajectory_times, device=device)[
            None, :, None
        ]
        self.facing = torch.zeros(num_envs, device=device)
        self.reset()

    def reset(self):
        # Different idle phases exercise independent playback in the shared batch.
        times = torch.linspace(
            0, 0.5, self.controller.num_envs, device=self.search.device
        )
        self.controller.reset(times=times)
        self.controller.hold()
        self.fsm.reset()
        self.search_delay.zero_()
        self.stop_remaining.fill_(torch.inf)
        self.facing.zero_()
        self.steps = 0
        self.apply_pose()

    def make_queries(self, env_ids, desired_velocity):
        _, bodies = self.controller.sample_robot_state(env_ids)
        origin, rotation, current_velocity, _ = self.controller.sample_reference_state(
            env_ids
        )
        forward = quat_wxyz_rotate(rotation, rotation.new_tensor([1, 0, 0]))
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        inverse_heading = quat_wxyz_conjugate(rotation)
        origin[:, 2] = 0

        # Use the same filtered reference frame as the database features.
        positions = quat_wxyz_rotate(
            inverse_heading[:, None],
            bodies.body_pos_w[:, self.position_ids] - origin[:, None],
        )
        velocities = quat_wxyz_rotate(
            inverse_heading[:, None], bodies.body_vel_w[:, self.velocity_ids]
        )
        speed = torch.linalg.vector_norm(desired_velocity[:, :2], dim=-1)
        self.facing[env_ids] = torch.where(
            speed > 1e-6,
            torch.atan2(desired_velocity[:, 1], desired_velocity[:, 0]),
            self.facing[env_ids],
        )
        rate = math.log(2) / TRAJECTORY_HALF_LIFE
        decay = torch.exp(-rate * self.horizons)
        current_velocity[:, 2] = 0
        trajectory = (
            desired_velocity[:, None] * self.horizons
            + (current_velocity - desired_velocity)[:, None] * (1 - decay) / rate
        )
        angle_error = torch.atan2(
            torch.sin(self.facing[env_ids] - yaw),
            torch.cos(self.facing[env_ids] - yaw),
        )
        angles = yaw[:, None, None] + angle_error[:, None, None] * (1 - decay)
        directions = torch.cat(
            (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), -1
        )
        local_trajectory = quat_wxyz_rotate(inverse_heading[:, None], trajectory)[
            ..., :2
        ]
        local_directions = quat_wxyz_rotate(inverse_heading[:, None], directions)[
            ..., :2
        ]
        blocks = (positions, velocities, local_trajectory, local_directions)
        return self.search.normalize(
            torch.cat([value.flatten(1) for value in blocks], dim=-1)
        )

    @torch.no_grad()
    def step(self, desired_velocity):
        moving = desired_velocity[:, :2].norm(dim=-1) > 0.05
        was_idle = self.fsm.state_ids == self.idle_idx
        self.search_delay -= DT
        self.search_delay.clamp_min_(0)
        self.stop_remaining -= DT
        changed = self.fsm.step(
            DT,
            {
                "move_requested": moving,
                "stop_requested": ~moving,
                # Let the selected stop clip finish its braking and foot plant.
                "stopped": (self.stop_remaining <= 0)
                & (self.fsm.elapsed >= MIN_STOP_TIME),
            },
        )
        idle = self.fsm.state_ids == self.idle_idx
        starting = changed & (self.fsm.state_ids == self.moving_idx)
        stopping = changed & (self.fsm.state_ids == self.stopping_idx)
        if (changed & idle).any():
            self.controller.hold(
                (changed & idle).nonzero().flatten().to(self.search.device)
            )

        # Enter authored start/stop motions once. transition() blends directly
        # from a held pose, without resuming the old clip first.
        entering = starting | stopping
        if entering.any():
            cpu_ids = entering.nonzero().flatten()
            env_ids = cpu_ids.to(self.search.device)
            velocity = desired_velocity[cpu_ids].to(self.search.device)
            allowed = torch.where(
                stopping[cpu_ids, None].to(self.search.device),
                self.stop_motion_mask,
                torch.where(
                    was_idle[cpu_ids, None].to(self.search.device),
                    self.start_motion_mask,
                    self.moving_motion_mask,
                ),
            )
            match = self.search.search(
                self.make_queries(env_ids, velocity), allowed_motion_mask=allowed
            )
            self.controller.transition(
                env_ids, motion_ids=match.motion_ids, times=match.times
            )
            self.search_delay[starting] = START_SEARCH_DELAY
            if stopping.any():
                remaining = (
                    self.clip_durations[match.motion_ids.cpu()] - match.times.cpu()
                )
                stop_rows = stopping[cpu_ids]
                self.stop_remaining[cpu_ids[stop_rows]] = remaining[stop_rows]

        # Preserve initial acceleration before periodic searches resume.
        search_rows = (
            (self.fsm.state_ids == self.moving_idx)
            & (self.search_delay <= 0)
            & ~changed
        )
        if self.steps % SEARCH_STEPS == 0 and search_rows.any():
            cpu_ids = search_rows.nonzero().flatten()
            env_ids = cpu_ids.to(self.search.device)
            self.controller.select(
                self.make_queries(
                    env_ids, desired_velocity[cpu_ids].to(self.search.device)
                ),
                env_ids=env_ids,
                continuation_bias=0.01,
                allowed_motion_mask=self.moving_motion_mask,
            )
        self.controller.advance(DT)
        self.apply_pose()
        self.steps += 1

    def apply_pose(self):
        self.native, _ = self.controller.sample_robot_state()
        self.model.joint_q.assign(wp.from_torch(self.native.joint_q.flatten()))
        self.model.joint_qd.assign(wp.from_torch(self.native.joint_qd.flatten()))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)


class G1Viewer(ViewerGL):
    def _update_camera(self, dt):
        # Reserve keyboard movement for the robot; mouse navigation stays native.
        pass


def keyboard_velocity(viewer):
    forward = np.array(viewer.camera.get_front(), dtype=np.float32)
    forward[2] = 0
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    right = np.cross(forward, [0, 0, 1])
    velocity = (
        int(viewer.is_key_down("w")) - int(viewer.is_key_down("s"))
    ) * forward + (int(viewer.is_key_down("d")) - int(viewer.is_key_down("a"))) * right
    speed = RUN_SPEED if viewer.is_key_down("shift") else WALK_SPEED
    return velocity * (speed / max(float(np.linalg.norm(velocity)), 1.0))


def run_viewer(batch, *, headless=False, frames=0):
    viewer = G1Viewer(width=1280, height=800, headless=headless)
    viewer.set_model(batch.model)
    viewer.set_world_offsets((0, 2.0, 0))
    viewer.set_camera(wp.vec3(-5, -6, 3), pitch=-15, yaw=45)
    reset_down = False
    frame = 0
    try:
        while viewer.is_running() and (frames <= 0 or frame < frames):
            start = perf_counter()
            previous_root = batch.native.joint_q[0, :3].clone()
            reset = viewer.is_key_down("r")
            if reset and not reset_down:
                batch.reset()
            reset_down = reset
            if viewer.should_step():
                velocity = torch.as_tensor(
                    keyboard_velocity(viewer),
                    dtype=torch.float32,
                ).expand(batch.controller.num_envs, 3)
                batch.step(velocity)
            delta = (batch.native.joint_q[0, :3] - previous_root).cpu().tolist()
            viewer.set_camera(
                viewer.camera.pos + wp.vec3(delta[0], delta[1], 0),
                pitch=viewer.camera.pitch,
                yaw=viewer.camera.yaw,
            )
            viewer.begin_frame(batch.steps * DT)
            viewer.log_state(batch.state)
            viewer.end_frame()
            frame += 1
            if not headless:
                sleep(max(0, DT - (perf_counter() - start)))
    finally:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    database = ke.animation.MotionMatchingDatabase.from_motions(
        _build_motions(args),
        config=CONFIG,
        feature_builder=ke.animation.WalkingFeatures(
            position_filter_window=POSITION_FILTER_WINDOW,
            rotation_filter_window=ROTATION_FILTER_WINDOW,
        ),
    )
    print(f"Database: {len(database.motions)} clips, {database.num_frames} candidates")
    batch = G1Batch(database, num_envs=args.num_envs, device=args.device)
    print("Newton ViewerGL | WASD: move | Shift: run | Space: pause | R: reset")
    run_viewer(batch, headless=args.headless, frames=args.frames)


if __name__ == "__main__":
    main()
