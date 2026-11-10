"""Unrestricted G1 motion matching; hold only after the output stops moving.

Run from the repository root:
    python python/examples/animation/motion_matching_g1_newton_idle.py

Shares the Newton viewer and query features with motion_matching_g1_newton.py.
WASD moves, Shift runs, Space pauses, and R resets.
"""

import argparse
from pathlib import Path

import torch
import warp as wp

import kangengine as ke
from kangengine.utils.batched_rotations import quat_wxyz_rotate, quat_xyzw_to_wxyz
from motion_matching_g1 import CONTACT_OFFSETS
from motion_matching_g1_newton import (
    CONFIG,
    DEFAULT_DATASET,
    DT,
    POSITION_FILTER_WINDOW,
    ROTATION_FILTER_WINDOW,
    SEARCH_STEPS,
    G1Batch,
    _build_motions,
    run_viewer,
)


# Mocap has residual motion: require a sustained quiet pose near the floor.
ROOT_STOP_SPEED = 0.08
ROOT_STOP_ANG_SPEED = 0.3
FOOT_STOP_SPEED = 0.12
JOINT_STOP_SPEED = 0.5
FOOT_HEIGHT_TOLERANCE = 0.04
STOP_DURATION = 0.2


class G1IdleHoldBatch(G1Batch):
    def reset(self):
        super().reset()
        body_count = self.model.body_count // self.controller.num_envs
        names = self.model.body_label[:body_count]
        self.foot_ids = [
            next(idx for idx, name in enumerate(names) if name.endswith(foot))
            for foot in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        self.foot_offsets = torch.as_tensor(
            CONTACT_OFFSETS, dtype=torch.float32, device=self.search.device
        ).reshape(2, 2, 3)
        self.previous_feet = self._get_feet_positions()
        self.previous_root = self.native.joint_q[:, :3].clone()
        # The reset clip is standing on a flat floor; retain that floor height.
        self.floor_height = self.previous_feet[..., 2].amin(dim=(1, 2))
        self.quiet_time = torch.zeros_like(self.controller.times)

    def _get_feet_positions(self):
        """World heel/toe positions, shaped [N, 2 feet, 2 points, 3]."""
        # Use rendered Newton FK, not separately interpolated body tracks.
        poses = wp.to_torch(self.state.body_q).reshape(self.controller.num_envs, -1, 7)[
            :, self.foot_ids
        ]
        rotation = quat_xyzw_to_wxyz(poses[..., 3:7])
        return poses[..., None, :3] + quat_wxyz_rotate(
            rotation[..., None, :], self.foot_offsets
        )

    def _stopped(self, moving):
        feet = self._get_feet_positions()
        root = self.native.joint_q[:, :3]
        root_speed = torch.linalg.vector_norm(root - self.previous_root, dim=-1) / DT
        foot_speed = (
            torch.linalg.vector_norm(feet - self.previous_feet, dim=-1).amax(dim=(1, 2))
            / DT
        )
        feet_down = (
            feet[..., 2] <= self.floor_height[:, None, None] + FOOT_HEIGHT_TOLERANCE
        ).all(dim=(1, 2))
        quiet = (
            ~moving.to(self.search.device)
            & (root_speed < ROOT_STOP_SPEED)
            & (foot_speed < FOOT_STOP_SPEED)
            & feet_down
            & (self.native.joint_qd[:, 3:6].norm(dim=-1) < ROOT_STOP_ANG_SPEED)
            & (self.native.joint_qd[:, 6:].abs().amax(dim=-1) < JOINT_STOP_SPEED)
        )
        self.quiet_time.copy_(torch.where(quiet, self.quiet_time + DT, 0.0))
        self.previous_feet = feet
        self.previous_root = root.clone()
        # This interactive viewer schedules on CPU; transfer only the [N] signal.
        return (self.quiet_time >= STOP_DURATION).cpu()

    @torch.no_grad()
    def step(self, desired_velocity):
        moving = desired_velocity[:, :2].norm(dim=-1) > 0.05
        was_idle = self.fsm.state_ids == self.idle_idx
        changed = self.fsm.step(
            DT,
            {
                "move_requested": moving,
                "stop_requested": ~moving,
                "stopped": self._stopped(moving),
            },
        )
        idle = self.fsm.state_ids == self.idle_idx
        entering_idle = changed & idle
        if entering_idle.any():
            self.controller.hold(
                entering_idle.nonzero().flatten().to(self.search.device)
            )

        # Resume directly into the best match, from the held pose and zero rates.
        leaving_idle = changed & was_idle
        if leaving_idle.any():
            cpu_ids = leaving_idle.nonzero().flatten()
            env_ids = cpu_ids.to(self.search.device)
            match = self.search.search(
                self.make_queries(
                    env_ids, desired_velocity[cpu_ids].to(self.search.device)
                )
            )
            self.controller.transition(
                env_ids, motion_ids=match.motion_ids, times=match.times
            )

        # Moving and Stopping share the same unrestricted search policy.
        search_rows = ~idle & ~leaving_idle
        if self.steps % SEARCH_STEPS == 0 and search_rows.any():
            cpu_ids = search_rows.nonzero().flatten()
            env_ids = cpu_ids.to(self.search.device)
            self.controller.select(
                self.make_queries(
                    env_ids, desired_velocity[cpu_ids].to(self.search.device)
                ),
                env_ids=env_ids,
                continuation_bias=0.01,
            )
        self.controller.advance(DT)
        self.apply_pose()
        self.steps += 1


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
    batch = G1IdleHoldBatch(database, num_envs=args.num_envs, device=args.device)
    print("Unrestricted MM | Hold after stable feet and low output velocities")
    run_viewer(batch, headless=args.headless, frames=args.frames)


if __name__ == "__main__":
    main()
