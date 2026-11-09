"""Per-environment placement and playback for native motion references."""

import math
from typing import Any, Generic

import torch

from ...adapters._motion_transition import NativeMotionTransition
from ...utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_rotate,
)
from ..motion_library import BackendSample, BodyMotionState, MotionLibrary
from .batch_search import MotionMatchBatch, MotionMatchingBatchSearch


class MotionMatchingBatchController(Generic[BackendSample]):
    """Align and smooth native motion targets for independent environments.

    Keep the shared search and library unchanged, with matching clips and order.
    Requires a floating-root PhysX, Newton, or MuJoCo adapter and body tracks.
    The task calls select() to choose targets, advance() to move time, and
    sample_robot_state() to read targets.

    Joints and world body tracks are smoothed separately; bodies are not exact
    FK of the smoothed joints. The CPU player smooths local skeleton rotations.
    """

    def __init__(
        self,
        search: MotionMatchingBatchSearch,
        molib: MotionLibrary[BackendSample],
        *,
        num_envs: int,
        transition_half_life: float = 0.1,
    ) -> None:
        if not math.isfinite(transition_half_life) or transition_half_life < 0:
            raise ValueError("transition_half_life must be finite and nonnegative")
        if search.database.has_reference_track:
            raise NotImplementedError("batch placement currently uses source roots")
        buffers = molib.body_buffers()
        native = molib.adapted_buffers()
        if buffers.body_vel_w is None or buffers.body_ang_vel_w is None:
            raise ValueError("batch playback requires precompute_kinematics=True")
        if molib.device != search.device:
            raise ValueError("motion library and search backend must share a device")
        # Check clip clocks once, before any per-step CUDA work.
        counts = [motion.num_frames() for motion in search.database.motions]
        fps = [motion.fps() for motion in search.database.motions]
        if molib.frame_counts.tolist() != counts or not torch.allclose(
            molib.fps.cpu(), torch.tensor(fps, dtype=molib.fps.dtype)
        ):
            raise ValueError("library clips must match database order and timing")

        self.search = search
        self._motion_lib = molib
        self.num_envs = num_envs
        self._half_life = transition_half_life
        self._decay_rate = (
            2 * math.log(2) / transition_half_life if transition_half_life > 0 else 0
        )
        create_transition = getattr(molib.adapter, "_create_transition", None)
        if create_transition is None:
            raise TypeError(
                "batch playback requires a PhysX, Newton, or MuJoCo adapter"
            )
        self._native_transition: NativeMotionTransition[BackendSample, Any] = (
            create_transition(native, buffers, num_envs, self._decay_rate)
        )
        root_pos, _ = self._native_transition.root_pose(native)
        self._up = "XYZ".index(search.database.config.up_axis)
        self._forward = root_pos.new_zeros(3)
        self._forward["XYZ".index(search.database.config.forward_axis)] = 1
        self._up_vector = root_pos.new_zeros(3)
        self._up_vector[self._up] = 1
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=search.device)
        self.times = root_pos.new_zeros(num_envs)
        self._ages = torch.zeros_like(self.times)
        self._alignment_quat = root_pos.new_zeros((num_envs, 4))
        self._alignment_quat[:, 0] = 1
        self._translation = root_pos.new_zeros((num_envs, 3))
        self.reset()

    @property
    def finished(self) -> torch.Tensor:
        """Boolean [N] mask; select a new match before advancing past clip ends."""
        return self.times >= self._motion_lib.durations[self.motion_ids]

    @torch.no_grad()
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        motion_ids: torch.Tensor | int = 0,
        times: torch.Tensor | float = 0.0,
        root_pos: torch.Tensor | None = None,
        root_quat_wxyz: torch.Tensor | None = None,
    ) -> None:
        """Reset selected environments without blending.

        IDs/times are scalars or [N], where N is the selected environment count.
        World root_pos [N, 3] overrides position; default is planar zero with
        source height. root_quat_wxyz [N, 4] sets heading only; default is identity.
        Supply tensors on the library device.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        self.motion_ids[env_ids] = motion_ids
        self.times[env_ids] = times
        state = self._motion_lib.sample_arti_state(
            self.motion_ids[env_ids], self.times[env_ids], loop=False
        )
        source_pos, source_quat = self._native_transition.root_pose(state)
        if root_pos is None:
            root_pos = torch.zeros_like(source_pos)
            root_pos[..., self._up] = source_pos[..., self._up]
        if root_quat_wxyz is None:
            root_quat_wxyz = torch.zeros_like(source_quat)
            root_quat_wxyz[..., 0] = 1
        self._alignment_quat[env_ids], self._translation[env_ids] = self._placement(
            source_pos, source_quat, root_pos, root_quat_wxyz
        )
        self._ages[env_ids] = 0
        self._native_transition.reset(env_ids)

    @torch.no_grad()
    def select(
        self,
        queries: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        continuation_bias: float = 0.01,
        allowed_motion_mask: torch.Tensor | None = None,
    ) -> MotionMatchBatch:
        """Search [selected N, D] queries and transition changed matches.

        A nearest-frame continuation retains subframe time and existing smoothing.
        Rows with no eligible match retain their current playback state.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        ids, times = self.motion_ids[env_ids], self.times[env_ids]
        match = self.search.search(
            queries,
            current_motion_ids=ids,
            current_times=times,
            continuation_bias=continuation_bias,
            allowed_motion_mask=allowed_motion_mask,
        )
        continuation = (
            (match.motion_ids == ids)
            & ((match.times - times).abs() <= 0.5 / self.search.fps[ids] + 1e-7)
            & (times < self._motion_lib.durations[ids])
        )
        changed = (match.row_ids >= 0) & ~continuation
        # Fixed-shape masking avoids CUDA nonzero() and host synchronization.
        self._transition(
            env_ids,
            torch.where(changed, match.motion_ids, ids),
            torch.where(changed, match.times, times),
            changed=changed,
        )
        return match

    @torch.no_grad()
    def transition(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        motion_ids: torch.Tensor | int,
        times: torch.Tensor | float,
        root_pos: torch.Tensor | None = None,
        root_quat_wxyz: torch.Tensor | None = None,
    ) -> None:
        """Switch clips with continuous pose and velocity when smoothing is enabled.

        Align to the current reference's planar root and heading, keeping the
        new source height. Override root placement as in reset() to follow a robot.
        A zero half-life switches immediately without smoothing.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        changed = torch.ones_like(self.motion_ids[env_ids], dtype=torch.bool)
        self._transition(
            env_ids,
            motion_ids,
            times,
            changed=changed,
            root_pos=root_pos,
            root_quat_wxyz=root_quat_wxyz,
        )

    @torch.no_grad()
    def advance(self, dt: float) -> None:
        """Advance all environments by nonnegative seconds; clamp at clip ends.

        finished marks environments whose clips have ended; select new clips explicitly.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        self.times.copy_(
            torch.minimum(self.times + dt, self._motion_lib.durations[self.motion_ids])
        )
        self._ages += dt

    @torch.no_grad()
    def sample_robot_state(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        future_offsets: torch.Tensor | None = None,
    ) -> tuple[BackendSample, BodyMotionState]:
        """Read independent target snapshots without changing playback.

        Returns [N, ...] for selected environments. Nonnegative future_offsets [K]
        in seconds give [N, K, ...], continuing current clips and smoothing.
        Future times clamp at clip ends; future matches are not predicted.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        return self._native_transition.to_states(
            self._sample_components(env_ids, future_offsets)
        )

    def _sample_components(self, env_ids, future_offsets=None):
        ids, times = self.motion_ids[env_ids], self.times[env_ids]
        age = self._ages[env_ids]
        rotation, translation = (
            self._alignment_quat[env_ids],
            self._translation[env_ids],
        )
        if future_offsets is not None:
            ids = ids[:, None]
            times = times[:, None] + future_offsets
            age = age[:, None] + future_offsets
            rotation, translation = rotation[:, None], translation[:, None]
        state, bodies = self._motion_lib.sample_robot_state(ids, times, loop=False)
        components = self._native_transition.align(state, bodies, rotation, translation)
        if self._half_life > 0:
            components = self._native_transition.apply_offsets(components, env_ids, age)
        return components

    def _heading(self, rotation):
        direction = quat_wxyz_rotate(rotation, self._forward)
        sine = (
            torch.cross(self._forward.expand_as(direction), direction, dim=-1)
            * self._up_vector
        ).sum(-1)
        cosine = (direction * self._forward).sum(-1)
        half = 0.5 * torch.atan2(sine, cosine).unsqueeze(-1)
        return torch.cat((torch.cos(half), torch.sin(half) * self._up_vector), -1)

    def _placement(self, source_pos, source_quat, position, rotation):
        source_heading = self._heading(source_quat)
        alignment = quat_wxyz_multiply(
            self._heading(rotation), quat_wxyz_conjugate(source_heading)
        )
        return alignment, position - quat_wxyz_rotate(alignment, source_pos)

    def _transition(
        self,
        env_ids,
        motion_ids,
        times,
        *,
        changed,
        root_pos=None,
        root_quat_wxyz=None,
    ):
        motion_ids = torch.as_tensor(
            motion_ids, dtype=torch.long, device=self.search.device
        ).expand_as(self.motion_ids[env_ids])
        times = torch.as_tensor(
            times, dtype=self.times.dtype, device=self.search.device
        ).expand_as(self.times[env_ids])
        if self._half_life > 0:
            before = self._sample_components(env_ids)
            previous_pos, previous_quat = self._native_transition.aligned_root(before)
            source, bodies = self._motion_lib.sample_robot_state(
                motion_ids, times, loop=False
            )
        else:
            # Raw transitions need root placement, not body sampling or offsets.
            if root_pos is None or root_quat_wxyz is None:
                previous = self._motion_lib.sample_arti_state(
                    self.motion_ids[env_ids], self.times[env_ids], loop=False
                )
                previous_pos, previous_quat = self._native_transition.root_pose(
                    previous
                )
                previous_pos = (
                    quat_wxyz_rotate(self._alignment_quat[env_ids], previous_pos)
                    + self._translation[env_ids]
                )
                previous_quat = quat_wxyz_multiply(
                    self._alignment_quat[env_ids],
                    previous_quat,
                )
            source = self._motion_lib.sample_arti_state(motion_ids, times, loop=False)
        source_pos, source_quat = self._native_transition.root_pose(source)
        if root_pos is None:
            root_pos = previous_pos.clone()
            root_pos[..., self._up] = source_pos[..., self._up]
        if root_quat_wxyz is None:
            root_quat_wxyz = previous_quat

        # Rebase only changed environments, including interrupted transitions.
        rotation, translation = self._placement(
            source_pos, source_quat, root_pos, root_quat_wxyz
        )
        self._alignment_quat[env_ids] = torch.where(
            changed[:, None], rotation, self._alignment_quat[env_ids]
        )
        self._translation[env_ids] = torch.where(
            changed[:, None], translation, self._translation[env_ids]
        )
        self.motion_ids[env_ids] = motion_ids
        self.times[env_ids] = times
        self._ages[env_ids] = torch.where(changed, 0.0, self._ages[env_ids])
        if self._half_life == 0:
            return

        after = self._native_transition.align(
            source, bodies, self._alignment_quat[env_ids], self._translation[env_ids]
        )
        self._native_transition.capture_offsets(before, after, env_ids, changed)
