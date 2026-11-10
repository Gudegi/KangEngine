"""Per-environment placement and playback for native motion references."""

import math
from typing import Any, Generic

import torch

from ...adapters._motion_transition import NativeMotionTransition
from ...utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_normalize,
    quat_wxyz_rotate,
    quat_wxyz_slerp,
    quat_wxyz_to_rotation_vector,
)
from ..motion_library import BackendSample, BodyMotionState, MotionLibrary
from .batch_search import MotionMatchBatch, MotionMatchingBatchSearch


class MotionMatchingBatchController(Generic[BackendSample]):
    """Align and smooth native motion targets for independent environments.

    Keep the shared search and library unchanged, with matching clips and order.
    Requires a floating-root PhysX, Newton, or MuJoCo adapter and body tracks.
    The task calls select() to choose targets, advance() to move time, and
    sample_robot_state() to read targets.
    Explicit DB reference tracks drive placement without filtering robot poses.

    Keep inputs on the shared library/search device; runtime calls do not move
    tensors between CPU and GPU. Reference tracks are uploaded once at creation.
    Without a track, root forward directions must project onto the ground plane.

    Joints and world body tracks are smoothed separately; bodies are not exact
    FK of the smoothed joints. The CPU player smooths local skeleton rotations.
    Joint targets are not clamped to limits. A zero transition_half_life disables
    smoothing; precompute_kinematics=True is required on the library.
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
        self.held = torch.zeros(num_envs, dtype=torch.bool, device=search.device)
        self._hold_enabled = False
        self._alignment_quat = root_pos.new_zeros((num_envs, 4))
        self._alignment_quat[:, 0] = 1
        self._translation = root_pos.new_zeros((num_envs, 3))
        self._frame_offsets = molib.source_buffers().frame_offsets
        self._reference_frames = self._prepare_reference_frames()
        self.reset()

    def _prepare_reference_frames(self):
        if not self.search.database.has_reference_track:
            return None

        # Prepare segment velocities on CPU once, then upload the shared tracks.
        tracks = []
        for idx, motion in enumerate(self.search.database.motions):
            pos = torch.tensor(
                self.search.database.reference_positions(idx), dtype=torch.float64
            )
            quat = torch.tensor(
                self.search.database.reference_quats_wxyz(idx), dtype=torch.float64
            )
            vel = torch.zeros_like(pos)
            ang_vel = torch.zeros_like(pos)
            vel[:-1] = torch.diff(pos, dim=0) * motion.fps()
            delta = quat_wxyz_multiply(quat[1:], quat_wxyz_conjugate(quat[:-1]))
            ang_vel[:-1] = quat_wxyz_to_rotation_vector(delta) * motion.fps()
            tracks.append(torch.cat((pos, quat, vel, ang_vel), dim=-1))
        return torch.cat(tracks).to(device=self.times.device, dtype=self.times.dtype)

    def _sample_reference(self, ids, times):
        ids, first, following, blend = self._motion_lib._resolve_frames(
            ids, times, loop=False
        )
        offsets = self._frame_offsets[ids]
        before = self._reference_frames[offsets + first]
        after = self._reference_frames[offsets + following]
        return (
            torch.lerp(before[..., :3], after[..., :3], blend[..., None]),
            quat_wxyz_slerp(before[..., 3:7], after[..., 3:7], blend),
            before[..., 7:10],
            before[..., 10:13],
        )

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
        World root_pos [N, 3] and root_quat_wxyz [N, 4] place the DB reference
        track, defaulting to zero and identity. Without a track they place the
        source root heading, retaining source height when root_pos is omitted.
        Supply tensors on the library device.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        self.motion_ids[env_ids] = motion_ids
        self.times[env_ids] = times
        if self._reference_frames is not None:
            source_pos, source_quat, _, _ = self._sample_reference(
                self.motion_ids[env_ids], self.times[env_ids]
            )
        else:
            state = self._motion_lib.sample_arti_state(
                self.motion_ids[env_ids], self.times[env_ids], loop=False
            )
            source_pos, source_quat = self._native_transition.root_pose(state)
        if root_pos is None:
            root_pos = torch.zeros_like(source_pos)
            if self._reference_frames is None:
                root_pos[..., self._up] = source_pos[..., self._up]
        if root_quat_wxyz is None:
            root_quat_wxyz = torch.zeros_like(source_quat)
            root_quat_wxyz[..., 0] = 1
        self._alignment_quat[env_ids], self._translation[env_ids] = self._placement(
            source_pos, source_quat, root_pos, root_quat_wxyz
        )
        self._ages[env_ids] = 0
        self.held[env_ids] = False
        self._native_transition.reset(env_ids)

    @torch.no_grad()
    def hold(self, env_ids: torch.Tensor | None = None) -> None:
        """Freeze the current output pose and zero root, joint, and body rates.

        held is a borrowed read-only [N] mask. Both playback and smoothing
        clocks stop, including future samples. select() preserves held rows;
        omit them from calls to select() to avoid their search work entirely.
        Environment selections are integer indices on the library device.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        self._hold_enabled = True
        self.held[env_ids] = True

    @torch.no_grad()
    def resume(self, env_ids: torch.Tensor | None = None) -> None:
        """Resume held rows by blending from their frozen pose and zero rates.

        A zero transition half-life resumes source velocity immediately.
        transition() can instead resume directly into a chosen clip.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        self._transition(
            env_ids,
            self.motion_ids[env_ids],
            self.times[env_ids],
            changed=self.held[env_ids].clone(),
        )

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
        Build queries with the database's feature layout, coordinates, and dtype,
        then call search.normalize(). NumPy scalar query helpers are not GPU builders.
        allowed_motion_mask follows MotionMatchingBatchSearch.search().
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
        changed = (match.row_ids >= 0) & ~continuation & ~self.held[env_ids]
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

        Align reference tracks across the switch, or root headings and planar
        positions without a track. Override placement as in reset().
        Held environments resume from their frozen pose and zero rates.
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
        delta = (~self.held) * dt if self._hold_enabled else dt
        next_time = torch.minimum(
            self.times + delta, self._motion_lib.durations[self.motion_ids]
        )
        self.times.copy_(
            torch.where(self.held, self.times, next_time)
            if self._hold_enabled
            else next_time
        )
        self._ages += delta

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

        The first state uses the library adapter's format: PhysXMotionState has
        split root fields and logical joint_pos/joint_vel; NewtonMotionState has
        joint_q (XYZW) and joint_qd; MuJoCoMotionState has qpos (WXYZ) and qvel.
        Newton root angular rates are world-space, ball rates parent-space;
        MuJoCo angular rates are local. Newton and MuJoCo support ball joints.
        The second state always uses world-space body poses and rates with WXYZ
        rotations. Fields such as body_pos_w have shape [N, bodies, 3].
        """
        env_ids = slice(None) if env_ids is None else env_ids
        return self._native_transition.to_states(
            self._sample_components(env_ids, future_offsets)
        )

    @torch.no_grad()
    def sample_reference_state(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the world reference frame used to build matching queries.

        Returns:
            (pos_w, quat_wxyz_w, vel_w, ang_vel_w), with shapes [N, 3],
            [N, 4], [N, 3], [N, 3]. Units are meters, seconds, and radians.
            Explicit tracks ignore pose smoothing and use right segment
            derivatives, zero at clip ends. Without a track, returns the
            smoothed output root position, heading, and root velocities.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        if self._reference_frames is None:
            state, _ = self._native_transition.to_states(
                self._sample_components(env_ids)
            )
            pos, quat = self._native_transition.root_pose(state)
            vel, ang_vel = self._native_transition.root_velocity(state)
            return pos, self._heading(quat), vel, ang_vel

        pos, quat, vel, ang_vel = self._sample_reference(
            self.motion_ids[env_ids], self.times[env_ids]
        )
        rotation = self._alignment_quat[env_ids]
        if self._hold_enabled:
            vel = vel.masked_fill(self.held[env_ids, None], 0)
            ang_vel = ang_vel.masked_fill(self.held[env_ids, None], 0)
        return (
            quat_wxyz_rotate(rotation, pos) + self._translation[env_ids],
            quat_wxyz_multiply(rotation, quat),
            quat_wxyz_rotate(rotation, vel),
            quat_wxyz_rotate(rotation, ang_vel),
        )

    def _sample_components(self, env_ids, future_offsets=None):
        ids, times = self.motion_ids[env_ids], self.times[env_ids]
        age = self._ages[env_ids]
        held = self.held[env_ids]
        rotation, translation = (
            self._alignment_quat[env_ids],
            self._translation[env_ids],
        )
        if future_offsets is not None:
            ids = ids[:, None]
            held = held[:, None]
            offsets = (
                torch.where(held, 0.0, future_offsets)
                if self._hold_enabled
                else future_offsets
            )
            times = times[:, None] + offsets
            age = age[:, None] + offsets
            rotation, translation = rotation[:, None], translation[:, None]
        state, bodies = self._motion_lib.sample_robot_state(ids, times, loop=False)
        components = self._native_transition.align(state, bodies, rotation, translation)
        if self._half_life > 0:
            components = self._native_transition.apply_offsets(components, env_ids, age)
        if self._hold_enabled:
            components = self._native_transition.zero_velocities(components, held)
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
        if self._reference_frames is None:
            source_quat, rotation = self._heading(source_quat), self._heading(rotation)
        alignment = quat_wxyz_multiply(rotation, quat_wxyz_conjugate(source_quat))
        if self._reference_frames is not None:
            alignment = quat_wxyz_normalize(alignment)
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
            if self._reference_frames is None:
                previous_pos, previous_quat = self._native_transition.aligned_root(
                    before
                )
            source, bodies = self._motion_lib.sample_robot_state(
                motion_ids, times, loop=False
            )
        elif self._reference_frames is None:
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
        if self._reference_frames is None:
            source_pos, source_quat = self._native_transition.root_pose(source)
        else:
            source_pos, source_quat, _, _ = self._sample_reference(motion_ids, times)
            if root_pos is None or root_quat_wxyz is None:
                previous_pos, previous_quat, _, _ = self.sample_reference_state(env_ids)
        if root_pos is None:
            root_pos = previous_pos.clone()
            if self._reference_frames is None:
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
        self.held[env_ids] = self.held[env_ids] & ~changed
        if self._half_life == 0:
            return

        after = self._native_transition.align(
            source, bodies, self._alignment_quat[env_ids], self._translation[env_ids]
        )
        self._native_transition.capture_offsets(before, after, env_ids, changed)
