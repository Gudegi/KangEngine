"""Shared inertial-offset math for native motion transitions."""

from typing import Protocol, TypeVar

import torch

from ..animation.articulation_motion import ArticulationCoordinateType
from ..animation.motion_library import BackendSample, BodyMotionState
from ..utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_from_angle_axis,
    quat_wxyz_multiply,
    quat_wxyz_rotate,
    quat_wxyz_to_rotation_vector,
)


# Linear position/velocity, WXYZ rotation, and parent-frame angular velocity.
TransitionChannels = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Components = TypeVar("Components")
EnvSelection = torch.Tensor | slice


class NativeMotionTransition(Protocol[BackendSample, Components]):
    """Engine-specific storage with a shared placement and smoothing contract.

    align() consumes owned library samples and may modify their tensors.
    apply_offsets() may modify its components. Never pass borrowed frame buffers
    to either method. to_states() returns snapshots independent of future calls.
    capture_offsets() reads both inputs and updates only persistent offsets.
    Offset capture/application requires smoothing to be enabled.
    """

    def root_pose(self, state: BackendSample) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world root position and WXYZ orientation."""
        ...

    def root_velocity(self, state: BackendSample) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world root linear and angular velocity."""
        ...

    def aligned_root(
        self, components: Components
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def reset(self, env_ids: EnvSelection) -> None: ...

    def zero_velocities(self, components: Components, held: torch.Tensor) -> Components:
        """Zero all rates in held [N] or [N, K] samples; preserve poses."""
        ...

    def align(
        self,
        state: BackendSample,
        bodies: BodyMotionState,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> Components: ...

    def capture_offsets(
        self,
        before: Components,
        after: Components,
        env_ids: EnvSelection,
        changed: torch.Tensor,
    ) -> None: ...

    def apply_offsets(
        self, components: Components, env_ids: EnvSelection, age: torch.Tensor
    ) -> Components: ...

    def to_states(
        self, components: Components
    ) -> tuple[BackendSample, BodyMotionState]: ...


class TransitionOffsets:
    """Persistent linear and rotation-vector offsets for selected environments."""

    def __init__(self, template, num_envs, position_shape, rotation_shape, rate):
        self.rate = rate
        self.pos = template.new_zeros((num_envs, *position_shape))
        self.vel = torch.zeros_like(self.pos)
        self.pos_slope = torch.zeros_like(self.pos)
        self.rot = template.new_zeros((num_envs, *rotation_shape))
        self.ang = torch.zeros_like(self.rot)
        self.rot_slope = torch.zeros_like(self.rot)

    def reset(self, env_ids):
        for value in (
            self.pos,
            self.vel,
            self.pos_slope,
            self.rot,
            self.ang,
            self.rot_slope,
        ):
            value[env_ids] = 0

    def capture_offsets(
        self,
        before: TransitionChannels,
        after: TransitionChannels,
        env_ids: EnvSelection,
        changed: torch.Tensor,
    ) -> None:
        old_pos, old_vel, old_quat, old_ang = before
        pos, vel, quat, ang = after
        difference = quat_wxyz_multiply(old_quat, quat_wxyz_conjugate(quat))
        rot_offset = quat_wxyz_to_rotation_vector(difference)
        ang_offset = _left_jacobian(
            rot_offset, old_ang - quat_wxyz_rotate(difference, ang), inverse=True
        )
        pos_offset, vel_offset = old_pos - pos, old_vel - vel
        for destination, value in (
            (self.pos, pos_offset),
            (self.vel, vel_offset),
            (self.rot, rot_offset),
            (self.ang, ang_offset),
            (self.pos_slope, vel_offset + self.rate * pos_offset),
            (self.rot_slope, ang_offset + self.rate * rot_offset),
        ):
            mask = changed.reshape((-1,) + (1,) * (value.ndim - 1))
            destination[env_ids] = torch.where(mask, value, destination[env_ids])

    def apply_offsets(
        self,
        components: TransitionChannels,
        env_ids: EnvSelection,
        age: torch.Tensor,
        decay: torch.Tensor,
    ) -> TransitionChannels:
        pos, vel, quat, ang_vel = components
        offsets = [
            value[env_ids]
            for value in (
                self.pos,
                self.vel,
                self.pos_slope,
                self.rot,
                self.ang,
                self.rot_slope,
            )
        ]
        if age.ndim == 2:
            offsets = [value[:, None] for value in offsets]
        pos_time = age.reshape((*age.shape, *((1,) * (pos.ndim - age.ndim))))
        rot_time = age.reshape((*age.shape, *((1,) * (quat.ndim - age.ndim))))
        pos_decay = decay.reshape(pos_time.shape)
        rot_decay = decay.reshape(rot_time.shape)
        pos_offset, vel_offset = _decay(*offsets[:3], pos_time, pos_decay, self.rate)
        rot_offset, ang_offset = _decay(*offsets[3:], rot_time, rot_decay, self.rate)
        difference = _rotation_quat(rot_offset)
        return (
            pos + pos_offset,
            vel + vel_offset,
            quat_wxyz_multiply(difference, quat),
            _left_jacobian(rot_offset, ang_offset)
            + quat_wxyz_rotate(difference, ang_vel),
        )


def coordinate_ids(layout, device, *, xyzw):
    """Cache native scalar, quaternion, and tangent indices once."""
    free = [
        block
        for block in layout.blocks
        if block.type == ArticulationCoordinateType.FREE
    ]
    if len(free) != 1 or free[0].body_index != 0:
        raise ValueError("batch playback requires one floating root")
    root = free[0]
    linear_q, linear_qd, rotation_q, rotation_qd = [], [], [], []
    for block in layout.blocks:
        if block.type in (
            ArticulationCoordinateType.FREE,
            ArticulationCoordinateType.SPHERICAL,
        ):
            offset = 3 if block.type == ArticulationCoordinateType.FREE else 0
            order = (3, 0, 1, 2) if xyzw else (0, 1, 2, 3)
            rotation_q.append([block.q_offset + offset + i for i in order])
            rotation_qd.append([block.qd_offset + offset + i for i in range(3)])
            width = offset
        else:
            width = block.q_size
        linear_q.extend(range(block.q_offset, block.q_offset + width))
        linear_qd.extend(range(block.qd_offset, block.qd_offset + width))
    ids = [
        torch.tensor(values, dtype=torch.long, device=device)
        for values in (linear_q, linear_qd, rotation_q, rotation_qd)
    ]
    return root, *ids


def align_bodies(bodies, rotation, translation):
    rotation = rotation.unsqueeze(-2)
    return BodyMotionState(
        quat_wxyz_rotate(rotation, bodies.body_pos_w) + translation.unsqueeze(-2),
        quat_wxyz_multiply(rotation, bodies.body_quat_wxyz_w),
        quat_wxyz_rotate(rotation, bodies.body_vel_w),
        quat_wxyz_rotate(rotation, bodies.body_ang_vel_w),
    )


def body_components(bodies: BodyMotionState) -> TransitionChannels:
    return (
        bodies.body_pos_w,
        bodies.body_vel_w,
        bodies.body_quat_wxyz_w,
        bodies.body_ang_vel_w,
    )


def body_state(components: TransitionChannels) -> BodyMotionState:
    pos, vel, quat, ang = components
    return BodyMotionState(pos, quat, vel, ang)


def _decay(position, velocity, slope, time, decay, rate):
    # Same critically damped offset equation as SkeletonPoseSmoother.
    return (position + slope * time) * decay, (velocity - slope * rate * time) * decay


def _rotation_quat(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1)
    return quat_wxyz_from_angle_axis(angle, vector)


def _left_jacobian(rotation, velocity, *, inverse=False):
    # Map rotation-vector derivatives to parent-frame angular rates (and back).
    squared = (rotation**2).sum(-1, keepdim=True)
    safe_squared = squared.clamp_min(1e-8)
    angle = safe_squared.sqrt()
    cross = torch.cross(rotation, velocity, dim=-1)
    double_cross = torch.cross(rotation, cross, dim=-1)
    if inverse:
        coefficient = torch.where(
            squared < 1e-6,
            1 / 12 + squared / 720,
            (1 - 0.5 * angle / torch.tan(0.5 * angle)) / safe_squared,
        )
        return velocity - 0.5 * cross + coefficient * double_cross
    first = torch.where(
        squared < 1e-6, 0.5 - squared / 24, (1 - torch.cos(angle)) / safe_squared
    )
    second = torch.where(
        squared < 1e-6,
        1 / 6 - squared / 120,
        (angle - torch.sin(angle)) / (safe_squared * angle),
    )
    return velocity + first * cross + second * double_cross
