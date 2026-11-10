"""PhysX transition channels retain split root and scalar-joint coordinates."""

import torch

from ...animation.motion_library import BodyMotionState
from ...utils.batched_rotations import (
    quat_wxyz_multiply,
    quat_wxyz_rotate,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from .._motion_transition import EnvSelection, TransitionChannels, TransitionOffsets
from .motion import PhysXMotionState


class PhysXMotionTransition:
    def __init__(
        self,
        native: PhysXMotionState,
        buffers: BodyMotionState,
        num_envs: int,
        rate: float,
    ) -> None:
        if native.root_pos is None:
            raise ValueError("batch playback requires a floating root")
        self._joint_count = native.joint_pos.shape[-1]
        self._body_count = buffers.body_pos_w.shape[-2]
        self._smoothing_enabled = rate > 0
        if self._smoothing_enabled:
            width = 3 + self._joint_count + 3 * self._body_count
            self._offsets = TransitionOffsets(
                native.root_pos, num_envs, (width,), (1 + self._body_count, 3), rate
            )

    def root_pose(self, state: PhysXMotionState) -> tuple[torch.Tensor, torch.Tensor]:
        return state.root_pos, quat_xyzw_to_wxyz(state.root_quat_xyzw)

    def root_velocity(
        self, state: PhysXMotionState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return state.root_vel, state.root_ang_vel

    def aligned_root(
        self, components: TransitionChannels
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return components[0][..., :3], components[2][..., 0, :]

    def reset(self, env_ids: EnvSelection) -> None:
        if self._smoothing_enabled:
            self._offsets.reset(env_ids)

    def capture_offsets(
        self,
        before: TransitionChannels,
        after: TransitionChannels,
        env_ids: EnvSelection,
        changed: torch.Tensor,
    ) -> None:
        self._offsets.capture_offsets(before, after, env_ids, changed)

    def zero_velocities(
        self, components: TransitionChannels, held: torch.Tensor
    ) -> TransitionChannels:
        pos, vel, quat, ang = components
        return (
            pos,
            vel.masked_fill(held[..., None], 0),
            quat,
            ang.masked_fill(held[..., None, None], 0),
        )

    def apply_offsets(
        self, components: TransitionChannels, env_ids: EnvSelection, age: torch.Tensor
    ) -> TransitionChannels:
        decay = torch.exp(-self._offsets.rate * age[..., None])
        return self._offsets.apply_offsets(components, env_ids, age, decay)

    def align(
        self,
        state: PhysXMotionState,
        bodies: BodyMotionState,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> TransitionChannels:
        # Keep internal rotations WXYZ; convert only at the public state boundary.
        body_rotation = rotation.unsqueeze(-2)
        body_pos_w = quat_wxyz_rotate(
            body_rotation, bodies.body_pos_w
        ) + translation.unsqueeze(-2)
        body_vel_w = quat_wxyz_rotate(body_rotation, bodies.body_vel_w)
        root_quat = quat_wxyz_multiply(
            rotation, quat_xyzw_to_wxyz(state.root_quat_xyzw)
        )
        return (
            torch.cat(
                (
                    quat_wxyz_rotate(rotation, state.root_pos) + translation,
                    state.joint_pos,
                    body_pos_w.flatten(-2),
                ),
                -1,
            ),
            torch.cat(
                (
                    quat_wxyz_rotate(rotation, state.root_vel),
                    state.joint_vel,
                    body_vel_w.flatten(-2),
                ),
                -1,
            ),
            torch.cat(
                (
                    root_quat.unsqueeze(-2),
                    quat_wxyz_multiply(body_rotation, bodies.body_quat_wxyz_w),
                ),
                -2,
            ),
            torch.cat(
                (
                    quat_wxyz_rotate(rotation, state.root_ang_vel).unsqueeze(-2),
                    quat_wxyz_rotate(body_rotation, bodies.body_ang_vel_w),
                ),
                -2,
            ),
        )

    def to_states(
        self, components: TransitionChannels
    ) -> tuple[PhysXMotionState, BodyMotionState]:
        pos, vel, quat, ang = components
        end = 3 + self._joint_count
        shape = (*pos.shape[:-1], self._body_count, 3)
        return (
            PhysXMotionState(
                pos[..., 3:end],
                vel[..., 3:end],
                pos[..., :3],
                quat_wxyz_to_xyzw(quat[..., 0, :]),
                vel[..., :3],
                ang[..., 0, :],
            ),
            BodyMotionState(
                pos[..., end:].reshape(shape),
                quat[..., 1:, :],
                vel[..., end:].reshape(shape),
                ang[..., 1:, :],
            ),
        )
