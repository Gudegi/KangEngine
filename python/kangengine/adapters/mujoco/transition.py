"""MuJoCo transitions in native generalized coordinates."""

import torch

from ...animation.articulation_motion import ArticulationCoordinateLayout
from ...animation.motion_library import BodyMotionState
from ...utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_rotate,
)
from .._motion_transition import (
    EnvSelection,
    TransitionChannels,
    TransitionOffsets,
    align_bodies,
    body_components,
    body_state,
    coordinate_ids,
)
from .motion import MuJoCoMotionState


TransitionState = tuple[MuJoCoMotionState, BodyMotionState]


class MuJoCoMotionTransition:
    def __init__(
        self,
        layout: ArticulationCoordinateLayout,
        native: MuJoCoMotionState,
        buffers: BodyMotionState,
        num_envs: int,
        rate: float,
    ) -> None:
        (
            root,
            self._linear_q_ids,
            self._linear_qd_ids,
            self._quat_q_ids,
            self._ang_qd_ids,
        ) = coordinate_ids(layout, native.qpos.device, xyzw=False)
        self._root_pos_slice = slice(root.q_offset, root.q_offset + 3)
        self._root_quat_slice = slice(root.q_offset + 3, root.q_offset + 7)
        self._root_vel_slice = slice(root.qd_offset, root.qd_offset + 3)
        self._root_ang_slice = slice(root.qd_offset + 3, root.qd_offset + 6)
        self._smoothing_enabled = rate > 0
        if self._smoothing_enabled:
            self._offsets = TransitionOffsets(
                native.qpos,
                num_envs,
                (self._linear_q_ids.numel(),),
                (self._quat_q_ids.shape[0], 3),
                rate,
            )
            self._body_offsets = TransitionOffsets(
                native.qpos,
                num_envs,
                (buffers.body_pos_w.shape[-2], 3),
                (buffers.body_pos_w.shape[-2], 3),
                rate,
            )

    def root_pose(self, state: MuJoCoMotionState) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            state.qpos[..., self._root_pos_slice],
            state.qpos[..., self._root_quat_slice],
        )

    def root_velocity(
        self, state: MuJoCoMotionState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            state.qvel[..., self._root_vel_slice],
            quat_wxyz_rotate(
                state.qpos[..., self._root_quat_slice],
                state.qvel[..., self._root_ang_slice],
            ),
        )

    def aligned_root(
        self, components: TransitionState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.root_pose(components[0])

    def reset(self, env_ids: EnvSelection) -> None:
        if self._smoothing_enabled:
            self._offsets.reset(env_ids)
            self._body_offsets.reset(env_ids)

    def align(
        self,
        state: MuJoCoMotionState,
        bodies: BodyMotionState,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> TransitionState:
        # Consume the library's owned sample without copying its native arrays.
        q, qd = state.qpos, state.qvel
        position, quaternion = self.root_pose(state)
        q[..., self._root_pos_slice] = (
            quat_wxyz_rotate(rotation, position) + translation
        )
        q[..., self._root_quat_slice] = quat_wxyz_multiply(rotation, quaternion)
        qd[..., self._root_vel_slice] = quat_wxyz_rotate(
            rotation, state.qvel[..., self._root_vel_slice]
        )
        # Left-multiplying the root orientation leaves local angular rates unchanged.
        return (
            MuJoCoMotionState(q, qd),
            align_bodies(bodies, rotation, translation),
        )

    def _joint_components(self, state: MuJoCoMotionState) -> TransitionChannels:
        quaternion = state.qpos[..., self._quat_q_ids]
        angular = state.qvel[..., self._ang_qd_ids]
        # Inertial rotation offsets use parent-frame rates, including the world root.
        angular = quat_wxyz_rotate(quaternion, angular)
        return (
            state.qpos[..., self._linear_q_ids],
            state.qvel[..., self._linear_qd_ids],
            quaternion,
            angular,
        )

    def capture_offsets(
        self,
        before: TransitionState,
        after: TransitionState,
        env_ids: EnvSelection,
        changed: torch.Tensor,
    ) -> None:
        self._offsets.capture_offsets(
            self._joint_components(before[0]),
            self._joint_components(after[0]),
            env_ids,
            changed,
        )
        self._body_offsets.capture_offsets(
            body_components(before[1]), body_components(after[1]), env_ids, changed
        )

    def apply_offsets(
        self, components: TransitionState, env_ids: EnvSelection, age: torch.Tensor
    ) -> TransitionState:
        state, bodies = components
        decay = (-self._offsets.rate * age[..., None]).exp()
        pos, vel, quat, ang = self._offsets.apply_offsets(
            self._joint_components(state), env_ids, age, decay
        )
        # align() owns these snapshots; scatter only the native coordinate groups.
        state.qpos[..., self._linear_q_ids] = pos
        state.qvel[..., self._linear_qd_ids] = vel
        state.qpos[..., self._quat_q_ids] = quat
        state.qvel[..., self._ang_qd_ids] = quat_wxyz_rotate(
            quat_wxyz_conjugate(quat), ang
        )
        bodies = body_state(
            self._body_offsets.apply_offsets(
                body_components(bodies), env_ids, age, decay
            )
        )
        return state, bodies

    def to_states(
        self, components: TransitionState
    ) -> tuple[MuJoCoMotionState, BodyMotionState]:
        return components

    def zero_velocities(
        self, components: TransitionState, held: torch.Tensor
    ) -> TransitionState:
        state, bodies = components
        state.qvel.masked_fill_(held[..., None], 0)
        bodies.body_vel_w.masked_fill_(held[..., None, None], 0)
        bodies.body_ang_vel_w.masked_fill_(held[..., None, None], 0)
        return components
