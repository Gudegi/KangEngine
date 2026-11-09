"""Newton transitions in native generalized coordinates."""

import torch

from ...animation.articulation_motion import ArticulationCoordinateLayout
from ...animation.motion_library import BodyMotionState
from ...utils.batched_rotations import (
    quat_wxyz_multiply,
    quat_wxyz_rotate,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
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
from .motion import NewtonMotionState


TransitionState = tuple[NewtonMotionState, BodyMotionState]


class NewtonMotionTransition:
    def __init__(
        self,
        layout: ArticulationCoordinateLayout,
        native: NewtonMotionState,
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
        ) = coordinate_ids(layout, native.joint_q.device, xyzw=True)
        self._root_pos_slice = slice(root.q_offset, root.q_offset + 3)
        self._root_quat_slice = slice(root.q_offset + 3, root.q_offset + 7)
        self._root_vel_slice = slice(root.qd_offset, root.qd_offset + 3)
        self._root_ang_slice = slice(root.qd_offset + 3, root.qd_offset + 6)
        self._smoothing_enabled = rate > 0
        if self._smoothing_enabled:
            self._offsets = TransitionOffsets(
                native.joint_q,
                num_envs,
                (self._linear_q_ids.numel(),),
                (self._quat_q_ids.shape[0], 3),
                rate,
            )
            self._body_offsets = TransitionOffsets(
                native.joint_q,
                num_envs,
                (buffers.body_pos_w.shape[-2], 3),
                (buffers.body_pos_w.shape[-2], 3),
                rate,
            )

    def root_pose(self, state: NewtonMotionState) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            state.joint_q[..., self._root_pos_slice],
            quat_xyzw_to_wxyz(state.joint_q[..., self._root_quat_slice]),
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
        state: NewtonMotionState,
        bodies: BodyMotionState,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> TransitionState:
        # Consume the library's owned sample without copying its native arrays.
        q, qd = state.joint_q, state.joint_qd
        position, quaternion = self.root_pose(state)
        q[..., self._root_pos_slice] = (
            quat_wxyz_rotate(rotation, position) + translation
        )
        q[..., self._root_quat_slice] = quat_wxyz_to_xyzw(
            quat_wxyz_multiply(rotation, quaternion)
        )
        qd[..., self._root_vel_slice] = quat_wxyz_rotate(
            rotation, state.joint_qd[..., self._root_vel_slice]
        )
        qd[..., self._root_ang_slice] = quat_wxyz_rotate(
            rotation, state.joint_qd[..., self._root_ang_slice]
        )
        return (
            NewtonMotionState(q, qd),
            align_bodies(bodies, rotation, translation),
        )

    def _joint_components(self, state: NewtonMotionState) -> TransitionChannels:
        quaternion = state.joint_q[..., self._quat_q_ids]
        angular = state.joint_qd[..., self._ang_qd_ids]
        # Root rates are world-space; ball-joint rates use the joint parent frame.
        return (
            state.joint_q[..., self._linear_q_ids],
            state.joint_qd[..., self._linear_qd_ids],
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
        state.joint_q[..., self._linear_q_ids] = pos
        state.joint_qd[..., self._linear_qd_ids] = vel
        state.joint_q[..., self._quat_q_ids] = quat
        state.joint_qd[..., self._ang_qd_ids] = ang
        bodies = body_state(
            self._body_offsets.apply_offsets(
                body_components(bodies), env_ids, age, decay
            )
        )
        return state, bodies

    def to_states(
        self, components: TransitionState
    ) -> tuple[NewtonMotionState, BodyMotionState]:
        return components
