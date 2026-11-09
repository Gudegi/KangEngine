"""Conversion between canonical articulation motion and PhysX buffers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt
import torch

from ..._core import _ke
from ...animation.articulation_motion import (
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
    ArticulationMotion,
    ArticulationMotionMapper,
)
from ...utils import batched_rotations
from ...utils.math import (
    quat_wxyz_from_rotation_vector_numpy,
    quat_wxyz_multiply_numpy,
    quat_wxyz_to_rotation_vector_numpy,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)


FloatArray = npt.NDArray[np.float32]

if TYPE_CHECKING:
    from .transition import PhysXMotionTransition


def _compose_angles(angles: FloatArray, blocks) -> FloatArray:
    result = np.zeros((angles.shape[0], 4), dtype=np.float32)
    result[:, 0] = 1.0
    for index, block in enumerate(blocks):
        axis = np.asarray(block.axis, dtype=np.float32)
        axis /= np.linalg.norm(axis)
        half = 0.5 * angles[:, index]
        rotation = np.empty_like(result)
        rotation[:, 0] = np.cos(half)
        rotation[:, 1:] = np.sin(half)[:, None] * axis
        result = quat_wxyz_multiply_numpy(result, rotation)
    return result


def _decompose_revolute_group(quaternion: FloatArray, blocks) -> FloatArray:
    """Solve authored sequential hinge angles for batch quaternions."""

    angles = np.zeros((quaternion.shape[0], len(blocks)), dtype=np.float32)
    step = 1.0e-4
    for frame in range(quaternion.shape[0]):
        target = quaternion[frame : frame + 1]
        value = angles[frame : frame + 1]
        for _ in range(24):
            current = _compose_angles(value, blocks)
            inverse = current.copy()
            inverse[:, 1:] *= -1.0
            error = quat_wxyz_to_rotation_vector_numpy(
                quat_wxyz_multiply_numpy(inverse, target)
            )[0]
            if np.linalg.norm(error) < 1.0e-6:
                break
            jacobian = np.empty((3, len(blocks)), dtype=np.float32)
            for column in range(len(blocks)):
                perturbed = value.copy()
                perturbed[:, column] += step
                next_rotation = _compose_angles(perturbed, blocks)
                next_inverse = next_rotation.copy()
                next_inverse[:, 1:] *= -1.0
                next_error = quat_wxyz_to_rotation_vector_numpy(
                    quat_wxyz_multiply_numpy(next_inverse, target)
                )[0]
                jacobian[:, column] = (next_error - error) / step
            delta = np.linalg.solve(
                jacobian.T @ jacobian + 1.0e-6 * np.eye(len(blocks), dtype=np.float32),
                -(jacobian.T @ error),
            )
            value[0] += delta
            if np.linalg.norm(delta) < 1.0e-7:
                break
    return angles


def _physx_group_coordinates(angles: FloatArray, blocks) -> FloatArray:
    rotation_vector = quat_wxyz_to_rotation_vector_numpy(
        _compose_angles(angles, blocks)
    )
    result = np.empty_like(angles)
    for index, block in enumerate(blocks):
        axis = np.asarray(block.axis, dtype=np.float32)
        axis /= np.linalg.norm(axis)
        result[:, index] = rotation_vector @ axis
    return result


def _physx_group_jacobian(angles: FloatArray, blocks) -> FloatArray:
    step = 1.0e-3
    jacobian = np.empty((angles.shape[0], len(blocks), len(blocks)), dtype=np.float32)
    for column in range(len(blocks)):
        positive = angles.copy()
        negative = angles.copy()
        positive[:, column] += step
        negative[:, column] -= step
        jacobian[:, :, column] = (
            _physx_group_coordinates(positive, blocks)
            - _physx_group_coordinates(negative, blocks)
        ) / (2.0 * step)
    return jacobian


def _twist_angles_wxyz(quaternion: FloatArray, axis: FloatArray) -> FloatArray:
    """Return the signed quaternion twist around one normalized axis."""

    normalized_axis = np.asarray(axis, dtype=np.float32)
    normalized_axis /= np.linalg.norm(normalized_axis)
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True),
        1.0e-8,
    )
    half_angle = np.arctan2(
        quaternion[:, 1:] @ normalized_axis,
        quaternion[:, 0],
    )
    return np.arctan2(
        np.sin(2.0 * half_angle),
        np.cos(2.0 * half_angle),
    )


@dataclass(frozen=True)
class PhysXMotionBuffers:
    """Frame-major PhysX root and logical joint state buffers."""

    joint_pos: FloatArray
    joint_vel: FloatArray
    root_pos: FloatArray | None
    root_quat_xyzw: FloatArray | None
    root_vel: FloatArray | None
    root_ang_vel: FloatArray | None


@dataclass(frozen=True)
class PhysXMotionState:
    """PhysX root and logical joint tensors for stored frames or queries."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor | None
    root_quat_xyzw: torch.Tensor | None
    root_vel: torch.Tensor | None
    root_ang_vel: torch.Tensor | None


class PhysXMotionAdapter:
    """Convert one reference trajectory into KangEngine PhysX logical order."""

    def __init__(self, layout: ArticulationCoordinateLayout) -> None:
        self._layout = layout
        self._model_signature = layout.model_signature
        self._free_block = None
        scalar_blocks = []
        for block in layout.blocks:
            if block.type == ArticulationCoordinateType.FIXED:
                continue
            if block.type == ArticulationCoordinateType.FREE:
                if self._free_block is not None:
                    raise ValueError("canonical layout contains multiple free roots")
                self._free_block = block
            elif block.type == ArticulationCoordinateType.SPHERICAL:
                raise NotImplementedError(
                    "KangEngine PhysX native spherical coordinates are not yet "
                    "exposed as a complete twist/swing state; use authored scalar "
                    "hinges or constrained IK before PhysX packing"
                )
            elif block.q_size == 1 and block.qd_size == 1:
                scalar_blocks.append(block)
            else:
                raise ValueError(
                    f"PhysX joint {block.joint_name!r} is not a scalar coordinate"
                )
        self._scalar_blocks = tuple(scalar_blocks)
        self._joint_names = tuple(block.joint_name for block in scalar_blocks)
        groups = {}
        for index, block in enumerate(self._scalar_blocks):
            groups.setdefault(block.body_index, []).append((index, block))
        self._scalar_groups = tuple(tuple(group) for group in groups.values())
        width = max((len(group) for group in self._scalar_groups), default=0)
        self._fk_joint_ids = np.zeros((len(groups), width), dtype=np.int64)
        self._fk_axes = np.zeros((len(groups), width, 3), dtype=np.float32)
        self._fk_body_ids = tuple(
            group[0][1].body_index for group in self._scalar_groups
        )
        for group_idx, group in enumerate(self._scalar_groups):
            for axis_idx, (joint_idx, block) in enumerate(group):
                axis = np.asarray(block.axis)
                self._fk_joint_ids[group_idx, axis_idx] = joint_idx
                self._fk_axes[group_idx, axis_idx] = axis / np.linalg.norm(axis)
        self._fk_supported = all(
            block.type == ArticulationCoordinateType.REVOLUTE
            for block in self._scalar_blocks
        )
        self._fk_reference = {}
        self._dof_mapping_cache = {}

    @property
    def layout(self) -> ArticulationCoordinateLayout:
        return self._layout

    def _create_transition(
        self, native, buffers, num_envs, rate
    ) -> PhysXMotionTransition:
        from .transition import PhysXMotionTransition

        return PhysXMotionTransition(native, buffers, num_envs, rate)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def make_dof_mapping(
        self,
        body_names: Sequence[str] | None = None,
        *,
        dof_offsets: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        """Map a requested body-grouped DOF order to PhysX logical indices.

        Omitting ``body_names`` returns the identity mapping. ``dof_offsets``
        optionally verifies the expected number of scalar coordinates in each
        requested body group.
        """

        if body_names is None:
            if dof_offsets is not None:
                raise ValueError("dof_offsets require body_names")
            return tuple(range(len(self._scalar_blocks)))

        requested = tuple(str(name) for name in body_names)
        if len(set(requested)) != len(requested):
            raise ValueError("body_names must not contain duplicates")
        indices_by_body: dict[str, list[int]] = {}
        for index, block in enumerate(self._scalar_blocks):
            indices_by_body.setdefault(block.body_name, []).append(index)

        offsets = (
            tuple(int(value) for value in dof_offsets)
            if dof_offsets is not None
            else None
        )
        if offsets is not None:
            if len(offsets) != len(requested) + 1 or offsets[0] != 0:
                raise ValueError("dof_offsets must delimit every requested body")
            if any(end <= start for start, end in zip(offsets, offsets[1:])):
                raise ValueError("dof_offsets must be strictly increasing")

        mapping = []
        for body_id, body_name in enumerate(requested):
            indices = indices_by_body.get(body_name)
            if indices is None:
                raise ValueError(f"PhysX layout has no scalar DOF for {body_name!r}")
            if offsets is not None:
                expected = offsets[body_id + 1] - offsets[body_id]
                if len(indices) != expected:
                    raise ValueError(
                        f"PhysX DOF count for {body_name!r} differs from "
                        f"requested order: {len(indices)} != {expected}"
                    )
            mapping.extend(indices)
        return tuple(mapping)

    def reorder_joint_values(self, values, mapping: Sequence[int] | None = None):
        """Return joint values in requested order, or unchanged without a mapping."""

        if mapping is None:
            return values
        indices = self._dof_mapping_indices(mapping, values, require_permutation=False)
        if isinstance(values, torch.Tensor):
            return torch.index_select(values, -1, indices)
        array = np.asarray(values)
        return np.take(array, indices, axis=-1)

    def restore_joint_values_order(self, values, mapping: Sequence[int] | None = None):
        """Restore reordered joint values to PhysX logical order."""

        if mapping is None:
            return values
        indices = self._dof_mapping_indices(mapping, values, require_permutation=True)
        if values.shape[-1] != len(indices):
            raise ValueError("joint value width does not match DOF mapping")
        result = (
            torch.empty_like(values)
            if isinstance(values, torch.Tensor)
            else np.empty_like(values)
        )
        if isinstance(values, torch.Tensor):
            result.index_copy_(-1, indices, values)
        else:
            result[..., indices] = values
        return result

    def reorder_joint_state(
        self,
        state: PhysXMotionBuffers | PhysXMotionState,
        mapping: Sequence[int] | None = None,
    ) -> PhysXMotionBuffers | PhysXMotionState:
        """Return a PhysX motion state with reordered joint position and velocity."""

        if mapping is None:
            return state
        return replace(
            state,
            joint_pos=self.reorder_joint_values(state.joint_pos, mapping),
            joint_vel=self.reorder_joint_values(state.joint_vel, mapping),
        )

    def restore_joint_state_order(
        self,
        state: PhysXMotionBuffers | PhysXMotionState,
        mapping: Sequence[int] | None = None,
    ) -> PhysXMotionBuffers | PhysXMotionState:
        """Return a reordered state restored to PhysX logical joint order."""

        if mapping is None:
            return state
        return replace(
            state,
            joint_pos=self.restore_joint_values_order(state.joint_pos, mapping),
            joint_vel=self.restore_joint_values_order(state.joint_vel, mapping),
        )

    def validate_articulation(self, articulation) -> None:
        """Verify a live PhysX articulation uses canonical logical DOF order."""

        if hasattr(articulation, "joint_names"):
            names = tuple(articulation.joint_names)
        else:
            names = tuple(articulation.get_dof_names())
        if names != self._joint_names:
            raise ValueError(
                "PhysX logical DOF order differs from canonical layout: "
                f"{names!r} != {self._joint_names!r}"
            )

    def pack(self, motion: ArticulationMotion) -> PhysXMotionBuffers:
        """Convert canonical motion into PhysX root and logical DOF buffers."""

        self._validate_motion(motion)
        skeleton_motion = ArticulationMotionMapper(self._layout).to_skeleton_motion(
            motion
        )
        return self.pack_skeleton_motion(skeleton_motion)

    def pack_skeleton_motion(self, motion) -> PhysXMotionBuffers:
        """Convert local joint quaternions directly to PhysX coordinates."""

        body_names = motion.node_names()
        if any(
            block.body_index >= len(body_names)
            or body_names[block.body_index] != block.body_name
            for block in self._layout.blocks
        ):
            raise ValueError("SkeletonMotion does not match PhysX layout body order")

        rotations = np.asarray(motion.local_rotations_wxyz(), dtype=np.float32)
        frames = rotations.shape[0]
        joint_pos = np.empty((frames, len(self._scalar_blocks)), dtype=np.float32)
        for group in self._scalar_groups:
            body_index = group[0][1].body_index
            block = group[0][1]
            reference = np.asarray(block.reference_rotation, dtype=np.float32)
            reference_inverse = np.broadcast_to(
                reference * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                (frames, 4),
            )
            relative = quat_wxyz_multiply_numpy(
                reference_inverse, rotations[:, body_index]
            )
            if len(group) == 1:
                index, item = group[0]
                joint_pos[:, index] = _twist_angles_wxyz(
                    relative,
                    np.asarray(item.axis, dtype=np.float32),
                )
            else:
                rotation_vector = quat_wxyz_to_rotation_vector_numpy(relative)
                for index, item in group:
                    axis = np.asarray(item.axis, dtype=np.float32)
                    axis /= np.linalg.norm(axis)
                    joint_pos[:, index] = rotation_vector @ axis

        joint_vel = np.zeros_like(joint_pos)
        if frames > 1:
            for group in self._scalar_groups:
                body_index = group[0][1].body_index
                inverse = rotations[:-1, body_index].copy()
                inverse[:, 1:] *= -1.0
                delta = quat_wxyz_multiply_numpy(inverse, rotations[1:, body_index])
                angular_velocity = quat_wxyz_to_rotation_vector_numpy(delta) * float(
                    motion.fps()
                )
                for index, block in group:
                    axis = np.asarray(block.axis, dtype=np.float32)
                    axis /= np.linalg.norm(axis)
                    joint_vel[:-1, index] = angular_velocity @ axis
            joint_vel[-1] = joint_vel[-2]

        if self._free_block is None:
            root_pos = None
            root_rotations = None
            root_vel = None
            root_ang_vel = None
        else:
            root_pos = np.asarray(motion.root_translations(), dtype=np.float32)
            root_rotations = quat_wxyz_to_xyzw(rotations[:, 0])
            root_vel = np.asarray(motion.root_linear_velocities(), dtype=np.float32)
            root_ang_vel = np.asarray(
                motion.global_angular_velocities(), dtype=np.float32
            )[:, 0]

        return PhysXMotionBuffers(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_pos=root_pos,
            root_quat_xyzw=root_rotations,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
        )

    def prepare_motion(self, motion) -> FloatArray:
        """Prepare DOFs and optional root state.

        Columns: DOF positions(D), velocities(D), optional free-root state(13).
        Root state is position, XYZW rotation, linear and angular velocity.
        """
        if isinstance(motion, ArticulationMotion):
            motion = ArticulationMotionMapper(self._layout).to_skeleton_motion(motion)
        buffers = self.pack_skeleton_motion(motion)
        columns = [buffers.joint_pos, buffers.joint_vel]
        if self._free_block is not None:
            columns.extend(
                (
                    buffers.root_pos,
                    buffers.root_quat_xyzw,
                    buffers.root_vel,
                    buffers.root_ang_vel,
                )
            )
        return np.concatenate(columns, axis=-1)

    def prepare_body_motion(self, data: FloatArray, *, fps: float):
        """Reconstruct robot-representable poses from prepared CPU frames."""
        state = self.frame_buffers(torch.as_tensor(data, dtype=torch.float32))
        root_pos, rotations = self.to_skeleton_pose(state)
        q = np.zeros(self._layout.nq, dtype=np.float32)
        if self._free_block is not None:
            q[self._free_block.q_offset + 3] = 1.0
        reference = ArticulationMotionMapper(self._layout).to_skeleton_state(q)
        return _ke.animation.SkeletonMotion.from_arrays(
            skeleton_tree=reference.skeleton_tree,
            root_translations=root_pos.numpy(),
            local_rotations_wxyz=rotations.numpy(),
            fps=fps,
        )

    def frame_buffers(self, data: torch.Tensor) -> PhysXMotionState:
        """Borrow typed PhysX frame buffers without copying or resampling."""
        count = len(self._scalar_blocks)
        root = data[..., 2 * count : 2 * count + 13]
        has_root = self._free_block is not None
        return PhysXMotionState(
            joint_pos=data[..., :count],
            joint_vel=data[..., count : 2 * count],
            root_pos=root[..., :3] if has_root else None,
            root_quat_xyzw=root[..., 3:7] if has_root else None,
            root_vel=root[..., 7:10] if has_root else None,
            root_ang_vel=root[..., 10:13] if has_root else None,
        )

    def pack_sample(
        self,
        *,
        backend_frame_data: torch.Tensor,
        next_backend_frame_data: torch.Tensor,
        blend: torch.Tensor,
    ) -> PhysXMotionState:
        """Interpolate prepared PhysX poses and velocities."""
        count = len(self._scalar_blocks)
        first, following = backend_frame_data, next_backend_frame_data
        joint_pos = first[..., :count] + (
            following[..., :count] - first[..., :count]
        ) * blend.unsqueeze(-1)

        root = first[..., 2 * count : 2 * count + 13]
        next_root = following[..., 2 * count : 2 * count + 13]
        has_root = self._free_block is not None
        root_rotation = None
        if has_root:
            root_rotation = batched_rotations.quat_wxyz_to_xyzw(
                batched_rotations.quat_wxyz_slerp(
                    batched_rotations.quat_xyzw_to_wxyz(root[..., 3:7]),
                    batched_rotations.quat_xyzw_to_wxyz(next_root[..., 3:7]),
                    blend,
                )
            )
        return PhysXMotionState(
            joint_pos=joint_pos,
            joint_vel=torch.lerp(
                first[..., count : 2 * count],
                following[..., count : 2 * count],
                blend.unsqueeze(-1),
            ),
            root_pos=(
                root[..., :3]
                + (next_root[..., :3] - root[..., :3]) * blend.unsqueeze(-1)
            )
            if has_root
            else None,
            root_quat_xyzw=root_rotation,
            root_vel=torch.lerp(
                root[..., 7:10], next_root[..., 7:10], blend.unsqueeze(-1)
            )
            if has_root
            else None,
            root_ang_vel=torch.lerp(
                root[..., 10:13], next_root[..., 10:13], blend.unsqueeze(-1)
            )
            if has_root
            else None,
        )

    def to_skeleton_pose(
        self, state: PhysXMotionState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert PhysX state to root position and local WXYZ joint rotations.

        Preserves leading batch dimensions on CPU/CUDA. Use adapter joint order.
        Fixed roots use the layout's reference pose; prismatic joints are unsupported.
        """
        if not isinstance(state, PhysXMotionState):
            raise TypeError("PhysX pose conversion requires PhysXMotionState")
        if not self._fk_supported:
            raise NotImplementedError("skeleton FK does not support prismatic joints")

        # Cache reference poses and axes once per device and dtype.
        joint_pos = state.joint_pos
        key = (joint_pos.device, joint_pos.dtype)
        if key not in self._fk_reference:
            q = np.zeros(self._layout.nq, dtype=np.float32)
            if self._free_block is not None:
                q[self._free_block.q_offset + 3] = 1.0
            reference = ArticulationMotionMapper(self._layout).to_skeleton_state(q)
            tree = reference.skeleton_tree
            root = tree.local_translation(0)
            rotations = []
            for idx in range(tree.num_joints()):
                rotation = tree.local_rotation(idx)
                rotations.append((rotation.w, rotation.x, rotation.y, rotation.z))
            self._fk_reference[key] = (
                joint_pos.new_tensor((root.x, root.y, root.z)),
                joint_pos.new_tensor(rotations),
                joint_pos.new_tensor(self._fk_axes),
                torch.tensor(self._fk_joint_ids, device=joint_pos.device),
                torch.tensor(
                    self._fk_body_ids, dtype=torch.long, device=joint_pos.device
                ),
            )
        reference_root, reference_quat, axes, joint_ids, body_ids = self._fk_reference[
            key
        ]

        # Sum each group's rotation-vector components, then convert all groups together.
        rotation_vector = (joint_pos[..., joint_ids].unsqueeze(-1) * axes).sum(dim=-2)
        relative = batched_rotations.quat_xyzw_to_wxyz(
            batched_rotations.quat_xyzw_from_rotation_vector(rotation_vector)
        )
        local_quat = reference_quat.expand(*joint_pos.shape[:-1], -1, -1).clone()
        local_quat[..., body_ids, :] = batched_rotations.quat_wxyz_multiply(
            reference_quat[body_ids], relative
        )

        if self._free_block is None:
            root_pos = reference_root.expand(*joint_pos.shape[:-1], 3)
        else:
            root_pos = state.root_pos
            local_quat[..., self._free_block.body_index, :] = (
                batched_rotations.quat_xyzw_to_wxyz(state.root_quat_xyzw)
            )
        return root_pos, local_quat

    def unpack(
        self,
        joint_pos: npt.ArrayLike,
        joint_vel: npt.ArrayLike,
        *,
        fps: float,
        root_pos: npt.ArrayLike | None = None,
        root_quat_xyzw: npt.ArrayLike | None = None,
        root_vel: npt.ArrayLike | None = None,
        root_ang_vel: npt.ArrayLike | None = None,
        motion_name: str = "Motion",
    ) -> ArticulationMotion:
        """Convert frame-major PhysX root and logical DOF state to canonical motion."""

        positions = np.asarray(joint_pos, dtype=np.float32)
        velocities = np.asarray(joint_vel, dtype=np.float32)
        expected_dofs = len(self._scalar_blocks)
        if positions.ndim != 2 or positions.shape[1] != expected_dofs:
            raise ValueError(f"joint_pos expected shape [frames, {expected_dofs}]")
        if velocities.shape != positions.shape:
            raise ValueError("joint_vel must match joint_pos shape")

        q = np.zeros((positions.shape[0], self._layout.nq), dtype=np.float32)
        qd = np.zeros((positions.shape[0], self._layout.nv), dtype=np.float32)
        for group in self._scalar_groups:
            indices = [item[0] for item in group]
            blocks = [item[1] for item in group]
            if len(group) == 1:
                block = blocks[0]
                q[:, block.q_offset] = positions[:, indices[0]]
                qd[:, block.qd_offset] = velocities[:, indices[0]]
                continue

            rotation_vector = np.zeros((positions.shape[0], 3), dtype=np.float32)
            for index, block in group:
                axis = np.asarray(block.axis, dtype=np.float32)
                axis /= np.linalg.norm(axis)
                rotation_vector += positions[:, index, None] * axis
            angles = _decompose_revolute_group(
                quat_wxyz_from_rotation_vector_numpy(rotation_vector), blocks
            )
            jacobian = _physx_group_jacobian(angles, blocks)
            angular_rates = np.empty_like(angles)
            for frame in range(positions.shape[0]):
                angular_rates[frame] = np.linalg.solve(
                    jacobian[frame], velocities[frame, indices]
                )
            for column, block in enumerate(blocks):
                q[:, block.q_offset] = angles[:, column]
                qd[:, block.qd_offset] = angular_rates[:, column]

        if self._free_block is not None:
            required = {
                "root_pos": root_pos,
                "root_quat_xyzw": root_quat_xyzw,
                "root_vel": root_vel,
                "root_ang_vel": root_ang_vel,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "free-root PhysX unpack requires " + ", ".join(missing)
                )
            root_position_values = np.asarray(root_pos, dtype=np.float32)
            root_rotation_values = np.asarray(root_quat_xyzw, dtype=np.float32)
            root_linear_values = np.asarray(root_vel, dtype=np.float32)
            root_angular_values = np.asarray(root_ang_vel, dtype=np.float32)
            frames = positions.shape[0]
            expected_shapes = {
                "root_pos": (frames, 3),
                "root_quat_xyzw": (frames, 4),
                "root_vel": (frames, 3),
                "root_ang_vel": (frames, 3),
            }
            actual_values = {
                "root_pos": root_position_values,
                "root_quat_xyzw": root_rotation_values,
                "root_vel": root_linear_values,
                "root_ang_vel": root_angular_values,
            }
            for name, expected in expected_shapes.items():
                if actual_values[name].shape != expected:
                    raise ValueError(f"{name} expected shape {expected}")

            block = self._free_block
            q[:, block.q_offset : block.q_offset + 3] = root_position_values
            q[:, block.q_offset + 3 : block.q_offset + 7] = quat_xyzw_to_wxyz(
                root_rotation_values
            )
            qd[:, block.qd_offset : block.qd_offset + 3] = root_linear_values
            qd[:, block.qd_offset + 3 : block.qd_offset + 6] = root_angular_values

        return ArticulationMotion.from_arrays(
            layout=self._layout,
            q=q,
            qd=qd,
            fps=fps,
            motion_name=motion_name,
        )

    def _validate_motion(self, motion: ArticulationMotion) -> None:
        if motion.layout.model_signature != self._model_signature:
            raise ValueError("ArticulationMotion layout does not match adapter layout")

    def _dof_mapping_indices(self, mapping, values, *, require_permutation):
        device = values.device if isinstance(values, torch.Tensor) else None
        key = (tuple(mapping), device, require_permutation)
        if key not in self._dof_mapping_cache:
            indices = self._validate_dof_mapping(
                mapping, require_permutation=require_permutation
            )
            self._dof_mapping_cache[key] = (
                torch.tensor(indices, dtype=torch.long, device=device)
                if device is not None
                else np.asarray(indices, dtype=np.int64)
            )
        return self._dof_mapping_cache[key]

    def _validate_dof_mapping(
        self,
        mapping: Sequence[int],
        *,
        require_permutation: bool,
    ) -> tuple[int, ...]:
        indices = tuple(int(value) for value in mapping)
        dof_count = len(self._scalar_blocks)
        if len(set(indices)) != len(indices):
            raise ValueError("DOF mapping must not contain duplicates")
        if any(index < 0 or index >= dof_count for index in indices):
            raise ValueError("DOF mapping index is outside PhysX logical order")
        if require_permutation and len(indices) != dof_count:
            raise ValueError("restoring PhysX order requires a complete DOF mapping")
        return indices
