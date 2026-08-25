"""Conversion between canonical articulation motion and PhysX buffers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from ...animation.articulation_motion import (
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
    ArticulationMotion,
    ArticulationMotionMapper,
)
from ...utils.math import (
    quat_wxyz_from_rotation_vector_numpy,
    quat_wxyz_multiply_numpy,
    quat_wxyz_to_rotation_vector_numpy,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from ...utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_to_rotation_vector,
)


FloatArray = npt.NDArray[np.float32]


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


@dataclass(frozen=True)
class PhysXMotionBuffers:
    """Frame-major PhysX root and logical joint state buffers."""

    joint_positions: FloatArray
    joint_velocities: FloatArray
    root_positions: FloatArray | None
    root_rotations_xyzw: FloatArray | None
    root_linear_velocities: FloatArray | None
    root_angular_velocities: FloatArray | None


@dataclass(frozen=True)
class PhysXMotionTensorSample:
    """Sample-major PhysX state tensors returned by MotionLibrary."""

    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    root_positions: torch.Tensor | None
    root_rotations_xyzw: torch.Tensor | None
    root_linear_velocities: torch.Tensor | None
    root_angular_velocities: torch.Tensor | None


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

    @property
    def layout(self) -> ArticulationCoordinateLayout:
        return self._layout

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

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
        joint_positions = np.empty((frames, len(self._scalar_blocks)), dtype=np.float32)
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
            # Three-axis PhysX spherical joints use exp-map components. A
            # one-axis hinge is the same angle-axis vector projected onto its
            # authored axis.
            rotation_vector = quat_wxyz_to_rotation_vector_numpy(relative)
            for index, item in group:
                axis = np.asarray(item.axis, dtype=np.float32)
                axis /= np.linalg.norm(axis)
                joint_positions[:, index] = rotation_vector @ axis

        joint_velocities = np.zeros_like(joint_positions)
        if frames > 1:
            for group in self._scalar_groups:
                body_index = group[0][1].body_index
                inverse = rotations[:-1, body_index].copy()
                inverse[:, 1:] *= -1.0
                delta = quat_wxyz_multiply_numpy(inverse, rotations[1:, body_index])
                angular_velocity = (
                    quat_wxyz_to_rotation_vector_numpy(delta) * float(motion.fps())
                )
                for index, block in group:
                    axis = np.asarray(block.axis, dtype=np.float32)
                    axis /= np.linalg.norm(axis)
                    joint_velocities[:-1, index] = angular_velocity @ axis
            joint_velocities[-1] = joint_velocities[-2]

        if self._free_block is None:
            root_positions = None
            root_rotations = None
            root_linear_velocities = None
            root_angular_velocities = None
        else:
            root_positions = np.asarray(motion.root_translations(), dtype=np.float32)
            root_rotations = quat_wxyz_to_xyzw(rotations[:, 0])
            root_linear_velocities = np.asarray(
                motion.root_linear_velocities(), dtype=np.float32
            )
            root_angular_velocities = np.asarray(
                motion.global_angular_velocities(), dtype=np.float32
            )[:, 0]

        return PhysXMotionBuffers(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            root_positions=root_positions,
            root_rotations_xyzw=root_rotations,
            root_linear_velocities=root_linear_velocities,
            root_angular_velocities=root_angular_velocities,
        )

    def prepare_motion(self, motion) -> FloatArray:
        """Precompute frame velocities consumed by MotionLibrary sampling."""

        if isinstance(motion, ArticulationMotion):
            motion = ArticulationMotionMapper(self._layout).to_skeleton_motion(motion)
        return self.pack_skeleton_motion(motion).joint_velocities

    def pack_sample(
        self,
        *,
        root_positions: torch.Tensor,
        root_rotations_wxyz: torch.Tensor,
        root_linear_velocities: torch.Tensor,
        root_angular_velocities: torch.Tensor,
        local_rotations_wxyz: torch.Tensor,
        backend_frame_data: torch.Tensor,
        next_backend_frame_data: torch.Tensor,
        blend: torch.Tensor,
    ) -> PhysXMotionTensorSample:
        """Project sampled local quaternions into PhysX logical DOFs."""

        del next_backend_frame_data, blend
        shape = local_rotations_wxyz.shape[:-2] + (len(self._scalar_blocks),)
        joint_positions = torch.empty(
            shape,
            dtype=local_rotations_wxyz.dtype,
            device=local_rotations_wxyz.device,
        )
        for group in self._scalar_groups:
            body_index = group[0][1].body_index
            reference = torch.as_tensor(
                np.asarray(group[0][1].reference_rotation, dtype=np.float32),
                dtype=local_rotations_wxyz.dtype,
                device=local_rotations_wxyz.device,
            )
            relative = quat_wxyz_multiply(
                quat_wxyz_conjugate(reference),
                local_rotations_wxyz[..., body_index, :],
            )
            rotation_vector = quat_wxyz_to_rotation_vector(relative)
            for index, block in group:
                axis = torch.as_tensor(
                    np.asarray(block.axis, dtype=np.float32),
                    dtype=local_rotations_wxyz.dtype,
                    device=local_rotations_wxyz.device,
                )
                axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-8)
                joint_positions[..., index] = torch.sum(rotation_vector * axis, dim=-1)

        has_root = self._free_block is not None
        return PhysXMotionTensorSample(
            joint_positions=joint_positions,
            joint_velocities=backend_frame_data,
            root_positions=root_positions if has_root else None,
            root_rotations_xyzw=(
                root_rotations_wxyz[..., [1, 2, 3, 0]] if has_root else None
            ),
            root_linear_velocities=root_linear_velocities if has_root else None,
            root_angular_velocities=root_angular_velocities if has_root else None,
        )

    def unpack(
        self,
        joint_positions: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        *,
        fps: float,
        root_positions: npt.ArrayLike | None = None,
        root_rotations_xyzw: npt.ArrayLike | None = None,
        root_linear_velocities: npt.ArrayLike | None = None,
        root_angular_velocities: npt.ArrayLike | None = None,
        motion_name: str = "Motion",
    ) -> ArticulationMotion:
        """Convert frame-major PhysX root and logical DOF state to canonical motion."""

        positions = np.asarray(joint_positions, dtype=np.float32)
        velocities = np.asarray(joint_velocities, dtype=np.float32)
        expected_dofs = len(self._scalar_blocks)
        if positions.ndim != 2 or positions.shape[1] != expected_dofs:
            raise ValueError(
                f"joint_positions expected shape [frames, {expected_dofs}]"
            )
        if velocities.shape != positions.shape:
            raise ValueError("joint_velocities must match joint_positions shape")

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
                "root_positions": root_positions,
                "root_rotations_xyzw": root_rotations_xyzw,
                "root_linear_velocities": root_linear_velocities,
                "root_angular_velocities": root_angular_velocities,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "free-root PhysX unpack requires " + ", ".join(missing)
                )
            root_position_values = np.asarray(root_positions, dtype=np.float32)
            root_rotation_values = np.asarray(root_rotations_xyzw, dtype=np.float32)
            root_linear_values = np.asarray(root_linear_velocities, dtype=np.float32)
            root_angular_values = np.asarray(root_angular_velocities, dtype=np.float32)
            frames = positions.shape[0]
            expected_shapes = {
                "root_positions": (frames, 3),
                "root_rotations_xyzw": (frames, 4),
                "root_linear_velocities": (frames, 3),
                "root_angular_velocities": (frames, 3),
            }
            actual_values = {
                "root_positions": root_position_values,
                "root_rotations_xyzw": root_rotation_values,
                "root_linear_velocities": root_linear_values,
                "root_angular_velocities": root_angular_values,
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
