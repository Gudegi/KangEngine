"""Conversion between canonical articulation motion and MuJoCo buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from ...animation.articulation_motion import (
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
    ArticulationMotion,
    ArticulationMotionMapper,
)
from ...utils.batched_rotations import quat_wxyz_conjugate, quat_wxyz_multiply
from ...utils.math import quat_wxyz_rotate_numpy
from ._dependency import load_mujoco

if TYPE_CHECKING:
    import mujoco


FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class MuJoCoMotionBuffers:
    """Frame-major MuJoCo configuration and velocity buffers."""

    qpos: FloatArray
    qvel: FloatArray


@dataclass(frozen=True)
class MuJoCoMotionTensorSample:
    """Sample-major MuJoCo qpos/qvel tensors."""

    qpos: torch.Tensor
    qvel: torch.Tensor


class MuJoCoMotionAdapter:
    """Convert a reference trajectory with matching MuJoCo coordinate order."""

    def __init__(self, layout: ArticulationCoordinateLayout) -> None:
        self._layout = layout
        self._model_signature = layout.model_signature
        self._rotation_blocks = tuple(
            block
            for block in layout.blocks
            if block.type
            in (
                ArticulationCoordinateType.FREE,
                ArticulationCoordinateType.SPHERICAL,
            )
        )
        self._revolute_blocks = tuple(
            block
            for block in layout.blocks
            if block.type == ArticulationCoordinateType.REVOLUTE
        )

    @property
    def layout(self) -> ArticulationCoordinateLayout:
        return self._layout

    def validate_model(self, model: mujoco.MjModel, *, strict: bool = True) -> None:
        """Verify that a compiled MjModel uses the canonical offsets."""

        mujoco = load_mujoco()
        if int(model.nq) != self._layout.nq or int(model.nv) != self._layout.nv:
            raise ValueError(
                f"MuJoCo model layout is {model.nq}/{model.nv}, expected "
                f"{self._layout.nq}/{self._layout.nv}"
            )

        joints_by_name = {
            str(model.joint(joint_id).name): joint_id
            for joint_id in range(int(model.njnt))
        }
        mapped_names = set()
        free_ids = [
            joint_id
            for joint_id in range(int(model.njnt))
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        for block in self._layout.blocks:
            if block.type == ArticulationCoordinateType.FIXED:
                continue
            if block.type == ArticulationCoordinateType.FREE:
                if len(free_ids) != 1:
                    raise ValueError(
                        "canonical free root requires exactly one MuJoCo free joint"
                    )
                joint_id = free_ids[0]
                name = str(model.joint(joint_id).name)
            else:
                name = block.joint_name
                try:
                    joint_id = joints_by_name[name]
                except KeyError as error:
                    raise ValueError(
                        f"MuJoCo model has no joint named {name!r}"
                    ) from error

            expected_type = {
                ArticulationCoordinateType.FREE: int(mujoco.mjtJoint.mjJNT_FREE),
                ArticulationCoordinateType.SPHERICAL: int(mujoco.mjtJoint.mjJNT_BALL),
                ArticulationCoordinateType.PRISMATIC: int(mujoco.mjtJoint.mjJNT_SLIDE),
                ArticulationCoordinateType.REVOLUTE: int(mujoco.mjtJoint.mjJNT_HINGE),
            }[block.type]
            if int(model.jnt_type[joint_id]) != expected_type:
                raise ValueError(f"MuJoCo joint {name!r} type differs from layout")
            if (
                int(model.jnt_qposadr[joint_id]) != block.q_offset
                or int(model.jnt_dofadr[joint_id]) != block.qd_offset
            ):
                raise ValueError(
                    f"MuJoCo joint {name!r} offsets differ from canonical layout"
                )
            mapped_names.add(name)

        if strict:
            extra = set(joints_by_name) - mapped_names
            if extra:
                raise ValueError(
                    "MuJoCo model contains joints absent from canonical layout: "
                    + ", ".join(sorted(extra))
                )

    def pack(self, motion: ArticulationMotion) -> MuJoCoMotionBuffers:
        """Convert canonical motion into frame-major MuJoCo qpos/qvel."""

        self._validate_motion(motion)
        qpos = np.asarray(motion.q, dtype=np.float32).copy()
        qvel = np.asarray(motion.qd, dtype=np.float32).copy()
        for block in self._rotation_blocks:
            rotation_offset = (
                block.q_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.q_offset
            )
            angular_offset = (
                block.qd_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.qd_offset
            )
            quaternion = qpos[:, rotation_offset : rotation_offset + 4]
            angular = qvel[:, angular_offset : angular_offset + 3]
            qvel[:, angular_offset : angular_offset + 3] = quat_wxyz_rotate_numpy(
                quaternion, angular, inverse=True
            )
        return MuJoCoMotionBuffers(qpos=qpos, qvel=qvel)

    def prepare_motion(self, motion) -> FloatArray:
        """Precompute MuJoCo qpos/qvel frames used by MotionLibrary."""

        articulation_motion = (
            motion
            if isinstance(motion, ArticulationMotion)
            else ArticulationMotionMapper(self._layout)
            .to_articulation_motion(motion)
            .motion
        )
        buffers = self.pack(articulation_motion)
        return np.concatenate((buffers.qpos, buffers.qvel), axis=1)

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
    ) -> MuJoCoMotionTensorSample:
        """Convert sampled quaternion poses to MuJoCo qpos/qvel tensors."""

        first_qpos = backend_frame_data[..., : self._layout.nq]
        next_qpos = next_backend_frame_data[..., : self._layout.nq]
        qpos = first_qpos + (next_qpos - first_qpos) * blend.unsqueeze(-1)
        qvel = backend_frame_data[..., self._layout.nq :].clone()
        for block in self._revolute_blocks:
            delta = (
                torch.remainder(
                    next_qpos[..., block.q_offset]
                    - first_qpos[..., block.q_offset]
                    + torch.pi,
                    2.0 * torch.pi,
                )
                - torch.pi
            )
            qpos[..., block.q_offset] = first_qpos[..., block.q_offset] + blend * delta

        for block in self._rotation_blocks:
            if block.type == ArticulationCoordinateType.FREE:
                qpos[..., block.q_offset : block.q_offset + 3] = root_positions
                quaternion = root_rotations_wxyz
                qvel[..., block.qd_offset : block.qd_offset + 3] = (
                    root_linear_velocities
                )
                world_angular = root_angular_velocities
            else:
                reference = torch.as_tensor(
                    np.asarray(block.reference_rotation, dtype=np.float32),
                    dtype=local_rotations_wxyz.dtype,
                    device=local_rotations_wxyz.device,
                )
                quaternion = quat_wxyz_multiply(
                    quat_wxyz_conjugate(reference),
                    local_rotations_wxyz[..., block.body_index, :],
                )
                angular_offset = block.qd_offset
                world_angular = qvel[
                    ..., angular_offset : angular_offset + block.qd_size
                ]
            rotation_offset = (
                block.q_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.q_offset
            )
            angular_offset = (
                block.qd_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.qd_offset
            )
            qpos[..., rotation_offset : rotation_offset + 4] = quaternion
            if block.type == ArticulationCoordinateType.FREE:
                vector = torch.cat(
                    (torch.zeros_like(world_angular[..., :1]), world_angular), dim=-1
                )
                local = quat_wxyz_multiply(
                    quat_wxyz_multiply(quat_wxyz_conjugate(quaternion), vector),
                    quaternion,
                )[..., 1:]
                qvel[..., angular_offset : angular_offset + 3] = local
        return MuJoCoMotionTensorSample(qpos=qpos, qvel=qvel)

    def unpack(
        self,
        qpos: npt.ArrayLike,
        qvel: npt.ArrayLike,
        *,
        fps: float,
        motion_name: str = "Motion",
    ) -> ArticulationMotion:
        """Convert frame-major MuJoCo qpos/qvel into canonical motion."""

        q = np.asarray(qpos, dtype=np.float32)
        qd = np.asarray(qvel, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self._layout.nq:
            raise ValueError(f"qpos expected shape [frames, {self._layout.nq}]")
        if qd.shape != (q.shape[0], self._layout.nv):
            raise ValueError(f"qvel expected shape [frames, {self._layout.nv}]")
        q = q.copy()
        qd = qd.copy()
        for block in self._rotation_blocks:
            rotation_offset = (
                block.q_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.q_offset
            )
            angular_offset = (
                block.qd_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.qd_offset
            )
            quaternion = q[:, rotation_offset : rotation_offset + 4]
            angular = qd[:, angular_offset : angular_offset + 3]
            qd[:, angular_offset : angular_offset + 3] = quat_wxyz_rotate_numpy(
                quaternion, angular
            )
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
