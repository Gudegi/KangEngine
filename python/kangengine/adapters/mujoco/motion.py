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
from ...utils import batched_rotations
from ...utils.math import quat_wxyz_rotate_numpy
from ._dependency import load_mujoco

if TYPE_CHECKING:
    import mujoco

    from .transition import MuJoCoMotionTransition


FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class MuJoCoMotionBuffers:
    """Frame-major MuJoCo configuration and velocity buffers."""

    qpos: FloatArray
    qvel: FloatArray


@dataclass(frozen=True)
class MuJoCoMotionState:
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
        offsets = [
            block.q_offset + (3 if block.type == ArticulationCoordinateType.FREE else 0)
            for block in self._rotation_blocks
        ]
        self._quaternion_ids = {
            torch.device("cpu"): torch.tensor(offsets, dtype=torch.long).unsqueeze(-1)
            + torch.arange(4)
        }

    @property
    def layout(self) -> ArticulationCoordinateLayout:
        return self._layout

    def _create_transition(
        self, native, buffers, num_envs, rate
    ) -> MuJoCoMotionTransition:
        from .transition import MuJoCoMotionTransition

        return MuJoCoMotionTransition(self._layout, native, buffers, num_envs, rate)

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

    def prepare_body_motion(self, data: FloatArray, *, fps: float):
        """Reconstruct robot-representable poses from prepared CPU frames."""
        motion = self.unpack(
            data[:, : self._layout.nq], data[:, self._layout.nq :], fps=fps
        )
        return ArticulationMotionMapper(self._layout).to_skeleton_motion(motion)

    def frame_buffers(self, data: torch.Tensor) -> MuJoCoMotionState:
        """Borrow typed MuJoCo frame buffers without copying."""
        return MuJoCoMotionState(
            data[..., : self._layout.nq], data[..., self._layout.nq :]
        )

    def pack_sample(
        self,
        *,
        backend_frame_data: torch.Tensor,
        next_backend_frame_data: torch.Tensor,
        blend: torch.Tensor,
    ) -> MuJoCoMotionState:
        """Interpolate prepared coordinates and native velocities."""
        first = backend_frame_data[..., : self._layout.nq]
        following = next_backend_frame_data[..., : self._layout.nq]
        qpos = first + (following - first) * blend.unsqueeze(-1)
        qvel = torch.lerp(
            backend_frame_data[..., self._layout.nq :],
            next_backend_frame_data[..., self._layout.nq :],
            blend.unsqueeze(-1),
        )

        # Reuse device indices and interpolate all quaternion joints together.
        if first.device not in self._quaternion_ids:
            self._quaternion_ids[first.device] = self._quaternion_ids[
                torch.device("cpu")
            ].to(first.device)
        ids = self._quaternion_ids[first.device]
        first_rotation = first[..., ids]
        next_rotation = following[..., ids]
        rotation = batched_rotations.quat_wxyz_slerp(
            first_rotation, next_rotation, blend.unsqueeze(-1)
        )
        qpos[..., ids] = rotation
        return MuJoCoMotionState(qpos=qpos, qvel=qvel)

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
