"""Conversion between canonical articulation motion and Newton buffers."""

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
from ...utils.math import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from ._dependency import load_newton

if TYPE_CHECKING:
    import newton

    from .transition import NewtonMotionTransition


FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class NewtonMotionBuffers:
    """Frame-major Newton generalized coordinate and velocity buffers."""

    joint_q: FloatArray
    joint_qd: FloatArray


@dataclass(frozen=True)
class NewtonMotionState:
    """Newton generalized-coordinate tensors for stored frames or queries."""

    joint_q: torch.Tensor
    joint_qd: torch.Tensor


def _leaf_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


class NewtonMotionAdapter:
    """Convert one reference trajectory into Newton coordinate convention."""

    def __init__(self, layout: ArticulationCoordinateLayout) -> None:
        self._layout = layout
        self._model_signature = layout.model_signature
        self._quaternion_blocks = tuple(
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
            for block in self._quaternion_blocks
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
    ) -> NewtonMotionTransition:
        from .transition import NewtonMotionTransition

        return NewtonMotionTransition(self._layout, native, buffers, num_envs, rate)

    def validate_model(self, model: newton.Model, *, strict: bool = True) -> None:
        """Verify an unreplicated Newton model against the canonical layout."""

        newton, _ = load_newton()
        if int(model.world_count) != 1:
            raise ValueError("validate_model requires the unreplicated reference model")
        if (
            int(model.joint_coord_count) != self._layout.nq
            or int(model.joint_dof_count) != self._layout.nv
        ):
            raise ValueError(
                f"Newton model layout is {model.joint_coord_count}/"
                f"{model.joint_dof_count}, expected "
                f"{self._layout.nq}/{self._layout.nv}"
            )

        types = np.asarray(model.joint_type.numpy(), dtype=np.int32)
        children = np.asarray(model.joint_child.numpy(), dtype=np.int32)
        q_starts = np.asarray(model.joint_q_start.numpy(), dtype=np.int32)
        qd_starts = np.asarray(model.joint_qd_start.numpy(), dtype=np.int32)
        body_names = [_leaf_name(str(label)) for label in model.body_label]
        blocks_by_body: dict[str, list] = {}
        free_blocks = []
        for block in self._layout.blocks:
            if block.type == ArticulationCoordinateType.FIXED:
                continue
            if block.type == ArticulationCoordinateType.FREE:
                free_blocks.append(block)
            else:
                blocks_by_body.setdefault(block.body_name, []).append(block)

        mapped_bodies = set()
        for joint_id in range(int(model.joint_count)):
            q_offset = int(q_starts[joint_id])
            qd_offset = int(qd_starts[joint_id])
            q_size = int(q_starts[joint_id + 1] - q_offset)
            qd_size = int(qd_starts[joint_id + 1] - qd_offset)
            if q_size == 0 and qd_size == 0:
                continue

            joint_type = int(types[joint_id])
            if joint_type == int(newton.JointType.FREE):
                if len(free_blocks) != 1:
                    raise ValueError(
                        "Newton free joint requires one canonical free root"
                    )
                blocks = free_blocks
                body_name = blocks[0].body_name
            else:
                child = int(children[joint_id])
                if child < 0 or child >= len(body_names):
                    raise ValueError("Newton joint has invalid child body index")
                body_name = body_names[child]
                try:
                    blocks = blocks_by_body[body_name]
                except KeyError as error:
                    if strict:
                        raise ValueError(
                            f"Newton body {body_name!r} has no canonical coordinates"
                        ) from error
                    continue

            expected_q_offset = int(blocks[0].q_offset)
            expected_qd_offset = int(blocks[0].qd_offset)
            expected_q_size = sum(int(block.q_size) for block in blocks)
            expected_qd_size = sum(int(block.qd_size) for block in blocks)
            if (
                q_offset != expected_q_offset
                or qd_offset != expected_qd_offset
                or q_size != expected_q_size
                or qd_size != expected_qd_size
            ):
                raise ValueError(
                    f"Newton joint for body {body_name!r} differs from "
                    "canonical offsets"
                )

            if len(blocks) > 1:
                expected_types = {int(newton.JointType.D6)}
            else:
                expected_types = {
                    ArticulationCoordinateType.FREE: {int(newton.JointType.FREE)},
                    ArticulationCoordinateType.SPHERICAL: {int(newton.JointType.BALL)},
                    ArticulationCoordinateType.PRISMATIC: {
                        int(newton.JointType.PRISMATIC),
                        int(newton.JointType.D6),
                    },
                    ArticulationCoordinateType.REVOLUTE: {
                        int(newton.JointType.REVOLUTE),
                        int(newton.JointType.D6),
                    },
                }[blocks[0].type]
            if joint_type not in expected_types:
                raise ValueError(
                    f"Newton joint type for body {body_name!r} differs from layout"
                )

            if len(blocks) > 1:
                label = str(model.joint_label[joint_id])
                positions = [label.find(block.joint_name) for block in blocks]
                if any(position < 0 for position in positions) or positions != sorted(
                    positions
                ):
                    raise ValueError(
                        f"Newton D6 order for body {body_name!r} differs from layout"
                    )
            mapped_bodies.add(body_name)

        if strict:
            expected_bodies = set(blocks_by_body)
            if free_blocks:
                expected_bodies.add(free_blocks[0].body_name)
            missing = expected_bodies - mapped_bodies
            if missing:
                raise ValueError(
                    "canonical bodies absent from Newton model: "
                    + ", ".join(sorted(missing))
                )

    def pack(self, motion: ArticulationMotion) -> NewtonMotionBuffers:
        """Convert canonical motion into frame-major Newton joint_q/joint_qd."""

        self._validate_motion(motion)
        joint_q = np.asarray(motion.q, dtype=np.float32).copy()
        joint_qd = np.asarray(motion.qd, dtype=np.float32).copy()
        for block in self._quaternion_blocks:
            offset = (
                block.q_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.q_offset
            )
            joint_q[:, offset : offset + 4] = quat_wxyz_to_xyzw(
                joint_q[:, offset : offset + 4]
            )
        return NewtonMotionBuffers(joint_q=joint_q, joint_qd=joint_qd)

    def prepare_motion(self, motion) -> FloatArray:
        """Precompute native Newton coordinates and velocities for device sampling."""

        articulation_motion = (
            motion
            if isinstance(motion, ArticulationMotion)
            else ArticulationMotionMapper(self._layout)
            .to_articulation_motion(motion)
            .motion
        )
        buffers = self.pack(articulation_motion)
        return np.concatenate((buffers.joint_q, buffers.joint_qd), axis=1)

    def prepare_body_motion(self, data: FloatArray, *, fps: float):
        """Reconstruct robot-representable poses from prepared CPU frames."""
        motion = self.unpack(
            data[:, : self._layout.nq], data[:, self._layout.nq :], fps=fps
        )
        return ArticulationMotionMapper(self._layout).to_skeleton_motion(motion)

    def frame_buffers(self, data: torch.Tensor) -> NewtonMotionState:
        """Borrow typed Newton frame buffers without copying."""
        return NewtonMotionState(
            data[..., : self._layout.nq], data[..., self._layout.nq :]
        )

    def pack_sample(
        self,
        *,
        backend_frame_data: torch.Tensor,
        next_backend_frame_data: torch.Tensor,
        blend: torch.Tensor,
    ) -> NewtonMotionState:
        """Interpolate prepared coordinates and native velocities."""
        first = backend_frame_data[..., : self._layout.nq]
        following = next_backend_frame_data[..., : self._layout.nq]
        joint_q = first + (following - first) * blend.unsqueeze(-1)
        joint_qd = torch.lerp(
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
        rotation = batched_rotations.quat_wxyz_to_xyzw(
            batched_rotations.quat_wxyz_slerp(
                batched_rotations.quat_xyzw_to_wxyz(first_rotation),
                batched_rotations.quat_xyzw_to_wxyz(next_rotation),
                blend.unsqueeze(-1),
            )
        )
        joint_q[..., ids] = rotation
        return NewtonMotionState(joint_q=joint_q, joint_qd=joint_qd)

    def unpack(
        self,
        joint_q: npt.ArrayLike,
        joint_qd: npt.ArrayLike,
        *,
        fps: float,
        motion_name: str = "Motion",
    ) -> ArticulationMotion:
        """Convert frame-major Newton joint_q/joint_qd into canonical motion."""

        q = np.asarray(joint_q, dtype=np.float32)
        qd = np.asarray(joint_qd, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self._layout.nq:
            raise ValueError(f"joint_q expected shape [frames, {self._layout.nq}]")
        if qd.shape != (q.shape[0], self._layout.nv):
            raise ValueError(f"joint_qd expected shape [frames, {self._layout.nv}]")
        q = q.copy()
        qd = qd.copy()
        for block in self._quaternion_blocks:
            offset = (
                block.q_offset + 3
                if block.type == ArticulationCoordinateType.FREE
                else block.q_offset
            )
            q[:, offset : offset + 4] = quat_xyzw_to_wxyz(q[:, offset : offset + 4])
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
