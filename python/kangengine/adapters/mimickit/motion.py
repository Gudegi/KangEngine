"""Import trusted MimicKit pickle clips as canonical articulation motion."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch

from ...animation.articulation_motion import (
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
    ArticulationMotion,
)
from ...utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_from_angle_axis,
    quat_wxyz_multiply,
    quat_wxyz_to_rotation_vector,
)


def load_articulation_motion(
    path: str | Path,
    layout: ArticulationCoordinateLayout,
    *,
    motion_name: str | None = None,
) -> ArticulationMotion:
    """Load one trusted MimicKit root-exp-map plus joint-DOF pickle clip."""

    resolved_path = Path(path).expanduser().resolve()
    with resolved_path.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 -- MimicKit's native format.
    if not isinstance(payload, dict) or not {"fps", "frames"} <= payload.keys():
        raise ValueError("MimicKit motion must contain fps and frames")
    frame_data = np.asarray(payload["frames"], dtype=np.float32)
    if frame_data.ndim != 2 or frame_data.shape[0] < 1 or frame_data.shape[1] < 6:
        raise ValueError("MimicKit frames expected shape [frames, 6 + dofs]")
    if not np.isfinite(frame_data).all():
        raise ValueError(f"MimicKit motion contains non-finite values: {resolved_path}")
    fps = float(payload["fps"])
    if fps <= 0.0:
        raise ValueError("MimicKit motion fps must be positive")

    root_blocks = [
        block
        for block in layout.blocks
        if block.type == ArticulationCoordinateType.FREE
    ]
    joint_blocks = [
        block
        for block in layout.blocks
        if block.type != ArticulationCoordinateType.FIXED
        and block.type != ArticulationCoordinateType.FREE
    ]
    if len(root_blocks) != 1:
        raise ValueError("MimicKit motion requires one canonical free root")
    if any(
        block.type != ArticulationCoordinateType.REVOLUTE
        or block.q_size != 1
        or block.qd_size != 1
        for block in joint_blocks
    ):
        raise NotImplementedError(
            "MimicKit pickle import currently supports scalar hinge characters"
        )
    if frame_data.shape[1] != 6 + len(joint_blocks):
        raise ValueError(
            f"MimicKit motion has {frame_data.shape[1] - 6} joint DOFs, "
            f"layout expects {len(joint_blocks)}"
        )

    q = np.zeros((frame_data.shape[0], layout.nq), dtype=np.float32)
    qd = np.zeros((frame_data.shape[0], layout.nv), dtype=np.float32)
    root_block = root_blocks[0]
    q[:, root_block.q_offset : root_block.q_offset + 3] = frame_data[:, :3]
    rotation_vector = torch.from_numpy(frame_data[:, 3:6])
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
    root_rotations = quat_wxyz_from_angle_axis(angle, rotation_vector)
    q[:, root_block.q_offset + 3 : root_block.q_offset + 7] = root_rotations.numpy()
    for frame_column, block in enumerate(joint_blocks, start=6):
        q[:, block.q_offset] = frame_data[:, frame_column]

    if frame_data.shape[0] > 1:
        qd[:-1, root_block.qd_offset : root_block.qd_offset + 3] = (
            frame_data[1:, :3] - frame_data[:-1, :3]
        ) * fps
        delta = quat_wxyz_multiply(
            root_rotations[1:], quat_wxyz_conjugate(root_rotations[:-1])
        )
        qd[:-1, root_block.qd_offset + 3 : root_block.qd_offset + 6] = (
            quat_wxyz_to_rotation_vector(delta).numpy() * fps
        )
        for block in joint_blocks:
            difference = (
                np.remainder(
                    q[1:, block.q_offset] - q[:-1, block.q_offset] + np.pi,
                    2.0 * np.pi,
                )
                - np.pi
            )
            qd[:-1, block.qd_offset] = difference * fps
        qd[-1] = qd[-2]

    return ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=fps,
        motion_name=resolved_path.stem if motion_name is None else motion_name,
    )
