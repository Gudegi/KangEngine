"""BVH serialization from KangEngine skeleton motions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..utils import quat_wxyz_to_matrix


def _safe_joint_names(names: list[str]) -> list[str]:
    used: set[str] = set()
    output = []
    for index, name in enumerate(names):
        token = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in str(name)
        ).strip("_")
        token = token or f"joint_{index}"
        candidate = token
        suffix = 1
        while candidate in used:
            candidate = f"{token}_{suffix}"
            suffix += 1
        used.add(candidate)
        output.append(candidate)
    return output


def _zyx_degrees(rotations_wxyz: np.ndarray) -> np.ndarray:
    matrices = quat_wxyz_to_matrix(torch.from_numpy(rotations_wxyz)).numpy()
    sin_y = np.clip(-matrices[..., 2, 0], -1.0, 1.0)
    y = np.arcsin(sin_y)
    cos_y = np.cos(y)
    regular = np.abs(cos_y) > 1e-6
    x = np.where(
        regular,
        np.arctan2(matrices[..., 2, 1], matrices[..., 2, 2]),
        0.0,
    )
    z = np.where(
        regular,
        np.arctan2(matrices[..., 1, 0], matrices[..., 0, 0]),
        np.arctan2(-matrices[..., 0, 1], matrices[..., 1, 1]),
    )
    return np.rad2deg(np.stack((z, y, x), axis=-1))


def _motion_arrays(motion) -> tuple[np.ndarray, np.ndarray]:
    frames = int(motion.num_frames())
    joints = int(motion.num_joints())
    root = np.asarray(motion.root_translations_flat(), np.float32).reshape(frames, 3)
    rotations = np.asarray(motion.local_rotations_wxyz_flat(), np.float32).reshape(
        frames, joints, 4
    )
    return root, rotations


def _motion_to_bvh(motion) -> str:
    """Convert a ``SkeletonMotion`` to BVH format and return a UTF-8 string.

    The hierarchy comes from ``motion.skeleton_tree``. The root uses XYZ
    position channels and every joint uses ZYX rotation channels. Leaf joints
    have no ``End Site`` block.

    Args:
        motion: KangEngine ``animation.SkeletonMotion`` to serialize.

    Returns:
        BVH text as a Python string.
    """
    frames = int(motion.num_frames())
    joints = int(motion.num_joints())
    if frames <= 0:
        raise ValueError("motion must contain at least one frame")
    if joints <= 0:
        raise ValueError("motion must contain at least one joint")
    fps = float(motion.fps())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("motion fps must be positive")

    tree = motion.skeleton_tree
    parents = [int(value) for value in tree.parent_indices()]
    if len(parents) != joints or parents[0] != -1:
        raise ValueError("BVH export requires one root at joint index 0")
    children: list[list[int]] = [[] for _ in range(joints)]
    for joint in range(1, joints):
        parent = parents[joint]
        if parent < 0 or parent >= joint:
            raise ValueError("BVH export requires parents to precede their children")
        children[parent].append(joint)

    names = _safe_joint_names(list(tree.node_names()))
    offsets = np.asarray(
        [
            [
                tree.local_translation(joint).x,
                tree.local_translation(joint).y,
                tree.local_translation(joint).z,
            ]
            for joint in range(joints)
        ],
        dtype=np.float32,
    )

    lines = ["HIERARCHY"]

    def write_joint(joint: int, depth: int) -> None:
        indent = "\t" * depth
        kind = "ROOT" if joint == 0 else "JOINT"
        lines.append(f"{indent}{kind} {names[joint]}")
        lines.append(f"{indent}{{")
        offset = offsets[joint]
        lines.append(
            f"{indent}\tOFFSET {offset[0]:.9g} {offset[1]:.9g} {offset[2]:.9g}"
        )
        if joint == 0:
            lines.append(
                f"{indent}\tCHANNELS 6 Xposition Yposition Zposition "
                "Zrotation Yrotation Xrotation"
            )
        else:
            lines.append(f"{indent}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        for child in children[joint]:
            write_joint(child, depth + 1)
        lines.append(f"{indent}}}")

    write_joint(0, 0)
    root_positions, local_rotations = _motion_arrays(motion)
    euler_degrees = _zyx_degrees(local_rotations)
    lines.extend(("MOTION", f"Frames: {frames}", f"Frame Time: {1.0 / fps:.12g}"))
    for frame in range(frames):
        values = [*root_positions[frame]]
        for joint in range(joints):
            values.extend(euler_degrees[frame, joint])
        lines.append(" ".join(f"{float(value):.9g}" for value in values))
    return "\n".join(lines) + "\n"


def _motion_to_bvh_bytes(motion) -> bytes:
    """Convert a ``SkeletonMotion`` to UTF-8 BVH bytes."""
    return _motion_to_bvh(motion).encode("utf-8")


def save_motion_bvh(path: str | Path, motion) -> Path:
    """Write a ``SkeletonMotion`` to a BVH file and return its path."""
    output = Path(path).expanduser()
    output.write_bytes(_motion_to_bvh_bytes(motion))
    return output


__all__ = ["save_motion_bvh"]
