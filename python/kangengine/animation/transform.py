"""Spatial transforms for complete skeleton motions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from typing import TYPE_CHECKING

from .._core import _ke
from ..utils.batched_rotations import (
    quat_wxyz_multiply,
    quat_wxyz_normalize,
    quat_wxyz_to_matrix,
)

if TYPE_CHECKING:
    from . import SkeletonMotion


def _vector3(value: npt.ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


def _rotation(value: npt.ArrayLike) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float64)
    if result.shape != (4,) or not torch.isfinite(result).all():
        raise ValueError("rotation_wxyz must contain four finite values")
    if torch.linalg.vector_norm(result) <= 1.0e-12:
        raise ValueError("rotation_wxyz must not be a zero quaternion")
    return quat_wxyz_normalize(result)


def transform_motion(
    motion: SkeletonMotion,
    *,
    rotation_wxyz: npt.ArrayLike = (1.0, 0.0, 0.0, 0.0),
    translation: npt.ArrayLike = (0.0, 0.0, 0.0),
    pivot: npt.ArrayLike = (0.0, 0.0, 0.0),
) -> SkeletonMotion:
    """Return ``motion`` transformed rigidly in world space.

    Root positions are rotated about ``pivot`` and then translated. The same
    world rotation is left-multiplied onto the root joint. Skeleton offsets and
    non-root local rotations are unchanged.
    """
    if not hasattr(motion, "root_translations") or not hasattr(
        motion, "local_rotations_wxyz"
    ):
        raise TypeError("motion must be a SkeletonMotion")

    rotation = _rotation(rotation_wxyz)
    rotation_matrix = quat_wxyz_to_matrix(rotation).cpu().numpy()
    offset = _vector3(translation, "translation")
    center = _vector3(pivot, "pivot")

    roots = np.asarray(motion.root_translations(), dtype=np.float64)
    transformed_roots = (roots - center) @ rotation_matrix.T + center + offset

    local_rotations = torch.as_tensor(
        np.asarray(motion.local_rotations_wxyz()), dtype=torch.float64
    ).clone()
    local_rotations[:, 0] = quat_wxyz_normalize(
        quat_wxyz_multiply(rotation, local_rotations[:, 0])
    )

    return _ke.animation.SkeletonMotion.from_arrays(
        motion.skeleton_tree,
        transformed_roots.astype(np.float32),
        local_rotations.to(torch.float32),
        motion.fps(),
        motion.motion_name(),
    )
