"""Coordinate-system conversion for skeleton animation data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .._core import _ke
from ..utils import CoordinateSystem, coordinate_conversion_matrix
from ..utils.batched_rotations import (
    matrix_to_quat_wxyz,
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_normalize,
)

if TYPE_CHECKING:
    from . import SkeletonMotion, SkeletonState, SkeletonTree


def _convert_vectors(values: object, basis: torch.Tensor) -> torch.Tensor:
    vectors = torch.as_tensor(values, dtype=torch.float64)
    return torch.matmul(vectors, basis.T)


def _convert_rotations(values: object, basis: torch.Tensor) -> torch.Tensor:
    rotations = quat_wxyz_normalize(torch.as_tensor(values, dtype=torch.float64))
    basis_quaternion = matrix_to_quat_wxyz(basis)
    return quat_wxyz_normalize(
        quat_wxyz_multiply(
            quat_wxyz_multiply(basis_quaternion, rotations),
            quat_wxyz_conjugate(basis_quaternion),
        )
    )


def convert_skeleton_coordinates(
    tree: SkeletonTree,
    *,
    source: CoordinateSystem,
    target: CoordinateSystem,
) -> SkeletonTree:
    """Return a skeleton expressed in another coordinate system."""
    basis = torch.from_numpy(coordinate_conversion_matrix(source, target)).to(
        torch.float64
    )
    translations = np.asarray(
        [
            (
                tree.local_translation(i).x,
                tree.local_translation(i).y,
                tree.local_translation(i).z,
            )
            for i in range(tree.num_joints())
        ],
        dtype=np.float64,
    )
    rotations = np.asarray(
        [
            (
                tree.local_rotation(i).w,
                tree.local_rotation(i).x,
                tree.local_rotation(i).y,
                tree.local_rotation(i).z,
            )
            for i in range(tree.num_joints())
        ],
        dtype=np.float64,
    )
    return _ke.animation.SkeletonTree(
        tree.node_names(),
        tree.parent_indices(),
        _convert_vectors(translations, basis).to(torch.float32),
        _convert_rotations(rotations, basis).to(torch.float32),
    )


def convert_state_coordinates(
    state: SkeletonState,
    *,
    source: CoordinateSystem,
    target: CoordinateSystem,
) -> SkeletonState:
    """Return a pose and its skeleton expressed in another coordinate system."""
    basis = torch.from_numpy(coordinate_conversion_matrix(source, target)).to(
        torch.float64
    )
    tree = convert_skeleton_coordinates(
        state.skeleton_tree, source=source, target=target
    )
    rotations = np.asarray(
        [
            (
                state.rotation(i).w,
                state.rotation(i).x,
                state.rotation(i).y,
                state.rotation(i).z,
            )
            for i in range(state.num_joints())
        ],
        dtype=np.float64,
    )
    root = state.root_translation()
    return _ke.animation.SkeletonState.from_rotation_and_root_translation(
        tree,
        _convert_rotations(rotations, basis).to(torch.float32),
        _convert_vectors((root.x, root.y, root.z), basis).to(torch.float32),
        state.is_local(),
    )


def convert_motion_coordinates(
    motion: SkeletonMotion,
    *,
    source: CoordinateSystem,
    target: CoordinateSystem,
) -> SkeletonMotion:
    """Return a complete motion expressed in another coordinate system."""
    basis = torch.from_numpy(coordinate_conversion_matrix(source, target)).to(
        torch.float64
    )
    tree = convert_skeleton_coordinates(
        motion.skeleton_tree, source=source, target=target
    )
    return _ke.animation.SkeletonMotion.from_arrays(
        tree,
        _convert_vectors(motion.root_translations(), basis).to(torch.float32),
        _convert_rotations(motion.local_rotations_wxyz(), basis).to(torch.float32),
        motion.fps(),
        motion.motion_name(),
    )
