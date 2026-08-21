"""Temporal filters for skeleton motion and trajectory data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from .._core import _ke
from ..utils.batched_rotations import matrix_to_quat_wxyz, quat_wxyz_to_matrix

if TYPE_CHECKING:
    from . import SkeletonMotion


def gaussian_filter_time(values: npt.ArrayLike, sigma: float) -> np.ndarray:
    """Gaussian-filter the first axis of an array."""
    source = np.asarray(values)
    if sigma <= 0.0 or len(source) < 2:
        return source.copy()
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(
        source, ((radius, radius),) + ((0, 0),) * (source.ndim - 1), mode="edge"
    )
    output = np.empty(source.shape, dtype=source.dtype, order="C")
    flattened = padded.reshape(len(padded), -1)
    filtered = output.reshape(len(output), -1)
    for channel in range(flattened.shape[1]):
        filtered[:, channel] = np.convolve(
            flattened[:, channel], kernel, mode="valid"
        )
    return output


def smooth_high_jerk(
    positions: npt.ArrayLike,
    fps: float,
    *,
    threshold: float = 2000.0,
    sigma: float = 3.0,
    context_before: int = 10,
    context_after: int = 11,
) -> np.ndarray:
    """Smooth trajectory windows whose mean third derivative is excessive."""
    source = np.asarray(positions)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("positions must have shape [frames, points, 3]")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if len(source) < 4:
        return source.copy()
    jerk = np.diff(source, n=3, axis=0) * (fps**3)
    high = np.flatnonzero(np.linalg.norm(jerk, axis=-1).mean(axis=1) >= threshold) + 1
    if len(high) == 0:
        return source.copy()
    split = np.flatnonzero(np.diff(high) > 1) + 1
    output = source.copy()
    for segment in np.split(high, split):
        start = max(0, int(segment[0]) - context_before)
        end = min(len(source), int(segment[-1]) + context_after)
        output[start:end] = gaussian_filter_time(source[start:end], sigma)
    return output


def _continuous(rotations: np.ndarray) -> None:
    for frame in range(1, len(rotations)):
        flip = np.sum(rotations[frame] * rotations[frame - 1], axis=-1) < 0.0
        rotations[frame, flip] *= -1.0


def gaussian_smooth_motion(
    motion: SkeletonMotion, sigma: float = 2.0
) -> SkeletonMotion:
    """Gaussian-smooth root and WXYZ channels while preserving unit rotations."""
    roots = gaussian_filter_time(motion.root_translations(), sigma).astype(np.float32)
    rotations = np.asarray(motion.local_rotations_wxyz(), dtype=np.float32).copy()
    _continuous(rotations)
    rotations = gaussian_filter_time(rotations, sigma)
    rotations /= np.maximum(np.linalg.norm(rotations, axis=-1, keepdims=True), 1e-8)
    return _ke.animation.SkeletonMotion.from_arrays(
        motion.skeleton_tree, roots, rotations, motion.fps(), motion.motion_name()
    )


def smooth_rotation_correction(
    reference: SkeletonMotion,
    corrected: SkeletonMotion,
    sigma: float = 1.5,
) -> SkeletonMotion:
    """Smooth only the local-rotation correction relative to a reference motion."""
    if reference.num_frames() != corrected.num_frames():
        raise ValueError("reference and corrected motions must have equal frame counts")
    if tuple(reference.node_names()) != tuple(corrected.node_names()):
        raise ValueError("reference and corrected motions must use the same skeleton")
    if sigma <= 0.0:
        return corrected
    reference_q = np.asarray(reference.local_rotations_wxyz(), np.float32)
    corrected_q = np.asarray(corrected.local_rotations_wxyz(), np.float32)
    reference_m = quat_wxyz_to_matrix(torch.from_numpy(reference_q)).numpy()
    corrected_m = quat_wxyz_to_matrix(torch.from_numpy(corrected_q)).numpy()
    delta_q = matrix_to_quat_wxyz(
        torch.from_numpy(corrected_m @ np.swapaxes(reference_m, -1, -2))
    ).numpy().astype(np.float32)
    _continuous(delta_q)
    delta_q = gaussian_filter_time(delta_q, sigma)
    delta_q /= np.maximum(np.linalg.norm(delta_q, axis=-1, keepdims=True), 1e-8)
    output_q = matrix_to_quat_wxyz(
        quat_wxyz_to_matrix(torch.from_numpy(delta_q))
        @ torch.from_numpy(reference_m)
    ).numpy().astype(np.float32)
    return _ke.animation.SkeletonMotion.from_arrays(
        corrected.skeleton_tree,
        np.asarray(corrected.root_translations(), np.float32),
        output_q,
        corrected.fps(),
        corrected.motion_name(),
    )
