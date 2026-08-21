"""Name-based high-level adapters for KangEngine's native full-body IK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._core import _ke

if TYPE_CHECKING:
    from . import SkeletonMotion


solve_full_body_ik = _ke.animation.solve_full_body_ik
solve_full_body_ik_batch = _ke.animation.solve_full_body_ik_batch


@dataclass(frozen=True)
class IKChain:
    """One end effector and its locally expressed joint control axes."""

    effector_joint: str
    joints: tuple[tuple[str, tuple[float, float, float]], ...]
    effector_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class IKSolveResult:
    motion: SkeletonMotion
    body_positions: np.ndarray
    final_errors: np.ndarray
    iterations: np.ndarray


def solve_ik_motion(
    motion: SkeletonMotion,
    target_positions: npt.ArrayLike,
    chains: tuple[IKChain, ...],
    *,
    max_iterations: int = 20,
    control_root_rotation: bool = True,
) -> IKSolveResult:
    """Solve named full-body IK chains for every frame using native L-BFGS."""
    targets = np.ascontiguousarray(target_positions, dtype=np.float32)
    if targets.shape != (motion.num_frames(), len(chains), 3):
        raise ValueError("target_positions must have shape [frames, chains, 3]")
    if not chains:
        raise ValueError("chains must not be empty")
    names = tuple(motion.node_names())
    name_to_index = {name: index for index, name in enumerate(names)}
    unknown = sorted(
        {
            name
            for chain in chains
            for name in (chain.effector_joint, *(item[0] for item in chain.joints))
            if name not in name_to_index
        }
    )
    if unknown:
        raise ValueError(f"unknown IK joints: {unknown}")

    effectors = np.asarray(
        [name_to_index[chain.effector_joint] for chain in chains], dtype=np.int32
    )
    offsets = np.ascontiguousarray(
        [chain.effector_offset for chain in chains], dtype=np.float32
    )
    controls: list[tuple[int, tuple[float, float, float]]] = []
    seen: set[tuple[int, tuple[float, float, float]]] = set()
    if control_root_rotation:
        root = 0
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            controls.append((root, axis))
            seen.add((root, axis))
    for chain in chains:
        for name, axis_values in chain.joints:
            axis = tuple(float(value) for value in axis_values)
            if (
                len(axis) != 3
                or not np.isfinite(axis).all()
                or np.linalg.norm(axis) <= 1e-8
            ):
                raise ValueError(f"invalid IK axis for joint {name!r}")
            item = (name_to_index[name], axis)
            if item not in seen:
                controls.append(item)
                seen.add(item)
    if not controls:
        raise ValueError("at least one IK control axis is required")

    corrected, positions, errors, iterations = _ke.animation.solve_full_body_ik_batch(
        motion,
        targets,
        effectors,
        offsets,
        np.asarray([item[0] for item in controls], dtype=np.int32),
        np.asarray([item[1] for item in controls], dtype=np.float32),
        max(0, int(max_iterations)),
    )
    return IKSolveResult(
        corrected,
        np.asarray(positions),
        np.asarray(errors),
        np.asarray(iterations),
    )
