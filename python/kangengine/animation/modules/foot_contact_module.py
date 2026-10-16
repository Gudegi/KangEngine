"""Foot contact tracking from successive skeleton poses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..contact_detector import FootContactDetector

if TYPE_CHECKING:
    from .. import SkeletonState, SkeletonTree


class FootContactModule:
    """Track foot probes and contact flags independently of playback and rendering.

    Pass final output poses from the same skeleton, after any pose corrections.
    Positions and contacts are the latest results, in the configured joint order.
    """

    def __init__(
        self,
        skeleton_tree: SkeletonTree,
        joints: Sequence[str],
        *,
        offsets: npt.ArrayLike | None = None,
        detector: FootContactDetector | None = None,
    ) -> None:
        """Select probes by joint name, allowing repeated joints for heel/toe points.

        Args:
            offsets: Joint-local probe offsets [K,3], in meters. Defaults to zero.
            detector: Optional detector with custom thresholds, owned by this module.
                Do not share its contact state between characters.
        """
        self._indices = [skeleton_tree.index(name) for name in joints]
        self._offsets = (
            np.zeros((len(joints), 3)) if offsets is None else np.array(offsets, copy=True)
        )
        self.detector = FootContactDetector() if detector is None else detector
        self.positions: npt.NDArray | None = None
        self.contacts = np.zeros(len(joints), dtype=bool)

    def _positions(self, state: SkeletonState) -> np.ndarray:
        transforms = state.compute_global_matrices()[self._indices]
        return transforms[:, :3, 3] + np.einsum(
            "kij,kj->ki", transforms[:, :3, :3], self._offsets
        )

    def reset(self, state: SkeletonState | None = None) -> None:
        """Clear contacts and optionally seed the position baseline after a teleport."""
        self.detector.reset()
        self.contacts[:] = False
        self.positions = None if state is None else self._positions(state)

    def update(
        self,
        state: SkeletonState,
        dt: float,
        *,
        ground_points: npt.ArrayLike = (0.0, 0.0, 0.0),
        ground_normals: npt.ArrayLike = (0.0, 0.0, 1.0),
        supported: npt.ArrayLike = True,
    ) -> npt.NDArray[np.bool_]:
        """Update world probe positions and contact flags for a new output pose.

        Args:
            dt: Positive seconds since the preceding pose; skip updates while paused.
            ground_points: Surface points [3] or [K,3], in world meters.
            ground_normals: Unit world normals [3] or [K,3]. Defaults to Z-up.
            supported: Ground-query hit mask, scalar or [K].

        Returns:
            Contact flags [K]. Without a baseline, the first update only records
            positions and returns false flags. Do not modify the stored results.
        """
        positions = self._positions(state)
        if self.positions is not None:
            self.contacts = self.detector.update(
                positions=positions,
                velocities=(positions - self.positions) / dt,
                ground_points=ground_points,
                ground_normals=ground_normals,
                supported=supported,
            )
        self.positions = positions
        return self.contacts
