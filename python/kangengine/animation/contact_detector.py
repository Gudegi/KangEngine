"""Kinematic contact detection for stationary support surfaces."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _contact_mask(heights, velocities, height_threshold, velocity_threshold):
    return (heights <= height_threshold) & (
        np.linalg.norm(velocities, axis=-1) <= velocity_threshold
    )


class FootContactDetector:
    """Estimate contact from surface distance and speed, with hysteresis.

    Contact points retain the same count and order between updates. Ground
    queries and pose extraction belong to the caller; no motion clip is needed.
    """

    def __init__(
        self,
        *,
        height_threshold: float = 0.045,
        velocity_threshold: float = 0.35,
        release_height: float = 0.065,
        release_velocity: float = 0.5,
    ) -> None:
        """Set entry and release limits in meters and m/s.

        Release limits must be at least their corresponding entry limits.
        """
        if not (
            0 <= height_threshold <= release_height
            and 0 <= velocity_threshold <= release_velocity
        ):
            raise ValueError(
                "release limits must be at least the nonnegative entry limits"
            )
        self.height_threshold = height_threshold
        self.velocity_threshold = velocity_threshold
        self.release_height = release_height
        self.release_velocity = release_velocity
        self.reset()

    def reset(self) -> None:
        """Clear contact state after a reset, teleport, or change of tracked points."""
        self._contacts = False

    def update(
        self,
        positions: npt.ArrayLike,
        velocities: npt.ArrayLike,
        *,
        ground_points: npt.ArrayLike = (0.0, 0.0, 0.0),
        ground_normals: npt.ArrayLike = (0.0, 0.0, 1.0),
        supported: npt.ArrayLike = True,
    ) -> npt.NDArray[np.bool_]:
        """Update contact estimates using the current support surface.

        Args:
            positions: World contact-point positions [K,3], in meters.
            velocities: World velocities [K,3], in m/s. Supply measured velocities,
                or differences between successive points divided by their dt.
            ground_points: Points on the support surface, [3] or [K,3].
            ground_normals: Unit surface normals toward the character, [3] or [K,3].
                Defaults to a Z-up plane at zero height.
            supported: Ground-query hit mask, scalar or [K]. False releases contact.
                Missed queries may use NaN points/normals with supported=False.

        Returns:
            Independent contact flags [K]. Negative surface distances count as
            below the height limit; this does not resolve penetration or sliding.
        """
        heights = np.sum(
            (np.asarray(positions) - ground_points) * ground_normals, axis=-1
        )
        height_limit = np.where(
            self._contacts, self.release_height, self.height_threshold
        )
        velocity_limit = np.where(
            self._contacts, self.release_velocity, self.velocity_threshold
        )
        self._contacts = np.asarray(supported, dtype=bool) & _contact_mask(
            heights, velocities, height_limit, velocity_limit
        )
        return self._contacts.copy()
