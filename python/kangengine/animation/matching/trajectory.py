"""Walking trajectory prediction and query construction from placed playback."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ..._core import _ke
from ...utils.math import (
    quat_wxyz_from_rotation_vector_numpy,
    quat_wxyz_rotate_numpy,
)
from .controller import MotionMatchingController


def predict_walking_trajectory(
    controller: MotionMatchingController,
    *,
    desired_velocity: npt.ArrayLike,
    facing_angle: float,
    half_life: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world positions/directions at time 0 and DB trajectory horizons.

    Starts from the placed reference (root if absent), planar velocity, yaw rate,
    and zero acceleration; leaves controller state unchanged.

    Args:
        controller: Current playback and DB trajectory horizons.
        desired_velocity: Planar world velocity [3] in m/s.
        facing_angle: Radians about up_axis, measured from forward_axis in world space.
        half_life: Spring decay in seconds; zero snaps future velocity and facing.

    Returns:
        World positions in meters and unit directions, each [H+1,3]. The first
        sample preserves the current pose; the rest follow DB horizon order.
    """
    config = controller.database.config
    up_index = "XYZ".index(config.up_axis)
    velocity = np.asarray(desired_velocity, dtype=np.float64)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("desired_velocity must contain three finite values")
    if abs(float(velocity[up_index])) > 1e-6:
        raise ValueError("desired_velocity must lie in the configured ground plane")
    if not math.isfinite(facing_angle):
        raise ValueError("facing_angle must be finite radians")
    up = np.eye(3)[up_index]
    forward = np.eye(3)["XYZ".index(config.forward_axis)]
    current_velocity = controller.reference_linear_velocity.copy()
    current_velocity[up_index] = 0
    times = np.array((0.0, *config.trajectory_times))
    positions, _, _ = _ke.animation.trajectory_spring_position(
        position=controller.reference_position,
        velocity=current_velocity,
        acceleration=(0, 0, 0),
        desired_velocity=velocity,
        half_life=half_life,
        dt=times,
    )
    rotations, _ = _ke.animation.trajectory_spring_rotation(
        rotation_wxyz=controller.reference_rotation_wxyz,
        angular_velocity=up * controller.reference_angular_velocity[up_index],
        desired_rotation_wxyz=quat_wxyz_from_rotation_vector_numpy(facing_angle * up),
        half_life=half_life,
        dt=times,
    )
    return positions, quat_wxyz_rotate_numpy(rotations, forward)


def make_walking_query(
    controller: MotionMatchingController,
    *,
    desired_velocity: npt.ArrayLike,
    facing_angle: float,
    half_life: float,
) -> np.ndarray:
    """Encode a predicted walking trajectory in the current reference frame.

    Use with step_with_query to refresh prediction at each search.

    Args:
        controller: Current playback and DB feature policy.
        desired_velocity: Planar world velocity [3] in m/s.
        facing_angle: Radians about up_axis, measured from forward_axis in world space.
        half_life: Spring decay in seconds; zero snaps future velocity and facing.

    Returns:
        Normalized, weighted features [D] ready for database.search().
    """
    positions, directions = predict_walking_trajectory(
        controller,
        desired_velocity=desired_velocity,
        facing_angle=facing_angle,
        half_life=half_life,
    )
    heading = controller.reference_rotation_wxyz
    return controller.database.make_query(
        controller.motion_index,
        controller.time,
        trajectory_positions=quat_wxyz_rotate_numpy(
            heading, positions[1:] - controller.reference_position, inverse=True
        ),
        trajectory_directions=quat_wxyz_rotate_numpy(
            heading, directions[1:], inverse=True
        ),
    )
