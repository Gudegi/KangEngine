"""Python trajectory orchestration with native search and inertial playback."""

from __future__ import annotations

import math
from collections import deque
from operator import index

import numpy as np
import numpy.typing as npt

from ..._core import _ke
from ...utils.math import (
    quat_wxyz_multiply_numpy,
    quat_wxyz_rotate_numpy,
    quat_wxyz_to_rotation_vector_numpy,
)
from .database import MotionMatch, MotionMatchingDatabase


class MotionMatchingController:
    """Select clips at a fixed cadence and return native ``SkeletonState`` poses.

    - Owns a native player and shares the DB; search defaults to C++.
    - step uses planar world velocity (m/s) and facing; step_with_query accepts
      custom queries.
    - Placement aligns the DB reference track or source root by default.
    """

    def __init__(
        self,
        database: MotionMatchingDatabase,
        *,
        search_interval: float = 0.15,
        continuation_bias: float = 0.01,
        transition_half_life: float = 0.1,
        history_size: int = 10,
        search_backend: str = "native",
        root_placement=None,
    ) -> None:
        """Create independent playback over a shared database.

        Args:
            database: Source clips and precomputed feature index.
            search_interval: Positive time between searches, in seconds.
            continuation_bias: Nonnegative cost reduction favoring continuation.
            transition_half_life: Smoothing decay in seconds; zero disables it.
            history_size: Output poses retained, including the current pose; at least 2.
            search_backend: "native" or the "python" reference search.
            root_placement: Optional callback (source_state, current_state) returning
                a rigid (rotation_wxyz [4], translation [3] in meters) from source
                to world, held until the next transition. current_state is None on
                reset. Must not re-enter playback.
        """
        if not isinstance(database, MotionMatchingDatabase):
            raise TypeError("database must be a MotionMatchingDatabase")
        if not math.isfinite(search_interval) or search_interval <= 0:
            raise ValueError("search_interval must be finite and positive")
        if not math.isfinite(continuation_bias) or continuation_bias < 0:
            raise ValueError("continuation_bias must be finite and nonnegative")
        if search_backend not in ("native", "python"):
            raise ValueError("search_backend must be 'native' or 'python'")
        history_size = index(history_size)
        if history_size < 2:
            raise ValueError("history_size must be at least 2")
        self._history = deque(maxlen=history_size)
        self._search_backend = search_backend
        self.root_placement = root_placement
        self.database = database
        self.search_interval = float(search_interval)
        self.continuation_bias = float(continuation_bias)
        self._up = "XYZ".index(database.config.up_axis)
        self._forward = np.eye(3, dtype=np.float32)[
            "XYZ".index(database.config.forward_axis)
        ]
        self._player = _ke.animation.SkeletonMotionPlayer(
            database.motions,
            up_axis=database.config.up_axis,
            forward_axis=database.config.forward_axis,
            transition_half_life=transition_half_life,
        )
        self.reset()

    def reset(self, *, motion_index: int = 0, time: float = 0.0) -> None:
        """Reset placement, smoothing, and the elapsed clock.

        Args:
            motion_index: Source clip to start from.
            time: Clip-local seconds within the sampled interval.
        """
        self._place(motion_index, time, reset=True)
        self._until_search = 0.0
        self._elapsed_time = 0.0
        self._last_match: MotionMatch | None = None
        self._transition_count = 0
        state = self.state
        self._history.clear()
        self._history.append(state)
        self._joint_positions = state.compute_global_matrices()[:, :3, 3].copy()
        self._joint_velocities = np.zeros_like(self._joint_positions)

    @property
    def history(self) -> tuple[_ke.animation.SkeletonState, ...]:
        """Recent output poses, oldest first."""
        return tuple(self._history)

    @property
    def root_placement(self):
        """Placement callback; None aligns the DB reference track or source root.

        Changing this policy does not reposition the current output pose.
        """
        return self._root_placement

    @root_placement.setter
    def root_placement(self, value) -> None:
        if value is not None and not callable(value):
            raise TypeError("root_placement must be callable or None")
        self._root_placement = value

    def _place(self, motion_index: int, time: float, *, reset: bool) -> None:
        if self._root_placement is None and not self.database.has_reference_track:
            operation = self._player.reset if reset else self._player.transition
            operation(motion_index=motion_index, time=time)
            return
        # Validate before user code and pass independent pose snapshots.
        clip = self.database._clip(motion_index)
        self.database._frame_at_time(clip, time)
        if self._root_placement is not None:
            motion = self.database.motions[motion_index]
            source = motion.frame(0)
            motion.sample_into(time=time, state=source, loop=False)
            rotation, translation = self._root_placement(
                source, None if reset else self.state
            )
        else:
            # Align the reference track while preserving source Hips residual motion.
            source_rotation = self.database.reference_rotation(motion_index, time)
            source_position = self.database.reference_position(motion_index, time)
            destination_rotation = (
                (1, 0, 0, 0) if reset else self.reference_rotation_wxyz
            )
            destination_position = np.zeros(3) if reset else self.reference_position
            rotation = quat_wxyz_multiply_numpy(
                destination_rotation, source_rotation * (1, -1, -1, -1)
            )
            rotation /= np.linalg.norm(rotation)
            translation = destination_position - quat_wxyz_rotate_numpy(
                rotation, source_position
            )
        operation = (
            self._player.reset_placed if reset else self._player.transition_placed
        )
        operation(
            motion_index=motion_index,
            time=time,
            rotation_wxyz=rotation,
            translation=translation,
        )
        self._reference_alignment_rotation = np.asarray(
            rotation, dtype=np.float64
        ).copy()
        self._reference_alignment_rotation /= np.linalg.norm(
            self._reference_alignment_rotation
        )
        self._reference_alignment_translation = np.asarray(
            translation, dtype=np.float64
        ).copy()

    @property
    def search_backend(self) -> str:
        return self._search_backend

    @property
    def motion_index(self) -> int:
        return self._player.motion_index

    @property
    def elapsed_time(self) -> float:
        """Seconds advanced since reset, independent of clip switches."""
        return self._elapsed_time

    @property
    def time(self) -> float:
        """Current clip-local playback time in seconds."""
        return self._player.time

    @property
    def transition_half_life(self) -> float:
        """Inertial spring decay setting in seconds; zero disables smoothing."""
        return self._player.transition_half_life

    @transition_half_life.setter
    def transition_half_life(self, value: float) -> None:
        self._player.transition_half_life = value

    @property
    def last_match(self) -> MotionMatch | None:
        return self._last_match

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def root_position(self) -> np.ndarray:
        """Current world root position; returns an independent array."""
        return self._player.root_position

    @property
    def root_linear_velocity(self) -> np.ndarray:
        """Current smoothed world root velocity, in meters/second."""
        return self._player.root_linear_velocity

    @property
    def root_angular_velocity(self) -> np.ndarray:
        """Current smoothed world root angular velocity, in radians/second."""
        return self._player.local_angular_velocities[0]

    @property
    def heading_rotation_wxyz(self) -> np.ndarray:
        """Current world heading about the configured up axis."""
        return self._player.heading_rotation_wxyz

    @property
    def reference_position(self) -> np.ndarray:
        """World trajectory origin; falls back to the output Hips without a track.

        - Explicit tracks are independent of pose inertialization.
        - Original Hips sway and vertical motion remain intact.
        """
        if not self.database.has_reference_track:
            return self.root_position
        return (
            quat_wxyz_rotate_numpy(
                self._reference_alignment_rotation,
                self.database.reference_position(self.motion_index, self.time),
            )
            + self._reference_alignment_translation
        )

    @property
    def reference_rotation_wxyz(self) -> np.ndarray:
        """World reference orientation used by trajectory queries and alignment."""
        if not self.database.has_reference_track:
            return self.heading_rotation_wxyz
        rotation = quat_wxyz_multiply_numpy(
            self._reference_alignment_rotation,
            self.database.reference_rotation(self.motion_index, self.time),
        )
        return rotation / np.linalg.norm(rotation)

    def _reference_velocity(self, *, angular: bool) -> np.ndarray:
        clip = self.database._clip(self.motion_index)
        frame = self.time * clip.fps
        if frame >= len(clip.roots) - 1:
            return np.zeros(3)
        before = int(np.floor(frame))
        if angular:
            delta = quat_wxyz_multiply_numpy(
                clip.headings[before + 1], clip.headings[before] * (1, -1, -1, -1)
            )
            velocity = quat_wxyz_to_rotation_vector_numpy(delta) * clip.fps
        else:
            velocity = (
                clip.reference_positions[before + 1] - clip.reference_positions[before]
            ) * clip.fps
        return quat_wxyz_rotate_numpy(self._reference_alignment_rotation, velocity)

    @property
    def reference_linear_velocity(self) -> np.ndarray:
        """Right derivative of the placed reference track; zero at the clip end."""
        if not self.database.has_reference_track:
            return self.root_linear_velocity
        return self._reference_velocity(angular=False)

    @property
    def reference_angular_velocity(self) -> np.ndarray:
        """World spatial angular velocity of the placed reference track."""
        if not self.database.has_reference_track:
            return self.root_angular_velocity
        return self._reference_velocity(angular=True)

    @property
    def state(self) -> _ke.animation.SkeletonState:
        """Sample an independent native pose with accumulated root placement."""
        return self._player.state

    def make_query(self, **feature_inputs) -> np.ndarray:
        """Build a query from output positions and backward-difference velocities.

        Args:
            **feature_inputs: Builder-specific goals, such as reference-frame
                trajectory_positions/directions. The builder must accept joint_positions,
                joint_velocities, skeleton_tree, reference_position, and
                reference_rotation_wxyz, as WalkingFeatures does.

        Returns:
            Normalized, weighted query features [D].
        """
        return self.database.make_query(
            self.motion_index,
            self.time,
            joint_positions=self._joint_positions,
            joint_velocities=self._joint_velocities,
            skeleton_tree=self.database.motions[0].skeleton_tree,
            reference_position=self.reference_position,
            reference_rotation_wxyz=self.reference_rotation_wxyz,
            **feature_inputs,
        )

    def _planar_vector(self, value: npt.ArrayLike, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain three finite values")
        if abs(float(result[self._up])) > 1e-6:
            raise ValueError(f"{name} must lie in the configured ground plane")
        return result

    def _walking_query(self, velocity: np.ndarray, facing: np.ndarray) -> np.ndarray:
        heading = self.reference_rotation_wxyz
        times = np.asarray(self.database.config.trajectory_times)[:, None]
        local_positions = quat_wxyz_rotate_numpy(
            heading, times * velocity, inverse=True
        )
        local_directions = quat_wxyz_rotate_numpy(
            heading, np.broadcast_to(facing, local_positions.shape), inverse=True
        )
        return self.make_query(
            trajectory_positions=local_positions,
            trajectory_directions=local_directions,
        )

    def _select(self, query: npt.ArrayLike) -> None:
        match = self.database.search(
            query,
            current_motion_index=self.motion_index,
            current_time=self.time,
            continuation_bias=self.continuation_bias,
            backend=self._search_backend,
        )
        half_frame = 0.5 / self.database.motions[self.motion_index].fps()
        if (
            match.motion_index == self.motion_index
            and abs(match.time - self.time) <= half_frame + 1e-7
            and self.time < self._player.duration - 1e-10
        ):
            # A nearest-frame continuation is not a seek: retain subframe time.
            self._last_match = match
            return

        self._place(match.motion_index, match.time, reset=False)
        self._last_match = match
        self._transition_count += 1

    def step(
        self,
        dt: float,
        *,
        desired_velocity: npt.ArrayLike,
        desired_facing: npt.ArrayLike | None = None,
    ) -> _ke.animation.SkeletonState:
        """Advance seconds, splitting at search deadlines and clip endpoints.

        Args:
            dt: Nonnegative seconds; zero samples without searching.
            desired_velocity: Planar world velocity [3] in m/s.
            desired_facing: Nonzero planar world direction [3]. None follows
                nonzero velocity, otherwise the current heading.

        Returns:
            An independent SkeletonState with world root placement and smoothing.
        """
        velocity = self._planar_vector(desired_velocity, "desired_velocity")
        if desired_facing is None:
            facing = velocity
            if np.linalg.norm(facing) <= 1e-8:
                facing = quat_wxyz_rotate_numpy(
                    self.reference_rotation_wxyz, self._forward
                )
        else:
            facing = self._planar_vector(desired_facing, "desired_facing")
        magnitude = np.linalg.norm(facing)
        if magnitude <= 1e-8:
            raise ValueError("desired_facing must have nonzero planar length")
        facing = facing / magnitude

        return self.step_with_query(
            dt,
            query_provider=lambda controller: controller._walking_query(
                velocity, facing
            ),
        )

    def step_with_query(
        self, dt: float, *, query_provider
    ) -> _ke.animation.SkeletonState:
        """Advance using an external query rule, without planar input assumptions.

        Args:
            dt: Nonnegative seconds; zero skips the query provider.
            query_provider: Callback (controller) returning normalized, weighted
                features [D], normally via database.make_query(). Called at each search boundary,
                possibly repeatedly per step; elapsed_time is seconds since reset.
                Must not re-enter or mutate playback.

        Returns:
            An independent SkeletonState with world root placement and smoothing.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if not callable(query_provider):
            raise TypeError("query_provider must be callable")

        remaining = float(dt)
        previous_positions = self._joint_positions
        advanced = 0.0
        while remaining > 1e-12:
            duration = self._player.duration
            if self._until_search <= 1e-10 or self.time >= duration - 1e-10:
                self._select(query_provider(self))
                self._until_search = self.search_interval
                duration = self._player.duration
            advance = min(remaining, self._until_search, duration - self.time)
            self._player.advance(dt=advance)
            self._elapsed_time += advance
            self._until_search -= advance
            remaining -= advance
            advanced += advance
            state = self.state
            self._joint_positions = state.compute_global_matrices()[:, :3, 3].copy()
            self._joint_velocities = (self._joint_positions - previous_positions) / advanced
        if advanced > 0:
            self._history.append(state)
        return self.state
