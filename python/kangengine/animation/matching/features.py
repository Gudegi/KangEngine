"""Default walking feature policy and the custom builder contract.

- Builders implement describe(), build_clip(), and make_query().
- Clip and query features share the declared block order.
- The DB applies normalization and search weights.
"""

from __future__ import annotations
import numpy as np
from ...utils.math import quat_wxyz_multiply_numpy, quat_wxyz_rotate_numpy
from ._arrays import _interpolate, _interpolate_rotation


class WalkingFeatures:
    """Ground-plane walking with optional offline reference-track filtering.

    - Per-clip Savitzky-Golay orders: position 4, unwrapped yaw 5.
    - Source motion stays unchanged; SciPy is needed only to build filtered features.
    """

    def __init__(self, *, position_filter_window=0.0, rotation_filter_window=0.0):
        """Configure filters used during database construction.

        Args:
            position_filter_window: Position window in seconds; zero disables it.
            rotation_filter_window: Unwrapped yaw window in seconds; zero disables it.
        """
        for name, value in (
            ("position_filter_window", position_filter_window),
            ("rotation_filter_window", rotation_filter_window),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative seconds")
            setattr(self, name, float(value))

    def describe(self, config) -> dict:
        """Describe feature blocks and semantics for DB construction and persistence.

        Args:
            config: Joint selection, trajectory horizons, axes, and block weights.

        Returns:
            JSON-compatible metadata identifying feature order and filter settings.
        """
        sizes = (
            3 * len(config.position_joints),
            3 * len(config.velocity_joints),
            2 * len(config.trajectory_times),
            2 * len(config.trajectory_times),
        )
        description = {
            "type": "joint_trajectory",
            "blocks": [
                {"name": name, "size": size, "weight": weight}
                for name, size, weight in zip(
                    (
                        "joint_positions",
                        "joint_velocities",
                        "trajectory_positions",
                        "trajectory_directions",
                    ),
                    sizes,
                    config.feature_weights,
                )
            ],
            "position_joints": list(config.position_joints),
            "velocity_joints": list(config.velocity_joints),
            "trajectory_times": list(config.trajectory_times),
            "frame": "root_heading",
            "planar_axes": [axis for axis in "XYZ" if axis != config.up_axis],
            "position_origin": "root_projected_to_zero_height_plane",
            "velocity": "world_linear_in_heading_frame",
            "pose_sampling": "linear_cached_features",
        }
        if self.position_filter_window or self.rotation_filter_window:
            description.update(
                frame="filtered_root_heading",
                position_origin="filtered_root_projected_to_zero_height_plane",
                reference_filter_type="savitzky_golay_v1",
                reference_filter={
                    "position_filter_window": self.position_filter_window,
                    "rotation_filter_window": self.rotation_filter_window,
                },
            )
        return description

    def build_clip(self, motion, *, config, candidate_count):
        """Build raw walking features from a motion validated by the database.

        Args:
            motion: Source clip in meters with unit WXYZ local rotations.
            config: Joint selection, trajectory horizons, and source axes.
            candidate_count: Number of leading frames with a full future horizon.

        Returns:
            Raw rows [N,D], full-clip pose cache [F,P], and reference-to-source WXYZ
            rotations [F,4]. N is candidate_count; F is the source frame count.
            Filtering adds reference positions [F,3] in source-space meters for
            placement; otherwise playback aligns the source root.
        """
        roots = np.asarray(motion.root_translations(), dtype=np.float64)
        rotations = np.array(motion.local_rotations_wxyz(), dtype=np.float64, copy=True)
        rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
        names = tuple(motion.node_names())
        parents = tuple(motion.parent_indices())
        tree = motion.skeleton_tree
        offsets = np.array(
            [
                tuple(getattr(tree.local_translation(j), axis) for axis in "xyz")
                for j in range(len(names))
            ],
            dtype=np.float64,
        )
        missing = set(config.position_joints + config.velocity_joints) - set(names)
        if missing:
            raise ValueError(f"missing feature joints: {', '.join(sorted(missing))}")
        frames, fps = len(roots), float(motion.fps())
        up = "XYZ".index(config.up_axis)
        forward = np.eye(3)["XYZ".index(config.forward_axis)]
        planar = [axis for axis in range(3) if axis != up]
        headings = self.reference_rotations(rotations[:, 0], config=config)
        reference_positions = None
        if self.position_filter_window or self.rotation_filter_window:
            reference_positions = self._filter(
                roots, fps, self.position_filter_window, order=4
            )
            reference_positions[:, up] = 0.0
            angles = np.unwrap(2 * np.arctan2(headings[:, up + 1], headings[:, 0]))
            angles = self._filter(angles, fps, self.rotation_filter_window, order=5)
            headings[:, 0] = np.cos(angles / 2)
            headings[:, up + 1] = np.sin(angles / 2)
        positions = np.empty((frames, len(names), 3), dtype=np.float64)
        global_rotations = np.empty_like(rotations)
        positions[:, 0] = roots
        global_rotations[:, 0] = rotations[:, 0]
        for joint in range(1, len(names)):
            parent = parents[joint]
            global_rotations[:, joint] = quat_wxyz_multiply_numpy(
                global_rotations[:, parent], rotations[:, joint]
            )
            positions[:, joint] = positions[:, parent] + quat_wxyz_rotate_numpy(
                global_rotations[:, parent], offsets[joint]
            )
        velocities = np.gradient(positions, 1.0 / fps, axis=0)
        trajectory_roots = roots if reference_positions is None else reference_positions
        ground_roots = trajectory_roots.copy()
        ground_roots[:, up] = 0.0
        position_ids = [names.index(name) for name in config.position_joints]
        velocity_ids = [names.index(name) for name in config.velocity_joints]
        local_positions = quat_wxyz_rotate_numpy(
            headings[:, None],
            positions[:, position_ids] - ground_roots[:, None],
            inverse=True,
        ).reshape(frames, -1)
        local_velocities = quat_wxyz_rotate_numpy(
            headings[:, None], velocities[:, velocity_ids], inverse=True
        ).reshape(frames, -1)
        pose = np.concatenate((local_positions, local_velocities), axis=-1)
        candidate_frames = np.arange(candidate_count)
        future_frames = (
            candidate_frames[:, None] + np.asarray(config.trajectory_times)[None] * fps
        )
        future_roots = _interpolate(trajectory_roots, future_frames)
        trajectory_positions = quat_wxyz_rotate_numpy(
            headings[:candidate_count, None],
            future_roots - trajectory_roots[:candidate_count, None],
            inverse=True,
        )[..., planar].reshape(candidate_count, -1)
        future_heading = _interpolate_rotation(headings, future_frames)
        future_directions = quat_wxyz_rotate_numpy(future_heading, forward)
        trajectory_directions = quat_wxyz_rotate_numpy(
            headings[:candidate_count, None], future_directions, inverse=True
        )[..., planar].reshape(candidate_count, -1)
        raw = np.concatenate(
            (pose[:candidate_count], trajectory_positions, trajectory_directions),
            axis=-1,
        )

        if reference_positions is not None:
            return raw, pose, headings, reference_positions
        return raw, pose, headings

    @staticmethod
    def _filter(values, fps, seconds, *, order):
        if seconds == 0 or len(values) <= order + 1:
            return values.copy()
        from scipy.signal import savgol_filter

        # Fit an odd window to the clip to avoid a half-frame shift.
        minimum = order + 1 if order % 2 == 0 else order + 2
        window = max(minimum, 2 * round(seconds * fps / 2) + 1)
        window = min(window, len(values) if len(values) % 2 else len(values) - 1)
        return savgol_filter(values, window, order, axis=0, mode="interp")

    @staticmethod
    def reference_rotations(rotations, *, config) -> np.ndarray:
        """Extract yaw about up_axis from projected source forward.

        Args:
            rotations: Unit WXYZ root rotations [F,4] in source coordinates.
            config: Source up and forward axes.

        Returns:
            Reference-to-source WXYZ heading rotations [F,4].

        Raises:
            ValueError: A vertical forward direction has no planar heading.
        """
        forward = np.eye(3)["XYZ".index(config.forward_axis)]
        up = "XYZ".index(config.up_axis)
        direction = quat_wxyz_rotate_numpy(rotations, forward).astype(np.float64)
        direction[:, up] = 0.0
        lengths = np.linalg.norm(direction, axis=-1)
        invalid = np.flatnonzero(lengths < 1e-6)
        if len(invalid):
            raise ValueError(
                f"frame {invalid[0]} has a vertical projected forward axis; choose a valid forward_axis or preprocess a heading track"
            )
        direction /= lengths[:, None]
        angles = np.unwrap(
            np.arctan2(np.cross(forward, direction)[:, up], direction @ forward)
        )
        headings = np.zeros((len(rotations), 4), dtype=np.float64)
        headings[:, 0] = np.cos(angles / 2.0)
        headings[:, up + 1] = np.sin(angles / 2.0)
        return headings

    def make_query(
        self,
        pose_features,
        *,
        frame,
        config,
        trajectory_positions,
        trajectory_directions,
        joint_positions=None,
        joint_velocities=None,
        skeleton_tree=None,
        reference_position=None,
        reference_rotation_wxyz=None,
    ):
        """Combine a live pose or an explicitly sampled clip with a trajectory.

        Args:
            pose_features: Full-clip pose cache [F,P] from build_clip().
            frame: Fractional source frame to sample.
            config: Trajectory horizons and source axes.
            trajectory_positions: Displacements [H,3] in reference-frame meters.
            trajectory_directions: Directions [H,3] in the same frame, one per
                horizon; planar projections must be nonzero and are normalized here.
            joint_positions: World joint positions [J,3] in meters. None uses the clip cache.
            joint_velocities: World joint velocities [J,3] in m/s, required with positions.
            skeleton_tree: Joint names/order of supplied positions and velocities.
            reference_position: World reference origin [3] in meters.
            reference_rotation_wxyz: Reference-to-world rotation [4].

        Returns:
            Raw features [D] in describe() block order, before DB normalization.
        """
        positions = np.asarray(trajectory_positions, dtype=np.float64)
        directions = np.asarray(trajectory_directions, dtype=np.float64)
        shape = (len(config.trajectory_times), 3)
        for name, values in (
            ("trajectory_positions", positions),
            ("trajectory_directions", directions),
        ):
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{name} must have shape {shape} with finite values")
        planar = [axis for axis in range(3) if axis != "XYZ".index(config.up_axis)]
        directions = directions[:, planar]
        lengths = np.linalg.norm(directions, axis=-1, keepdims=True)
        if np.any(lengths < 1e-6):
            raise ValueError(
                "trajectory_directions must have a nonzero planar direction"
            )
        return np.concatenate(
            (
                _interpolate(pose_features, frame)
                if joint_positions is None
                else self.encode_pose(
                    joint_positions,
                    joint_velocities,
                    skeleton_tree=skeleton_tree,
                    config=config,
                    reference_position=reference_position,
                    reference_rotation_wxyz=reference_rotation_wxyz,
                ),
                positions[:, planar].reshape(-1),
                (directions / lengths).reshape(-1),
            )
        )

    @staticmethod
    def encode_pose(
        joint_positions,
        joint_velocities,
        *,
        skeleton_tree,
        config,
        reference_position,
        reference_rotation_wxyz,
    ):
        """Encode current joint positions and velocities in the DB reference frame.

        Args:
            joint_positions: World positions [J,3] in meters, in skeleton order.
            joint_velocities: World linear velocities [J,3] in m/s, in skeleton order.
            skeleton_tree: Joint names/order of the supplied arrays.
            config: Feature joints and ground-plane axes used by the database.
            reference_position: World origin [3] in meters; height is projected to zero.
            reference_rotation_wxyz: Unit reference-to-world rotation [4].

        Returns:
            Raw position/velocity features [P], before normalization and weights.
        """
        positions = np.asarray(joint_positions)
        velocities = np.asarray(joint_velocities)
        shape = (skeleton_tree.num_joints(), 3)
        if positions.shape != shape or velocities.shape != shape:
            raise ValueError(f"joint positions and velocities must have shape {shape}")
        origin = np.array(reference_position, dtype=np.float64, copy=True)
        origin["XYZ".index(config.up_axis)] = 0.0
        position_ids = [skeleton_tree.index(name) for name in config.position_joints]
        velocity_ids = [skeleton_tree.index(name) for name in config.velocity_joints]
        return np.concatenate(
            (
                quat_wxyz_rotate_numpy(
                    reference_rotation_wxyz, positions[position_ids] - origin, inverse=True
                ).ravel(),
                quat_wxyz_rotate_numpy(
                    reference_rotation_wxyz, velocities[velocity_ids], inverse=True
                ).ravel(),
            )
        )
