"""Custom height feature and query using the LAFAN NPZ, without a viewer.

Run motion_matching_lafan.py first; the source NPZ is left unchanged.
"""

from pathlib import Path

import numpy as np

from kangengine import animation
from kangengine.utils.math import quat_wxyz_rotate_numpy


LAFAN_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "output/motion_matching_lafan/database.npz"
)


class HeightAwareFeatures(animation.WalkingFeatures):
    """Add root height to walking features."""

    def describe(self, config):
        description = super().describe(config)
        description["type"] = "example_height_aware_walking_v1"
        description["blocks"].append({"name": "root_height", "size": 1, "weight": 0.5})
        description["height_origin"] = "source_zero_height_plane"
        return description

    def build_clip(self, motion, *, config, candidate_count):
        rows, *cache = super().build_clip(
            motion, config=config, candidate_count=candidate_count
        )
        heights = np.asarray(motion.root_translations())[
            :, ["XYZ".index(config.up_axis)]
        ]
        return (
            np.concatenate((rows, heights[:candidate_count]), axis=1),
            *cache,
        )

    def make_query(self, pose_features, *, frame, config, desired_height, **trajectory):
        walking = super().make_query(
            pose_features, frame=frame, config=config, **trajectory
        )
        return np.concatenate((walking, [desired_height]))


def main():
    source: animation.MotionMatchingDatabase = (
        animation.MotionMatchingDatabase.from_npz(LAFAN_CACHE_PATH)
    )
    config = source.config
    database: animation.MotionMatchingDatabase = (
        animation.MotionMatchingDatabase.from_motions(
            source.motions,
            config=config,
            feature_builder=HeightAwareFeatures(
                **source.schema["features"].get("reference_filter", {})
            ),
        )
    )
    controller: animation.MotionMatchingController = animation.MotionMatchingController(
        database
    )
    forward = np.eye(3)["XYZ".index(config.forward_axis)]
    positions = np.asarray(config.trajectory_times)[:, None] * (1.2 * forward)
    directions = np.tile(forward, (len(positions), 1))

    def query(current: animation.MotionMatchingController):
        # Convert world displacements/directions to the query frame.
        heading = current.reference_rotation_wxyz
        return database.make_query(
            current.motion_index,
            current.time,
            desired_height=0.95,
            trajectory_positions=quat_wxyz_rotate_numpy(
                heading, positions, inverse=True
            ),
            trajectory_directions=quat_wxyz_rotate_numpy(
                heading, directions, inverse=True
            ),
        )

    for _ in range(180):
        state: animation.SkeletonState = controller.step_with_query(
            1 / 60, query_provider=query
        )
        if not np.isfinite(state.compute_global_matrices()).all():
            raise RuntimeError("custom playback produced a nonfinite pose")
    print(
        f"Custom walking: {database.features.shape[1]} features, root={controller.root_position.tolist()}"
    )


if __name__ == "__main__":
    main()
