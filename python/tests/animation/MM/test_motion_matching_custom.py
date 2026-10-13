"""Customization boundaries with walking defaults and a nonplanar test policy."""

import unittest

import numpy as np

from kangengine.animation import (
    MotionMatchingConfig,
    MotionMatchingController,
    MotionMatchingDatabase,
    SkeletonMotion,
    SkeletonMotionPlayer,
    WalkingFeatures,
)
from test_motion_matching import make_motion, make_config


class DisplacementFeatures:
    """Test-only policy: full 3D future displacement, no ground or heading."""

    def describe(self, config):
        return {
            "type": "test_displacement_3d",
            "frame": "source_world",
            "horizon": config.trajectory_times[-1],
            "blocks": [{"name": "displacement", "size": 3, "weight": 1}],
        }

    def build_clip(self, motion, *, config, candidate_count):
        roots = np.asarray(motion.root_translations())
        frames = np.arange(len(roots))
        future = np.arange(candidate_count) + config.trajectory_times[-1] * motion.fps()
        future_roots = np.stack(
            [np.interp(future, frames, roots[:, axis]) for axis in range(3)], axis=1
        )
        return (
            future_roots - roots[:candidate_count],
            np.zeros((len(roots), 0)),
            np.tile((1, 0, 0, 0), (len(roots), 1)),
        )

    def make_query(self, pose_features, *, frame, config, desired_displacement):
        return desired_displacement


class MotionMatchingCustomizationTests(unittest.TestCase):
    def test_invalid_rotations_are_rejected_before_feature_building(self):
        source = make_motion()
        for builder in (WalkingFeatures(), DisplacementFeatures()):
            for value in (0.0, 2.0, np.nan, np.inf):
                with self.subTest(builder=type(builder).__name__, value=value):
                    rotations = source.local_rotations_wxyz().copy()
                    rotations[0, 0] = (value, 0, 0, 0)
                    motion = SkeletonMotion.from_arrays(
                        source.skeleton_tree,
                        source.root_translations(),
                        rotations,
                        source.fps(),
                    )
                    with self.assertRaisesRegex(ValueError, "local rotations"):
                        MotionMatchingDatabase.from_motions(
                            [motion], config=make_config(), feature_builder=builder
                        )

    def test_explicit_walking_builder_preserves_defaults(self):
        motions = [make_motion(animated=True), make_motion(fps=60)]
        default = MotionMatchingDatabase.from_motions(motions, config=make_config())
        explicit = MotionMatchingDatabase.from_motions(
            motions, config=make_config(), feature_builder=WalkingFeatures()
        )
        np.testing.assert_array_equal(default.features, explicit.features)
        self.assertEqual(default.schema, explicit.schema)

    def make_custom_database(self):
        ordinary = make_motion(velocity=(0.7, 0, 0))
        upright_motion = make_motion(velocity=(0, 0, 1))
        rotations = np.asarray(upright_motion.local_rotations_wxyz()).copy()
        rotations[:, 0] = (np.sqrt(0.5), 0, -np.sqrt(0.5), 0)
        tilted = SkeletonMotion.from_arrays(
            upright_motion.skeleton_tree,
            upright_motion.root_translations(),
            rotations,
            upright_motion.fps(),
            "vertical_forward",
        )
        return MotionMatchingDatabase.from_motions(
            [ordinary, tilted],
            config=MotionMatchingConfig(
                position_joints=(), velocity_joints=(), trajectory_times=(0.2,)
            ),
            feature_builder=DisplacementFeatures(),
        )

    def test_custom_feature_query_placement_and_vertical_forward_work_together(self):
        database = self.make_custom_database()
        self.assertEqual(database.features.shape[1], 3)
        self.assertEqual(database.schema["feature_type"], "test_displacement_3d")
        self.assertEqual(database.schema["features"]["frame"], "source_world")
        placements = []

        def place(source, current):
            placements.append(current is None)
            return (1, 0, 0, 0), (3, 4, 5)

        controller = MotionMatchingController(
            database, root_placement=place, transition_half_life=0
        )
        calls = []

        def query(current):
            calls.append((current.motion_index, current.time, current.elapsed_time))
            return database.make_query(
                current.motion_index, current.time, desired_displacement=(0, 0, 0.2)
            )

        controller.step_with_query(0, query_provider=query)
        self.assertEqual(calls, [])
        output = controller.step_with_query(0.03, query_provider=query)
        self.assertEqual(controller.motion_index, 1)
        self.assertEqual(placements, [True, False])
        np.testing.assert_allclose(
            output.compute_global_matrices()[0, :3, 3], (3, 4, 6.03), atol=2e-6
        )
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            _ = controller.heading_rotation_wxyz
        controller.step_with_query(0.3, query_provider=query)
        self.assertGreaterEqual(len(calls), 3)
        np.testing.assert_allclose(
            [call[2] for call in calls], (0, 0.15, 0.3), atol=1e-12
        )
        self.assertAlmostEqual(controller.elapsed_time, 0.33)
        controller.reset(motion_index=1, time=0.1)
        self.assertTrue(placements[-1])
        self.assertEqual(controller.elapsed_time, 0)

    def test_external_trajectory_provider_matches_walking_step(self):
        database = MotionMatchingDatabase.from_motions(
            [make_motion(animated=True)], config=make_config()
        )
        walking, external = (
            MotionMatchingController(database),
            MotionMatchingController(database),
        )
        from kangengine.utils.math import quat_wxyz_rotate_numpy

        def query(controller):
            heading = controller.heading_rotation_wxyz
            positions = np.asarray(database.config.trajectory_times)[:, None] * (
                1,
                0,
                0,
            )
            directions = np.tile((1, 0, 0), (len(positions), 1))
            return controller.make_query(
                trajectory_positions=quat_wxyz_rotate_numpy(
                    heading, positions, inverse=True
                ),
                trajectory_directions=quat_wxyz_rotate_numpy(
                    heading, directions, inverse=True
                ),
            )

        for _ in range(20):
            first = walking.step(0.071, desired_velocity=(1, 0, 0))
            second = external.step_with_query(0.071, query_provider=query)
            np.testing.assert_array_equal(
                first.compute_global_matrices(), second.compute_global_matrices()
            )

    def test_player_accepts_nonplanar_clips_and_placement_per_operation(self):
        clip = self.make_custom_database().motions[1]
        player = SkeletonMotionPlayer([clip, make_motion()], transition_half_life=0)
        np.testing.assert_allclose(player.root_position, (0, 0, 1))
        player.reset_placed(0, 0, rotation_wxyz=(1, 0, 0, 0), translation=(3, 4, 5))
        np.testing.assert_allclose(player.root_position, (3, 4, 6))
        player.advance(0.1)
        player.transition_placed(
            0, 0.2, rotation_wxyz=(1, 0, 0, 0), translation=(7, 8, 9)
        )
        np.testing.assert_allclose(player.root_position, (7, 8, 10.2), atol=2e-6)
        before = player.state.compute_global_matrices()
        with self.assertRaises(ValueError):
            player.reset(0, 0.4)  # Walking heading is undefined for this pose.
        np.testing.assert_array_equal(player.state.compute_global_matrices(), before)
        with self.assertRaises(ValueError):
            player.transition_placed(
                0, 0.8, rotation_wxyz=(0, 0, 0, 0), translation=(0, 0, 0)
            )
        np.testing.assert_array_equal(player.state.compute_global_matrices(), before)
        self.assertAlmostEqual(player.time, 0.2)
        # An explicit operation does not change subsequent walking defaults.
        player.reset(1, 0)
        np.testing.assert_allclose(player.root_position, (0, 0, 1))
        player.transition_placed(
            1, 0, rotation_wxyz=(1, 0, 0, 0), translation=(3, 4, 0)
        )
        player.transition(1, 0.7)
        np.testing.assert_allclose(player.root_position, (3, 4, 1), atol=2e-6)

    def test_controller_can_change_placement_without_recreating_player(self):
        database = MotionMatchingDatabase.from_motions(
            [make_motion(), make_motion(velocity=(2, 0, 0))], config=make_config()
        )
        calls = []

        def place(source, current):
            calls.append((source, current))
            return (1, 0, 0, 0), (3, 4, 5)

        controller = MotionMatchingController(
            database, root_placement=place, transition_half_life=0
        )
        player = controller._player
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0][1])
        np.testing.assert_allclose(controller.root_position, (3, 4, 6))

        def place_again(source, current):
            calls.append((source, current))
            return (1, 0, 0, 0), (7, 8, 9)

        controller.root_placement = place_again
        np.testing.assert_allclose(controller.root_position, (3, 4, 6))
        query = database.features[database.motion_indices == 1][10].copy()
        controller.step_with_query(0.01, query_provider=lambda _: query)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(
            calls[1][1].compute_global_matrices()[0, :3, 3], (3, 4, 6)
        )
        np.testing.assert_allclose(controller.root_position[1:], (8, 10))
        controller.root_placement = None
        controller.reset()
        self.assertIs(controller._player, player)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(controller.root_position, (0, 0, 1))
        np.testing.assert_allclose(
            calls[0][0].compute_global_matrices()[0, :3, 3], (0, 0, 1)
        )
        with self.assertRaises(TypeError):
            controller.root_placement = False
        self.assertIsNone(controller.root_placement)

    def test_schema_drift_bad_builder_shapes_and_queries_are_rejected(self):
        config = make_config()
        motion = make_motion()

        class BadRows(DisplacementFeatures):
            def build_clip(self, motion, *, config, candidate_count):
                rows, pose, frames = super().build_clip(
                    motion, config=config, candidate_count=candidate_count
                )
                return rows[:, :2], pose, frames

        with self.assertRaises(ValueError):
            MotionMatchingDatabase.from_motions(
                [motion], config=config, feature_builder=BadRows()
            )
        with self.assertRaises(AttributeError):
            MotionMatchingDatabase.from_motions(
                [motion], config=config, feature_builder=object()
            )
        database = self.make_custom_database()
        for bad in ((0, 0), (0, 0, np.nan)):
            with self.assertRaises(ValueError):
                database.make_query(0, 0, desired_displacement=bad)

    def test_rigid_placement_transforms_pose_and_velocity_before_smoothing(self):
        player = SkeletonMotionPlayer(
            [make_motion(velocity=(0.4, 0, 0))],
            transition_half_life=0,
        )
        rotation = (np.sqrt(0.5), 0, -np.sqrt(0.5), 0)
        player.reset_placed(0, 0, rotation_wxyz=rotation, translation=(3, 4, 5))
        player.advance(0.1)
        np.testing.assert_allclose(player.root_position, (2, 4, 5.04), atol=2e-6)
        np.testing.assert_allclose(player.root_linear_velocity, (0, 0, 0.4), atol=2e-6)
        player.transition_half_life = 0.1
        before = player.state.compute_global_matrices()
        velocity = player.root_linear_velocity.copy()
        player.transition_placed(
            0, 0.2, rotation_wxyz=(1, 0, 0, 0), translation=(8, 0, 0)
        )
        np.testing.assert_array_equal(player.state.compute_global_matrices(), before)

        animated = make_motion(animated=True)
        source = SkeletonMotionPlayer(
            [animated],
            transition_half_life=0,
        )
        source.reset_placed(0, 0, rotation_wxyz=(1, 0, 0, 0), translation=(0, 0, 0))
        placed = SkeletonMotionPlayer([animated], transition_half_life=0)
        placed.reset_placed(0, 0, rotation_wxyz=rotation, translation=(3, 4, 5))
        source.advance(0.13)
        placed.advance(0.13)
        basis = np.array(((0, 0, -1), (0, 1, 0), (1, 0, 0)))
        np.testing.assert_allclose(
            placed.root_linear_velocity, basis @ source.root_linear_velocity, atol=2e-6
        )
        np.testing.assert_allclose(
            placed.local_angular_velocities[0],
            basis @ source.local_angular_velocities[0],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            placed.local_angular_velocities[1:],
            source.local_angular_velocities[1:],
            atol=2e-6,
        )
        np.testing.assert_array_equal(player.root_linear_velocity, velocity)
        with self.assertRaises(ValueError):
            player.reset_placed(0, 0, rotation_wxyz=(0, 0, 0, 0), translation=(0, 0, 0))
        np.testing.assert_array_equal(player.state.compute_global_matrices(), before)


if __name__ == "__main__":
    unittest.main()
