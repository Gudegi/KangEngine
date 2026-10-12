"""Motion matching contracts, using native synthetic clips without asset files."""

import json
import math
import unittest

import numpy as np

from kangengine.animation import (
    MotionMatchingConfig,
    MotionMatchingController,
    MotionMatchingDatabase,
    SkeletonMotion,
    SkeletonState,
    SkeletonTree,
    transform_motion,
)


def make_motion(
    *,
    fps: float = 30.0,
    duration: float = 2.0,
    velocity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 1.0),
    heading: float = 0.0,
    animated: bool = False,
    tree: SkeletonTree | None = None,
) -> SkeletonMotion:
    if tree is None:
        tree = SkeletonTree(
            ["root", "left_foot", "right_foot"],
            [-1, 0, 0],
            ((0.0, 0.0, 0.0), (0.0, 0.2, -1.0), (0.0, -0.2, -1.0)),
            np.tile((1.0, 0.0, 0.0, 0.0), (3, 1)),
        )
    times = np.arange(round(duration * fps) + 1, dtype=np.float64) / fps
    roots = np.asarray(origin) + times[:, None] * np.asarray(velocity)
    headings = np.full(times.shape, heading)
    if animated:
        roots[:, 0] += 0.1 * np.sin(times * 3.0)
        roots[:, 1] += 0.1 * np.cos(times * 2.0)
        roots[:, 2] += 0.03 * np.sin(times * 5.0)
        headings += 0.2 * np.sin(times * 2.0)
    rotations = np.tile((1.0, 0.0, 0.0, 0.0), (len(times), 3, 1))
    rotations[:, 0, 0] = np.cos(headings / 2.0)
    rotations[:, 0, 3] = np.sin(headings / 2.0)
    if animated:
        swing = 0.3 * np.sin(times * 6.0)
        rotations[:, 1:, 0] = np.cos(swing[:, None] / 2.0)
        rotations[:, 1, 2] = np.sin(swing / 2.0)
        rotations[:, 2, 2] = -np.sin(swing / 2.0)
    return SkeletonMotion.from_arrays(tree, roots, rotations, fps, "synthetic")


def make_config(**overrides: object) -> MotionMatchingConfig:
    values = {
        "position_joints": ("left_foot", "right_foot"),
        "velocity_joints": ("left_foot", "right_foot", "root"),
    }
    values.update(overrides)
    return MotionMatchingConfig(**values)


def root_position(state: SkeletonState) -> np.ndarray:
    return np.asarray(state.compute_global_matrices())[0, :3, 3].copy()


class MotionMatchingDatabaseTests(unittest.TestCase):
    def test_features_ignore_world_yaw_and_planar_translation(self) -> None:
        motion = make_motion(animated=True)
        yaw = 0.7
        transformed = transform_motion(
            motion,
            rotation_wxyz=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
            translation=(3.0, -4.0, 0.0),
        )
        database = MotionMatchingDatabase.from_motions(
            [motion, transformed], config=make_config()
        )
        first = database.features[database.motion_indices == 0]
        second = database.features[database.motion_indices == 1]
        self.assertEqual(first.shape[1], 27)
        np.testing.assert_allclose(first, second, atol=2e-3, rtol=2e-3)

    def test_future_horizon_uses_seconds_and_stays_in_each_clip(self) -> None:
        motions = [
            make_motion(fps=30.0, duration=2.0),
            make_motion(fps=60.0, duration=1.5, origin=(100.0, 0.0, 1.0)),
        ]
        database = MotionMatchingDatabase.from_motions(motions, config=make_config())
        # A one-second future horizon leaves 31 and 31 usable source frames.
        self.assertEqual(database.num_frames, 62)
        for index, last_frame in enumerate((30, 30)):
            frames = database.frame_indices[database.motion_indices == index]
            np.testing.assert_array_equal(frames, np.arange(last_frame + 1))
            times = frames / motions[index].fps()
            self.assertTrue(np.all(times + 1.0 <= motions[index].duration() + 1e-6))

    def test_tiny_positive_horizon_excludes_final_frame_and_allows_restart(
        self,
    ) -> None:
        motion = make_motion(duration=0.1)
        database = MotionMatchingDatabase.from_motions(
            [motion], config=make_config(trajectory_times=(1e-10,))
        )
        self.assertEqual(database.num_frames, motion.num_frames() - 1)
        self.assertTrue(np.all(database.frame_indices < motion.num_frames() - 1))
        controller = MotionMatchingController(
            database, search_interval=10.0, transition_half_life=0.0
        )
        exact_end = (motion.num_frames() - 1) / motion.fps()
        controller.reset(time=exact_end)
        state = controller.step(0.025, desired_velocity=(1.0, 0.0, 0.0))
        self.assertTrue(np.isfinite(state.compute_global_matrices()).all())
        np.testing.assert_allclose(root_position(state), (0.025, 0.0, 1.0), atol=1e-6)
        self.assertGreaterEqual(controller.transition_count, 1)
        self.assertGreaterEqual(controller.time, 0.0)
        self.assertLess(controller.time, exact_end)

    def test_static_clip_has_finite_features_and_zero_cost_self_match(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion(velocity=(0.0, 0.0, 0.0))], config=make_config()
        )
        for array in (
            database.features,
            database.mean,
            database.scale,
            database.weights,
        ):
            self.assertTrue(np.isfinite(array).all())
        self.assertTrue(np.all(database.scale > 0.0))
        query = database.make_query(
            motion_index=0,
            time=0.0,
            trajectory_positions=np.zeros((3, 3)),
            trajectory_directions=np.tile((1.0, 0.0, 0.0), (3, 1)),
        )
        match = database.search(query)
        self.assertEqual(match.motion_index, 0)
        self.assertAlmostEqual(match.cost, 0.0, places=6)

    def test_search_agrees_with_exhaustive_distance_oracle(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion(animated=True), make_motion(velocity=(0.0, 1.0, 0.0))],
            config=make_config(feature_weights=(0.7, 1.2, 2.5, 0.4)),
        )
        horizons = np.asarray(database.config.trajectory_times)
        query = database.make_query(
            motion_index=0,
            time=0.4,
            trajectory_positions=horizons[:, None] * np.array((0.2, 0.8, 0.0)),
            trajectory_directions=np.tile((1.0, 0.0, 0.0), (len(horizons), 1)),
        )
        costs = np.sum(np.square(database.features - query), axis=1)
        best = int(np.argmin(costs))
        match = database.search(query)
        self.assertEqual(match.motion_index, int(database.motion_indices[best]))
        self.assertEqual(match.frame_index, int(database.frame_indices[best]))
        self.assertAlmostEqual(match.cost, float(costs[best]), places=5)
        self.assertAlmostEqual(
            match.time, match.frame_index / database.motions[match.motion_index].fps()
        )

    def test_target_trajectory_selects_sideways_motion(self) -> None:
        config = make_config(feature_weights=(0.0, 0.0, 1.0, 0.0))
        database = MotionMatchingDatabase.from_motions(
            [make_motion(), make_motion(velocity=(0.0, 1.0, 0.0))], config=config
        )
        horizons = np.asarray(config.trajectory_times)
        query = database.make_query(
            motion_index=0,
            time=0.0,
            trajectory_positions=horizons[:, None] * np.array((0.0, 1.0, 0.0)),
            trajectory_directions=np.tile((1.0, 0.0, 0.0), (len(horizons), 1)),
        )
        self.assertEqual(database.search(query).motion_index, 1)

    def test_continuation_bias_only_favors_current_searchable_frame(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion(velocity=(0.0, 0.0, 0.0))], config=make_config()
        )
        query = database.features[0]
        match = database.search(
            query,
            current_motion_index=0,
            current_time=0.5,
            continuation_bias=0.25,
        )
        self.assertEqual(match.frame_index, 15)
        self.assertAlmostEqual(match.cost, -0.25)
        # An incomplete future horizon cannot retain a tail frame indefinitely.
        tail = database.search(
            query,
            current_motion_index=0,
            current_time=1.5,
            continuation_bias=0.25,
        )
        self.assertEqual(tail.frame_index, 0)
        self.assertAlmostEqual(tail.cost, 0.0)

    def test_public_arrays_are_read_only(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion()], config=make_config()
        )
        for name in (
            "features",
            "mean",
            "scale",
            "weights",
            "motion_indices",
            "frame_indices",
        ):
            with self.subTest(name=name):
                array = getattr(database, name)
                self.assertFalse(array.flags.writeable)
                with self.assertRaises(ValueError):
                    array.flat[0] = 123.0

    def test_rejects_skeletons_with_different_bind_offsets(self) -> None:
        other_tree = SkeletonTree(
            ["root", "left_foot", "right_foot"],
            [-1, 0, 0],
            ((0.0, 0.0, 0.0), (0.0, 0.2, -2.0), (0.0, -0.2, -1.0)),
            np.tile((1.0, 0.0, 0.0, 0.0), (3, 1)),
        )
        with self.assertRaises(ValueError):
            MotionMatchingDatabase.from_motions(
                [make_motion(), make_motion(tree=other_tree)], config=make_config()
            )

    def test_rejects_empty_and_too_short_databases(self) -> None:
        for motions in ([], [make_motion(duration=0.5)]):
            with self.subTest(motions=len(motions)), self.assertRaises(ValueError):
                MotionMatchingDatabase.from_motions(motions, config=make_config())

    def test_rejects_nonfinite_fps_and_pose_data(self) -> None:
        valid = make_motion()
        for field, value in (("fps", np.nan), ("fps", np.inf), ("roots", np.nan)):
            with self.subTest(field=field, value=value):
                roots = valid.root_translations().copy()
                fps = valid.fps()
                if field == "fps":
                    fps = value
                else:
                    roots[3, 0] = value
                invalid = SkeletonMotion.from_arrays(
                    valid.skeleton_tree, roots, valid.local_rotations_wxyz(), fps
                )
                with self.assertRaises(ValueError):
                    MotionMatchingDatabase.from_motions([invalid], config=make_config())

    def test_rejects_missing_feature_joint(self) -> None:
        with self.assertRaises(ValueError):
            MotionMatchingDatabase.from_motions(
                [make_motion()], config=make_config(position_joints=("missing",))
            )

    def test_rejects_invalid_config(self) -> None:
        cases = (
            {"up_axis": "X", "forward_axis": "X"},
            {"trajectory_times": (0.0,)},
            {"trajectory_times": (0.5, 0.25)},
            {"trajectory_times": (np.inf,)},
            {"feature_weights": (1.0, -1.0, 1.0, 1.0)},
            {"feature_weights": (1.0, 1.0, np.nan, 1.0)},
            {"feature_weights": (0.0, 0.0, 0.0, 0.0)},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                MotionMatchingDatabase.from_motions(
                    [make_motion()], config=make_config(**overrides)
                )

    def test_rejects_malformed_and_nonfinite_search_queries(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion()], config=make_config()
        )
        for query in (np.zeros(26), np.zeros((1, 27)), np.full(27, np.nan)):
            with self.subTest(shape=query.shape), self.assertRaises(ValueError):
                database.search(query)


class MotionMatchingControllerTests(unittest.TestCase):
    def test_y_up_playback_preserves_height(self) -> None:
        half = math.sqrt(0.5)
        motion = transform_motion(make_motion(), rotation_wxyz=(half, -half, 0.0, 0.0))
        database = MotionMatchingDatabase.from_motions(
            [motion], config=make_config(up_axis="Y", forward_axis="X")
        )
        controller = MotionMatchingController(database)
        state = controller.step(0.3, desired_velocity=(1.0, 0.0, 0.0))
        np.testing.assert_allclose(root_position(state), (0.3, 1.0, 0.0), atol=1e-5)

    def test_zero_step_does_not_search_or_change_pose(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion(), make_motion(velocity=(0.0, 1.0, 0.0))],
            config=make_config(),
        )
        controller = MotionMatchingController(database)
        before = np.asarray(controller.state.compute_global_matrices()).copy()
        state = controller.step(0.0, desired_velocity=(0.0, 1.0, 0.0))
        np.testing.assert_array_equal(state.compute_global_matrices(), before)
        self.assertEqual(controller.time, 0.0)
        self.assertEqual(controller.transition_count, 0)
        self.assertIsNone(controller.last_match)

    def test_transition_aligns_planar_root_and_heading(self) -> None:
        # Clip 2 is authored far away and faces +Y; locally it moves sideways +Y.
        database = MotionMatchingDatabase.from_motions(
            [
                make_motion(origin=(5.0, -3.0, 1.0)),
                make_motion(
                    origin=(100.0, 40.0, 1.0),
                    heading=math.pi / 2.0,
                    velocity=(-1.0, 0.0, 0.0),
                ),
            ],
            config=make_config(feature_weights=(0.0, 0.0, 1.0, 0.0)),
        )
        # Raw-mode parity isolates root placement from velocity smoothing.
        controller = MotionMatchingController(
            database, continuation_bias=0.0, transition_half_life=0.0
        )
        before = np.asarray(controller.state.compute_global_matrices())[0].copy()
        state = controller.step(
            0.05, desired_velocity=(0.0, 1.0, 0.0), desired_facing=(1.0, 0.0, 0.0)
        )
        self.assertIsInstance(state, SkeletonState)
        self.assertEqual(controller.motion_index, 1)
        self.assertGreaterEqual(controller.transition_count, 1)
        after = np.asarray(state.compute_global_matrices())[0]
        np.testing.assert_allclose(after[:3, :3], before[:3, :3], atol=1e-5)
        np.testing.assert_allclose(
            after[:3, 3] - before[:3, 3], (0.0, 0.05, 0.0), atol=2e-5
        )
        self.assertIsNotNone(controller.last_match)

    def test_constant_request_is_independent_of_step_partition(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion()], config=make_config()
        )
        whole = MotionMatchingController(database, search_interval=0.15)
        split = MotionMatchingController(database, search_interval=0.15)
        whole.step(0.7, desired_velocity=(1.0, 0.0, 0.0))
        for _ in range(7):
            split.step(0.1, desired_velocity=(1.0, 0.0, 0.0))
        np.testing.assert_allclose(
            whole.state.compute_global_matrices(),
            split.state.compute_global_matrices(),
            atol=2e-5,
        )
        self.assertEqual(whole.motion_index, split.motion_index)
        self.assertAlmostEqual(whole.time, split.time, places=6)

    def test_clip_end_reselects_without_looping_root_backwards(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion(duration=1.2)], config=make_config()
        )
        controller = MotionMatchingController(
            database, search_interval=10.0, transition_half_life=0.0
        )
        before = root_position(controller.state)
        controller.step(3.0, desired_velocity=(1.0, 0.0, 0.0))
        np.testing.assert_allclose(
            root_position(controller.state) - before, (3.0, 0.0, 0.0), atol=2e-5
        )
        self.assertGreaterEqual(controller.transition_count, 1)
        self.assertGreaterEqual(controller.time, 0.0)
        self.assertLessEqual(controller.time, database.motions[0].duration())

    def test_nonintegral_fps_endpoint_keeps_advancing(self) -> None:
        motion = make_motion(fps=29.97, duration=1.2)
        database = MotionMatchingDatabase.from_motions([motion], config=make_config())
        controller = MotionMatchingController(
            database, search_interval=10.0, transition_half_life=0.0
        )
        exact_end = (motion.num_frames() - 1) / motion.fps()
        controller.reset(time=exact_end)
        dt = 3.0 * exact_end + 0.01
        state = controller.step(dt, desired_velocity=(1.0, 0.0, 0.0))
        self.assertTrue(np.isfinite(state.compute_global_matrices()).all())
        np.testing.assert_allclose(
            root_position(state), (dt, 0.0, 1.0), atol=2e-5
        )
        self.assertGreaterEqual(controller.transition_count, 3)
        self.assertGreaterEqual(controller.time, 0.0)
        self.assertLessEqual(controller.time, exact_end)

    def test_reset_samples_requested_time_at_canonical_planar_origin(self) -> None:
        motions = [make_motion(), make_motion(origin=(10.0, 2.0, 1.0))]
        database = MotionMatchingDatabase.from_motions(motions, config=make_config())
        controller = MotionMatchingController(database)
        controller.step(0.5, desired_velocity=(1.0, 0.0, 0.0))
        controller.reset(motion_index=1, time=0.25)
        self.assertEqual(controller.motion_index, 1)
        self.assertAlmostEqual(controller.time, 0.25)
        expected = np.asarray(
            motions[1].sample(time=0.25, loop=False).compute_global_matrices()
        ).copy()
        expected[:, :2, 3] -= expected[0, :2, 3].copy()
        np.testing.assert_allclose(
            controller.state.compute_global_matrices(), expected, atol=1e-6
        )
        self.assertEqual(controller.transition_count, 0)
        self.assertIsNone(controller.last_match)

    def test_reset_rejects_time_outside_source_clip(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion()], config=make_config()
        )
        controller = MotionMatchingController(database)
        for time in (-0.1, 2.1, np.nan, np.inf):
            with self.subTest(time=time), self.assertRaises(ValueError):
                controller.reset(time=time)

    def test_invalid_step_does_not_change_current_pose(self) -> None:
        database = MotionMatchingDatabase.from_motions(
            [make_motion()], config=make_config()
        )
        controller = MotionMatchingController(database)
        before = np.asarray(controller.state.compute_global_matrices()).copy()
        for dt in (-0.1, np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                controller.step(dt, desired_velocity=(1.0, 0.0, 0.0))
            np.testing.assert_array_equal(
                controller.state.compute_global_matrices(), before
            )
        for velocity in ((1.0, 0.0), (np.nan, 0.0, 0.0)):
            with self.subTest(velocity=velocity), self.assertRaises(ValueError):
                controller.step(0.1, desired_velocity=velocity)
            np.testing.assert_array_equal(
                controller.state.compute_global_matrices(), before
            )


class MotionMatchingSchemaTests(unittest.TestCase):
    def test_schema_records_feature_layout_rig_clocks_and_normalization(self):
        database = MotionMatchingDatabase.from_motions(
            [make_motion(animated=True), make_motion(fps=60)],
            config=make_config(),
        )
        schema = json.loads(json.dumps(database.schema, allow_nan=False))
        self.assertEqual(schema["schema_version"], 1)
        self.assertEqual(schema["length_unit"], "m")
        self.assertEqual(schema["quaternion_order"], "wxyz")
        self.assertEqual(schema["features"]["planar_axes"], ["X", "Y"])
        self.assertEqual(
            schema["skeleton"]["joint_names"], list(database.motions[0].node_names())
        )
        self.assertEqual([clip["fps"] for clip in schema["clips"]], [30, 60])
        self.assertEqual(
            sum(clip["candidate_count"] for clip in schema["clips"]),
            database.num_frames,
        )
        offset = 0
        for block in schema["features"]["layout"]:
            self.assertEqual(block["offset"], offset)
            offset += block["size"]
        self.assertEqual(offset, database.features.shape[1])
        self.assertEqual(schema["feature_dimension"], offset)
        for name, array in (
            ("mean", database.mean),
            ("scale", database.scale),
            ("multipliers", database.weights),
        ):
            np.testing.assert_array_equal(schema["normalization"][name], array)
        schema["normalization"]["mean"][0] = 12345
        schema["skeleton"]["joint_names"][0] = "changed"
        self.assertNotEqual(database.schema["normalization"]["mean"][0], 12345)
        self.assertEqual(database.schema["skeleton"]["joint_names"][0], "root")

    def test_meters_are_fixed_and_empty_blocks_are_explicit(self):
        motion = make_motion(animated=True)
        database = MotionMatchingDatabase.from_motions(
            [motion], config=make_config(position_joints=())
        )
        self.assertEqual(database.schema["length_unit"], "m")
        self.assertEqual(database.schema["features"]["layout"][0]["size"], 0)
        with self.assertRaises(TypeError):
            make_config(length_unit="m")


if __name__ == "__main__":
    unittest.main()
