"""Native pose playback and transition contracts without renderer or assets."""

import math
import gc
import unittest

import numpy as np

from kangengine.animation import (
    MotionMatchingConfig,
    MotionMatchingController,
    MotionMatchingDatabase,
    SkeletonMotion,
    SkeletonMotionPlayer,
    SkeletonTree,
)


def make_tree(*, up_axis: int = 2, arm_length: float = 0.3) -> SkeletonTree:
    offsets = np.array(
        (
            (0, 0, 0),
            (0, 0, 0.4),
            (0, 0, 0.3),
            (0, arm_length, 0.1),
            (0, arm_length, 0),
            (0, -0.2, -0.8),
        ),
        dtype=np.float32,
    )
    if up_axis == 1:
        offsets = offsets[:, (0, 2, 1)]
    return SkeletonTree(
        ["root", "spine", "head", "shoulder", "hand", "foot"],
        [-1, 0, 1, 1, 3, 0],
        offsets,
        np.tile((1.0, 0.0, 0.0, 0.0), (6, 1)),
    )


def make_clip(
    *,
    tree: SkeletonTree | None = None,
    fps: float = 60.0,
    height: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float] = (0.4, 0.1, 0.0),
    angles: tuple[float, ...] = (0.0, 0.1, -0.1, 0.2, 0.4, -0.1),
    angular_speeds: tuple[float, ...] = (0.2, 0.1, -0.2, 0.3, -0.4, 0.1),
    rotation_axis: int = 1,
    up_axis: int = 2,
    alternate_quaternion_signs: bool = False,
) -> SkeletonMotion:
    tree = make_tree(up_axis=up_axis) if tree is None else tree
    times = np.arange(181, dtype=np.float64) / fps
    roots = np.asarray(origin) + times[:, None] * np.asarray(velocity)
    roots[:, up_axis] += height
    joint_angles = np.asarray(angles)[None, :] + times[:, None] * np.asarray(
        angular_speeds
    )
    rotations = np.zeros((len(times), 6, 4), dtype=np.float64)
    rotations[:, :, 0] = np.cos(joint_angles / 2)
    rotations[:, 1:, rotation_axis + 1] = np.sin(joint_angles[:, 1:] / 2)
    rotations[:, 0, up_axis + 1] = np.sin(joint_angles[:, 0] / 2)
    if alternate_quaternion_signs:
        rotations[1::2] *= -1
    return SkeletonMotion.from_arrays(tree, roots, rotations, fps, "player_test")


def rotations_wxyz(state) -> np.ndarray:
    return np.array(
        [
            (q.w, q.x, q.y, q.z)
            for q in (state.rotation(index=i) for i in range(state.num_joints()))
        ],
        dtype=np.float64,
    )


def angular_difference(before: np.ndarray, after: np.ndarray, dt: float) -> np.ndarray:
    """Measure spatial local-parent angular velocity from two quaternion poses."""
    bw, bv = before[:, :1], -before[:, 1:]
    aw, av = after[:, :1], after[:, 1:]
    scalar = aw * bw - np.sum(av * bv, axis=1, keepdims=True)
    vector = aw * bv + bw * av + np.cross(av, bv)
    vector = np.where(scalar < 0, -vector, vector)
    length = np.linalg.norm(vector, axis=1, keepdims=True)
    angle = 2 * np.arctan2(length, np.abs(scalar))
    return (
        vector
        * np.divide(angle, length, out=np.full_like(length, 2), where=length > 1e-12)
        / dt
    )


class SkeletonMotionPlayerTests(unittest.TestCase):
    def test_shared_clips_keep_player_states_independent_and_survive_source_deletion(
        self,
    ):
        clips = [make_clip(), make_clip(height=1.6)]
        source_roots = clips[0].root_translations()
        source_rotations = clips[0].local_rotations_wxyz()
        first, second, expected = (SkeletonMotionPlayer(clips) for _ in range(3))
        # Array access still returns snapshots, even though native copies share
        # immutable storage. Mutating exported arrays cannot affect playback.
        source_roots[:] = 99
        source_rotations[:] = 0
        initial = second.state.compute_global_matrices().copy()
        first.advance(0.23)
        first.transition(1, 0.4)
        first.advance(0.07)
        self.assertEqual(second.time, 0)
        np.testing.assert_array_equal(second.state.compute_global_matrices(), initial)

        del clips
        gc.collect()
        second.advance(0.37)
        expected.advance(0.37)
        np.testing.assert_array_equal(
            second.state.compute_global_matrices(),
            expected.state.compute_global_matrices(),
        )
        second_pose = second.state.compute_global_matrices().copy()
        second_velocity = second.root_linear_velocity.copy()
        first.reset_placed(0, 0.1, rotation_wxyz=(1, 0, 0, 0), translation=(10, 20, 0))
        first.transition_half_life = 0
        first.advance(0.3)
        np.testing.assert_array_equal(
            second.state.compute_global_matrices(), second_pose
        )
        np.testing.assert_array_equal(second.root_linear_velocity, second_velocity)
        self.assertAlmostEqual(second.transition_half_life, 0.1)

    def make_pair(self, **kwargs) -> list[SkeletonMotion]:
        return [
            make_clip(**kwargs),
            make_clip(
                height=1.7,
                origin=(7.0, -3.0, 0.0),
                velocity=(-0.2, 0.5, 0.15),
                angles=(1.2, -0.5, 0.7, -0.6, -0.9, 0.3),
                angular_speeds=(-0.1, -0.3, 0.5, -0.2, 0.2, -0.3),
                rotation_axis=0,
                **kwargs,
            ),
        ]

    def assert_pose_equal(self, first, second, *, atol=2e-5) -> None:
        np.testing.assert_allclose(
            first.compute_global_matrices(),
            second.compute_global_matrices(),
            atol=atol,
            rtol=0,
        )

    def test_transition_preserves_every_joint_root_height_and_velocity(self) -> None:
        player = SkeletonMotionPlayer(self.make_pair(), transition_half_life=0.1)
        player.advance(dt=0.31)
        before = player.state
        linear = np.array(player.root_linear_velocity)
        angular = np.array(player.local_angular_velocities)
        player.transition(motion_index=1, time=0.43)
        self.assertEqual(player.motion_index, 1)
        self.assertAlmostEqual(player.time, 0.43)
        self.assert_pose_equal(before, player.state)
        self.assertIsNone(player.advance(dt=0))
        self.assert_pose_equal(before, player.state)
        np.testing.assert_allclose(player.root_linear_velocity, linear, atol=2e-5)
        np.testing.assert_allclose(player.local_angular_velocities, angular, atol=2e-5)

    def test_reported_velocities_match_finite_differences_across_transition(
        self,
    ) -> None:
        player = SkeletonMotionPlayer(self.make_pair(), transition_half_life=0.15)
        player.advance(dt=0.35)
        player.transition(motion_index=1, time=0.47)
        for elapsed in (0.0, 0.08, 0.17):
            with self.subTest(elapsed=elapsed):
                player.advance(dt=elapsed)
                root = np.array(player.root_position)
                rotations = rotations_wxyz(player.state)
                linear = np.array(player.root_linear_velocity)
                angular = np.array(player.local_angular_velocities)
                dt = 1e-4
                player.advance(dt=dt)
                after = player.state
                measured_linear = (np.asarray(player.root_position) - root) / dt
                measured_angular = angular_difference(
                    rotations, rotations_wxyz(after), dt
                )
                np.testing.assert_allclose(
                    measured_linear, linear, atol=0.012, rtol=0.01
                )
                np.testing.assert_allclose(
                    measured_angular, angular, atol=0.02, rtol=0.01
                )

    def test_interrupted_transition_preserves_current_blended_pose_and_velocity(
        self,
    ) -> None:
        player = SkeletonMotionPlayer(self.make_pair(), transition_half_life=0.15)
        player.advance(dt=0.3)
        for index, time in ((1, 0.4), (0, 0.7), (1, 0.8), (0, 0.6)):
            before = player.state
            linear = np.array(player.root_linear_velocity)
            angular = np.array(player.local_angular_velocities)
            player.transition(motion_index=index, time=time)
            self.assert_pose_equal(before, player.state)
            np.testing.assert_allclose(player.root_linear_velocity, linear, atol=3e-5)
            np.testing.assert_allclose(
                player.local_angular_velocities, angular, atol=3e-5
            )
            player.advance(dt=0.037)

    def test_split_advance_matches_single_advance(self) -> None:
        motions = self.make_pair()
        whole = SkeletonMotionPlayer(motions, transition_half_life=0.12)
        split = SkeletonMotionPlayer(motions, transition_half_life=0.12)
        for player in (whole, split):
            player.advance(dt=0.21)
            player.transition(motion_index=1, time=0.32)
        whole.advance(dt=0.24)
        for _ in range(24):
            split.advance(dt=0.01)
        self.assert_pose_equal(whole.state, split.state, atol=4e-5)
        np.testing.assert_allclose(
            whole.root_linear_velocity, split.root_linear_velocity, atol=5e-5
        )
        np.testing.assert_allclose(
            whole.local_angular_velocities, split.local_angular_velocities, atol=5e-5
        )

    def test_static_offsets_converge_without_changing_source_data(self) -> None:
        first = make_clip(velocity=(0, 0, 0), angles=(0,) * 6, angular_speeds=(0,) * 6)
        second = make_clip(
            height=1.8,
            velocity=(0, 0, 0),
            angles=(0, 0.8, -0.5, 1, -0.6, 0.2),
            angular_speeds=(0,) * 6,
        )
        roots_before = second.root_translations().copy()
        rotations_before = second.local_rotations_wxyz().copy()
        player = SkeletonMotionPlayer([first, second], transition_half_life=0.08)
        player.transition(motion_index=1, time=0)
        heights = [float(player.root_position[2])]
        for _ in range(10):
            player.advance(dt=0.1)
            heights.append(float(player.root_position[2]))
        self.assertTrue(np.all(np.diff(heights) > -1e-6))
        self.assertLessEqual(max(heights), 1.80001)
        self.assert_pose_equal(
            player.state, second.sample(time=1, loop=False), atol=5e-4
        )
        np.testing.assert_array_equal(second.root_translations(), roots_before)
        np.testing.assert_array_equal(second.local_rotations_wxyz(), rotations_before)

    def test_static_height_correction_has_analytic_decay_even_after_clip_end(
        self,
    ) -> None:
        clips = [
            make_clip(
                height=height,
                velocity=(0, 0, 0),
                angles=(0,) * 6,
                angular_speeds=(0,) * 6,
            )
            for height in (1, 2)
        ]
        half_life = 0.2
        player = SkeletonMotionPlayer(clips, transition_half_life=half_life)
        player.transition(motion_index=1, time=player.duration)
        elapsed = 0.0
        decay_rate = 2 * math.log(2) / half_life
        for dt in (0.025, 0.07, 0.2, 0.8):
            elapsed += dt
            player.advance(dt=dt)
            expected_height = 2 - (1 + decay_rate * elapsed) * math.exp(
                -decay_rate * elapsed
            )
            expected_speed = decay_rate**2 * elapsed * math.exp(-decay_rate * elapsed)
            self.assertAlmostEqual(
                float(player.root_position[2]), expected_height, places=5
            )
            self.assertAlmostEqual(
                float(player.root_linear_velocity[2]), expected_speed, places=5
            )
            self.assertEqual(player.time, player.duration)

    def test_quaternion_signs_do_not_change_playback_or_transition(self) -> None:
        normal = SkeletonMotionPlayer(self.make_pair(), transition_half_life=0.1)
        signed = SkeletonMotionPlayer(
            self.make_pair(alternate_quaternion_signs=True), transition_half_life=0.1
        )
        for player in (normal, signed):
            player.advance(dt=0.253)
            player.transition(motion_index=1, time=0.341)
        for dt in (0, 0.013, 0.06, 0.2):
            normal.advance(dt=dt)
            signed.advance(dt=dt)
            self.assert_pose_equal(normal.state, signed.state)
            np.testing.assert_allclose(
                normal.local_angular_velocities,
                signed.local_angular_velocities,
                atol=3e-5,
            )

    def test_disabled_blending_uses_raw_destination_height_and_nonroot_rotations(
        self,
    ) -> None:
        clips = self.make_pair()
        player = SkeletonMotionPlayer(clips, transition_half_life=0)
        player.advance(dt=0.35)
        root_before = np.array(player.root_position)
        heading_before = np.array(player.heading_rotation_wxyz)
        player.transition(motion_index=1, time=0.43)
        np.testing.assert_allclose(player.root_position[:2], root_before[:2], atol=1e-5)
        self.assertAlmostEqual(
            float(player.root_position[2]),
            float(clips[1].sample(time=0.43, loop=False).root_translation().z),
            places=5,
        )
        np.testing.assert_allclose(
            player.heading_rotation_wxyz, heading_before, atol=1e-5
        )
        target = rotations_wxyz(clips[1].sample(time=0.43, loop=False))[1:]
        actual = rotations_wxyz(player.state)[1:]
        np.testing.assert_allclose(
            np.abs(np.sum(target * actual, axis=1)), 1, atol=1e-6
        )

    def test_reset_clears_transition_and_uses_canonical_planar_placement(self) -> None:
        clips = self.make_pair()
        player = SkeletonMotionPlayer(clips)
        player.advance(dt=0.2)
        player.transition(motion_index=1, time=0.3)
        player.advance(dt=0.04)
        player.reset(motion_index=1, time=0.41)
        fresh = SkeletonMotionPlayer(clips)
        fresh.reset(motion_index=1, time=0.41)
        self.assert_pose_equal(player.state, fresh.state)
        np.testing.assert_allclose(player.root_position[:2], 0, atol=1e-6)
        np.testing.assert_allclose(
            player.heading_rotation_wxyz, (1, 0, 0, 0), atol=1e-6
        )

    def test_clamps_nonintegral_fps_at_end_without_looping(self) -> None:
        clip = make_clip(fps=29.97)
        player = SkeletonMotionPlayer([clip])
        duration = (clip.num_frames() - 1) / clip.fps()
        self.assertAlmostEqual(player.duration, duration, places=12)
        player.advance(dt=1000)
        state = player.state
        self.assertAlmostEqual(player.time, duration, places=12)
        self.assert_pose_equal(state, clip.frame(frame_index=clip.num_frames() - 1))
        np.testing.assert_allclose(player.root_linear_velocity, 0, atol=1e-7)
        np.testing.assert_allclose(player.local_angular_velocities, 0, atol=1e-7)
        player.advance(dt=0.2)
        self.assert_pose_equal(state, player.state)
        player.reset(motion_index=0, time=duration)
        self.assertAlmostEqual(player.time, duration, places=12)

    def test_accepts_float_motion_duration_and_clamps_to_exact_player_duration(
        self,
    ) -> None:
        tree = make_tree()
        clip = SkeletonMotion.from_arrays(
            tree,
            ((0, 0, 1), (0.1, 0, 1)),
            np.tile((1.0, 0, 0, 0), (2, 6, 1)),
            30.0,
        )
        player = SkeletonMotionPlayer([clip])
        self.assertGreater(clip.duration(), player.duration)
        for operation in (player.reset, player.transition):
            with self.subTest(operation=operation.__name__):
                operation(motion_index=0, time=clip.duration())
                self.assertEqual(player.time, player.duration)
                self.assertEqual(player.time, 1 / clip.fps())

    def test_exact_half_turn_quaternion_signs_have_identical_pose_and_velocity(
        self,
    ) -> None:
        tree = make_tree()
        clips = []
        for first_sign, last_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rotations = np.zeros((2, 6, 4))
            rotations[0, :, 0] = first_sign
            rotations[1, 0, 3] = last_sign
            rotations[1, 1:, 1] = last_sign
            clips.append(
                SkeletonMotion.from_arrays(
                    tree,
                    ((0, 0, 1), (0.2, 0, 1)),
                    rotations,
                    1.0,
                )
            )
        expected_velocity = np.tile((math.pi, 0, 0), (6, 1))
        expected_velocity[0] = (0, 0, math.pi)
        expected_player = SkeletonMotionPlayer([clips[0]])
        expected_player.advance(dt=0.25)
        for clip in clips:
            player = SkeletonMotionPlayer([clip])
            player.advance(dt=0.25)
            self.assert_pose_equal(player.state, expected_player.state)
            self.assert_pose_equal(
                clip.sample(time=0.25, loop=False),
                clips[0].sample(time=0.25, loop=False),
            )
            np.testing.assert_allclose(
                player.local_angular_velocities, expected_velocity, atol=2e-6
            )
            # Reset at a subframe exercises source heading interpolation.
            player.reset(motion_index=0, time=0.25)
            baseline = SkeletonMotionPlayer([clips[0]])
            baseline.reset(motion_index=0, time=0.25)
            self.assert_pose_equal(player.state, baseline.state)
            np.testing.assert_allclose(
                player.heading_rotation_wxyz, (1, 0, 0, 0), atol=2e-6
            )

    def test_y_up_preserves_height_and_all_joint_poses_during_transition(self) -> None:
        clips = [
            make_clip(up_axis=1, height=1),
            make_clip(up_axis=1, height=2, angles=(0.8,) * 6),
        ]
        player = SkeletonMotionPlayer(clips, up_axis="Y", forward_axis="X")
        player.advance(dt=0.3)
        before = player.state
        player.transition(motion_index=1, time=0.4)
        self.assert_pose_equal(before, player.state)
        player.advance(dt=1)
        self.assertAlmostEqual(float(player.root_position[1]), 2.14, delta=0.005)

    def test_controller_smooths_selected_upper_body_and_root_height_by_default(
        self,
    ) -> None:
        clips = [
            make_clip(velocity=(0, 0, 0), angular_speeds=(0,) * 6),
            make_clip(
                height=1.7,
                velocity=(0, 1, 0),
                angles=(0, -0.5, 0.7, -0.6, -0.9, 0.3),
                angular_speeds=(0,) * 6,
                rotation_axis=0,
            ),
        ]
        database = MotionMatchingDatabase.from_motions(
            clips,
            config=MotionMatchingConfig(
                position_joints=("hand", "foot"),
                velocity_joints=(),
                feature_weights=(0, 0, 1, 0),
            ),
        )
        smooth = MotionMatchingController(database)
        raw = MotionMatchingController(database, transition_half_life=0)
        before = smooth.state
        dt = 1e-4
        smooth_state = smooth.step(dt, desired_velocity=(0, 1, 0))
        raw_state = raw.step(dt, desired_velocity=(0, 1, 0))
        for controller in (smooth, raw):
            self.assertEqual(controller.motion_index, 1)
            self.assertEqual(controller.transition_count, 1)
            self.assertEqual(controller.last_match.motion_index, 1)
        self.assertAlmostEqual(smooth.transition_half_life, 0.1, places=6)
        self.assert_pose_equal(smooth_state, before, atol=1e-3)
        self.assert_pose_equal(smooth_state, smooth.state)
        self.assertAlmostEqual(float(smooth.root_position[2]), 1, delta=1e-4)
        self.assertAlmostEqual(float(raw.root_position[2]), 1.7, places=5)
        original = rotations_wxyz(before)
        smoothed = rotations_wxyz(smooth_state)
        jumped = rotations_wxyz(raw_state)
        # Spine, head, shoulder and hand must all use the blended native pose.
        for joint in (1, 2, 3, 4):
            with self.subTest(joint=joint):
                self.assertGreater(
                    abs(float(original[joint] @ smoothed[joint])), 0.99999
                )
                self.assertLess(abs(float(original[joint] @ jumped[joint])), 0.99)

    def test_changing_half_life_to_zero_clears_live_offsets(self) -> None:
        clips = self.make_pair()
        smooth = SkeletonMotionPlayer(clips, transition_half_life=0.2)
        raw = SkeletonMotionPlayer(clips, transition_half_life=0)
        for player in (smooth, raw):
            player.advance(dt=0.3)
            player.transition(motion_index=1, time=0.5)
            player.advance(dt=0.04)
        before = smooth.state
        before_velocity = np.array(smooth.root_linear_velocity)
        before_angular = np.array(smooth.local_angular_velocities)
        for value in (-1, math.inf, math.nan):
            with self.subTest(half_life=value), self.assertRaises(ValueError):
                smooth.transition_half_life = value
            self.assertAlmostEqual(smooth.transition_half_life, 0.2, places=6)
            self.assert_pose_equal(smooth.state, before)
            np.testing.assert_array_equal(smooth.root_linear_velocity, before_velocity)
            np.testing.assert_array_equal(
                smooth.local_angular_velocities, before_angular
            )
        self.assertGreater(
            abs(float(smooth.root_position[2] - raw.root_position[2])), 0.5
        )
        smooth.transition_half_life = 0
        self.assert_pose_equal(smooth.state, raw.state)
        np.testing.assert_allclose(
            smooth.root_linear_velocity, raw.root_linear_velocity, atol=1e-6
        )
        np.testing.assert_allclose(
            smooth.local_angular_velocities, raw.local_angular_velocities, atol=1e-6
        )
        smooth.advance(dt=0.1)
        raw.advance(dt=0.1)
        self.assert_pose_equal(smooth.state, raw.state)

    def test_output_states_and_arrays_are_independent_snapshots(self) -> None:
        player = SkeletonMotionPlayer([make_clip()])
        snapshot = player.state
        snapshot_matrices = snapshot.compute_global_matrices().copy()
        snapshot.set_root_translation(translation=(99, 98, 97))
        snapshot.set_rotation(index=4, rotation=(0, 0, 0, 1))
        np.testing.assert_allclose(
            player.state.compute_global_matrices(), snapshot_matrices, atol=1e-6
        )
        player.advance(dt=0.02)
        untouched = player.state
        untouched_matrices = untouched.compute_global_matrices().copy()
        for name in (
            "root_position",
            "heading_rotation_wxyz",
            "root_linear_velocity",
            "local_angular_velocities",
        ):
            value = getattr(player, name)
            value.flat[0] = 99
            self.assertNotEqual(float(getattr(player, name).flat[0]), 99)
        player.advance(dt=0.2)
        np.testing.assert_array_equal(
            untouched.compute_global_matrices(), untouched_matrices
        )

    def test_sample_into_matches_existing_sampling_including_loop_seams(self) -> None:
        clip = make_clip(fps=29.97)
        state = clip.frame(frame_index=0)
        for loop in (False, True):
            for time in (-0.4, 0, 0.137, clip.duration(), clip.duration() + 0.33):
                with self.subTest(loop=loop, time=time):
                    self.assertIsNone(
                        clip.sample_into(time=time, state=state, loop=loop)
                    )
                    self.assert_pose_equal(state, clip.sample(time=time, loop=loop))
        wrong_state = make_clip(tree=make_tree(arm_length=0.9)).frame(frame_index=0)
        with self.assertRaises(ValueError):
            clip.sample_into(time=0.1, state=wrong_state)

    def test_rejects_invalid_constructor_configuration(self) -> None:
        clip = make_clip()
        cases = (
            {"up_axis": "X", "forward_axis": "X"},
            {"up_axis": "Q"},
            {"forward_axis": ""},
            {"transition_half_life": -1},
            {"transition_half_life": math.nan},
            {"transition_half_life": math.inf},
        )
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                SkeletonMotionPlayer([clip], **config)
        with self.assertRaises(ValueError):
            SkeletonMotionPlayer([])
        with self.assertRaises(ValueError):
            SkeletonMotionPlayer([clip, make_clip(tree=make_tree(arm_length=0.7))])

    def test_rejects_nonfinite_pose_data_and_degenerate_quaternions(self) -> None:
        clip = make_clip()
        for field, value in (
            ("fps", math.nan),
            ("fps", math.inf),
            ("roots", math.nan),
            ("rotations", math.inf),
            ("rotations", 0),
        ):
            roots, rotations = (
                clip.root_translations().copy(),
                clip.local_rotations_wxyz().copy(),
            )
            fps = clip.fps()
            if field == "fps":
                fps = value
            elif field == "roots":
                roots[3, 0] = value
            else:
                rotations[4, 3] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                invalid = SkeletonMotion.from_arrays(
                    clip.skeleton_tree, roots, rotations, fps
                )
                SkeletonMotionPlayer([invalid])

    def test_invalid_placement_preserves_playback(self) -> None:
        clips = self.make_pair()
        player = SkeletonMotionPlayer(clips)
        baseline = SkeletonMotionPlayer(clips)
        player.advance(dt=0.31)
        baseline.advance(dt=0.31)
        for operation in (player.reset_placed, player.transition_placed):
            for rotation, translation in (
                ((0, 0, 0, 0), (0, 0, 0)),
                ((1, 0, 0, 0), (math.nan, 0, 0)),
            ):
                with self.subTest(operation=operation.__name__, rotation=rotation):
                    with self.assertRaises(ValueError):
                        operation(
                            motion_index=1,
                            time=0.7,
                            rotation_wxyz=rotation,
                            translation=translation,
                        )
                    self.assertEqual(player.motion_index, baseline.motion_index)
                    self.assertEqual(player.time, baseline.time)
                    self.assert_pose_equal(player.state, baseline.state)
                    player.advance(dt=0.1)
                    baseline.advance(dt=0.1)
                    self.assert_pose_equal(player.state, baseline.state)

    def test_invalid_playback_requests_preserve_valid_state(self) -> None:
        player = SkeletonMotionPlayer(self.make_pair())
        player.advance(dt=0.31)
        before = player.state
        for operation in (player.reset, player.transition):
            for index, time in (
                (-1, 0),
                (2, 0),
                (0, -0.1),
                (0, 100),
                (0, math.nan),
                (0, math.inf),
            ):
                with self.subTest(operation=operation.__name__, index=index, time=time):
                    with self.assertRaises((ValueError, IndexError)):
                        operation(motion_index=index, time=time)
                    self.assert_pose_equal(player.state, before)
        for dt in (-0.1, math.nan, math.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                player.advance(dt=dt)
            self.assert_pose_equal(player.state, before)
        for value in (-1, math.nan, math.inf):
            with self.subTest(half_life=value), self.assertRaises(ValueError):
                player.transition_half_life = value
        player.transition_half_life = 0.2
        self.assertAlmostEqual(player.transition_half_life, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
