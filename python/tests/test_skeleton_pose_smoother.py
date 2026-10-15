"""Common transition input works with generated poses, without any motion DB."""

import math
import unittest

import numpy as np

from kangengine.animation import (
    SkeletonPoseSample,
    SkeletonPoseSmoother,
    SkeletonState,
)
from test_skeleton_motion_player import make_tree, angular_difference, rotations_wxyz


def make_sample(
    *,
    time=0.0,
    height=1.0,
    angle=0.0,
    tree=None,
    is_local=True,
    root_velocity=(0.4, 0, 0),
    angular_speed=0.3,
):
    tree = make_tree() if tree is None else tree
    angles = angle + angular_speed * time
    rotations = np.tile((math.cos(angles / 2), math.sin(angles / 2), 0, 0), (6, 1))
    root = np.asarray(root_velocity) * time + (0, 0, height)
    state = SkeletonState.from_rotation_and_root_translation(
        tree=tree, rotations_wxyz=rotations, root_translation=root, is_local=is_local
    )
    return SkeletonPoseSample(
        state,
        root_linear_velocity=root_velocity,
        local_angular_velocities=np.tile((angular_speed, 0, 0), (6, 1)),
    )


class SkeletonPoseSmootherTests(unittest.TestCase):
    def assert_samples_equal(self, first, second):
        np.testing.assert_allclose(
            first.state.compute_global_matrices(),
            second.state.compute_global_matrices(),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            first.root_linear_velocity, second.root_linear_velocity, atol=2e-6
        )
        np.testing.assert_allclose(
            first.local_angular_velocities, second.local_angular_velocities, atol=2e-6
        )

    def test_generated_targets_preserve_pose_velocity_and_analytic_decay(self):
        initial = make_sample(height=1, angular_speed=0, root_velocity=(0, 0, 0))
        target = make_sample(
            height=2, angle=0.8, angular_speed=0, root_velocity=(0, 0, 0)
        )
        smoother = SkeletonPoseSmoother(initial, half_life=0.2)
        smoother.transition(target=target)
        self.assert_samples_equal(smoother.sample, initial)
        elapsed = 0
        for dt in (0.01, 0.1, 0.25, 0.7):
            elapsed += dt
            smoother.advance(target=target, dt=dt)
            decay_rate = 2 * math.log(2) / smoother.half_life
            expected = 2 - (1 + decay_rate * elapsed) * math.exp(-decay_rate * elapsed)
            self.assertAlmostEqual(
                smoother.sample.state.compute_global_matrices()[0, 2, 3],
                expected,
                places=5,
            )
        smoother.half_life = 0
        self.assert_samples_equal(smoother.sample, target)

    def test_moving_target_and_interrupted_transition_derivatives(self):
        smoother = SkeletonPoseSmoother(
            make_sample(angle=-0.5, root_velocity=(1, 0, 0)), half_life=0.2
        )
        smoother.transition(target=make_sample(height=1.8, angle=0.7))
        smoother.advance(
            target=make_sample(time=0.07, height=1.8, angle=0.7), dt=0.07
        )
        before = smoother.sample
        smoother.transition(target=make_sample(angle=1.4))
        self.assert_samples_equal(smoother.sample, before)
        dt = 1e-4
        smoother.advance(target=make_sample(time=dt, angle=1.4), dt=dt)
        after = smoother.sample
        displacement = (
            after.state.compute_global_matrices()[0, :3, 3]
            - before.state.compute_global_matrices()[0, :3, 3]
        )
        np.testing.assert_allclose(
            displacement / dt, before.root_linear_velocity, atol=0.02
        )
        measured = angular_difference(
            rotations_wxyz(before.state), rotations_wxyz(after.state), dt
        )
        np.testing.assert_allclose(measured, before.local_angular_velocities, atol=0.02)

    def test_advance_consumes_target_without_integrating_it_twice(self):
        smoother = SkeletonPoseSmoother(make_sample(), half_life=0)
        target = make_sample(time=0.4)
        smoother.advance(target=target, dt=0.4)
        self.assert_samples_equal(smoother.sample, target)
        smoother.advance(target=make_sample(time=9), dt=0)
        self.assert_samples_equal(smoother.sample, target)
        smoother.reset(target=make_sample(height=5))
        self.assert_samples_equal(smoother.sample, make_sample(height=5))

    def test_input_and_output_snapshots_are_independent(self):
        source = make_sample()
        smoother = SkeletonPoseSmoother(source)
        state = source.state
        state.set_root_translation(translation=(8, 9, 10))
        result = smoother.sample
        result.root_linear_velocity[:] = 100
        result.local_angular_velocities[:] = 100
        result.state.set_root_translation(translation=(10, 10, 10))
        self.assert_samples_equal(smoother.sample, make_sample())

    def test_invalid_targets_and_times_leave_previous_output_unchanged(self):
        initial = make_sample()
        smoother = SkeletonPoseSmoother(initial)
        bad_root = SkeletonPoseSample(
            initial.state,
            root_linear_velocity=(np.nan, 0, 0),
            local_angular_velocities=np.zeros((6, 3)),
        )
        bad_angular = SkeletonPoseSample(
            initial.state,
            root_linear_velocity=(0, 0, 0),
            local_angular_velocities=np.full((6, 3), np.inf),
        )
        bad_rotation = initial.state
        bad_rotation.set_joint_quat_wxyz(index=0, quat_wxyz=(0, 0, 0, 0))
        bad_rotation_sample = SkeletonPoseSample(
            bad_rotation,
            root_linear_velocity=(0, 0, 0),
            local_angular_velocities=np.zeros((6, 3)),
        )
        targets = (
            make_sample(is_local=False),
            make_sample(tree=make_tree(arm_length=0.7)),
            bad_root,
            bad_angular,
            bad_rotation_sample,
        )
        for target in targets:
            for operation in (
                smoother.reset,
                smoother.transition,
                lambda target: smoother.advance(target, 0.1),
            ):
                with (
                    self.subTest(target=target, operation=operation),
                    self.assertRaises(ValueError),
                ):
                    operation(target)
                self.assert_samples_equal(smoother.sample, initial)
        for dt in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                smoother.advance(target=initial, dt=dt)
        for half_life in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                smoother.half_life = half_life
        self.assert_samples_equal(smoother.sample, initial)


if __name__ == "__main__":
    unittest.main()
