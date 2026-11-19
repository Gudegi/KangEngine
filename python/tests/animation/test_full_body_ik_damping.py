"""Pose damping balances target accuracy against root displacement."""

import numpy as np

from kangengine.animation import SkeletonMotion, SkeletonTree
from kangengine.animation.IK import IKChain, solve_ik_motion


def test_batch_damping_preserves_default_and_controls_pose_displacement():
    tree = SkeletonTree(['root'], [-1], [[0, 0, 0]], [[1, 0, 0, 0]])
    motion = SkeletonMotion.from_arrays(tree, [[0, 0, 0]], [[[1, 0, 0, 0]]], 30)
    chains = (IKChain('root', (('root', (0, 0, 1)),)),)
    targets = np.array([[[1, 0, 0]]], np.float32)
    default = solve_ik_motion(motion, targets, chains, max_iterations=0)
    explicit = solve_ik_motion(motion, targets, chains, max_iterations=0, pose_damping=0.01)
    strong = solve_ik_motion(motion, targets, chains, max_iterations=0, pose_damping=0.5)
    np.testing.assert_array_equal(default.motion.root_translations(), explicit.motion.root_translations())
    np.testing.assert_allclose(strong.motion.root_translations(), [[0.5, 0, 0]], atol=1e-6)
    assert default.final_errors[0, 0] < strong.final_errors[0, 0]
    np.testing.assert_array_equal(motion.root_translations(), [[0, 0, 0]])
