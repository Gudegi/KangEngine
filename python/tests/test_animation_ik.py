import unittest

import numpy as np

from kangengine.animation import SkeletonMotion, SkeletonTree
from kangengine.animation.IK import IKChain, solve_ik_motion


class AnimationIKTests(unittest.TestCase):
    def test_named_motion_solver(self) -> None:
        tree = SkeletonTree(
            ["root", "hand"],
            [-1, 0],
            ((0, 0, 0), (1, 0, 0)),
            ((1, 0, 0, 0), (1, 0, 0, 0)),
        )
        motion = SkeletonMotion.from_arrays(
            tree,
            ((0, 0, 0),),
            (((1, 0, 0, 0), (1, 0, 0, 0)),),
            30.0,
        )
        chain = IKChain("hand", (("root", (0.0, 0.0, 1.0)),))
        result = solve_ik_motion(
            motion,
            np.asarray([[[0.0, 1.0, 0.0]]], np.float32),
            (chain,),
            max_iterations=30,
        )
        self.assertEqual(result.body_positions.shape, (1, 2, 3))
        self.assertEqual(result.final_errors.shape, (1, 1))
        self.assertLess(float(result.final_errors[0, 0]), 0.05)


if __name__ == "__main__":
    unittest.main()
