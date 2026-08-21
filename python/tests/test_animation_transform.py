import math
import unittest

import numpy as np

from kangengine.animation import SkeletonMotion, SkeletonTree, transform_motion


class AnimationTransformTests(unittest.TestCase):
    def test_world_transform_preserves_shape(self) -> None:
        tree = SkeletonTree(
            ["root", "child"],
            [-1, 0],
            ((0, 0, 0), (1, 0, 0)),
            ((1, 0, 0, 0), (1, 0, 0, 0)),
        )
        motion = SkeletonMotion.from_arrays(
            tree,
            ((2, 0, 0),),
            (((1, 0, 0, 0), (1, 0, 0, 0)),),
            30.0,
            "test",
        )
        half = math.sqrt(0.5)
        transformed = transform_motion(
            motion,
            rotation_wxyz=(half, 0, 0, half),
            translation=(0, 1, 0),
            pivot=(1, 0, 0),
        )

        positions = np.asarray(transformed.global_positions())
        np.testing.assert_allclose(positions[0, 0], (1, 2, 0), atol=1e-6)
        np.testing.assert_allclose(positions[0, 1], (1, 3, 0), atol=1e-6)
        self.assertEqual(transformed.motion_name(), "test")
        self.assertEqual(transformed.fps(), 30.0)

    def test_rejects_zero_rotation(self) -> None:
        tree = SkeletonTree(["root"], [-1], ((0, 0, 0),), ((1, 0, 0, 0),))
        motion = SkeletonMotion.from_arrays(
            tree, ((0, 0, 0),), (((1, 0, 0, 0),),), 30.0
        )
        with self.assertRaises(ValueError):
            transform_motion(motion, rotation_wxyz=(0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
