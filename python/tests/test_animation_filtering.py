import unittest

import numpy as np

from kangengine.animation import SkeletonMotion, SkeletonTree
from kangengine.animation.filter import (
    gaussian_filter_time,
    gaussian_smooth_motion,
    smooth_rotation_correction,
)


def _motion(rotations: np.ndarray) -> SkeletonMotion:
    tree = SkeletonTree(["root"], [-1], ((0, 0, 0),), ((1, 0, 0, 0),))
    roots = np.zeros((len(rotations), 3), np.float32)
    return SkeletonMotion.from_arrays(tree, roots, rotations[:, None], 30.0, "test")


class AnimationFilteringTests(unittest.TestCase):
    def test_gaussian_filter_reduces_impulse(self) -> None:
        values = np.zeros((9, 1), np.float32)
        values[4] = 1.0
        filtered = gaussian_filter_time(values, 1.0)
        self.assertLess(float(filtered[4, 0]), 1.0)
        self.assertGreater(float(filtered[3, 0]), 0.0)

    def test_motion_filters_preserve_unit_quaternions(self) -> None:
        rotations = np.asarray(
            [(1, 0, 0, 0), (0.9238795, 0, 0.3826834, 0), (1, 0, 0, 0)],
            np.float32,
        )
        reference = _motion(np.tile((1, 0, 0, 0), (3, 1)).astype(np.float32))
        corrected = _motion(rotations)
        for result in (
            gaussian_smooth_motion(corrected, sigma=1.0),
            smooth_rotation_correction(reference, corrected, sigma=1.0),
        ):
            norms = np.linalg.norm(result.local_rotations_wxyz(), axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
