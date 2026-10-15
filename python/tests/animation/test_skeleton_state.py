"""Quaternion ordering and node selection at the SkeletonState boundary."""

import unittest

import numpy as np
import torch

import kangengine as ke
from kangengine.animation import SkeletonState, SkeletonTree


class SkeletonStateQuaternionTest(unittest.TestCase):
    def make_state(self, *, is_local=True):
        identity = np.tile((1.0, 0.0, 0.0, 0.0), (2, 1))
        tree = SkeletonTree(
            ["root", "child"], [-1, 0],
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), identity,
        )
        return SkeletonState.from_rotation_and_root_translation(
            tree=tree, rotations_wxyz=identity,
            root_translation=(0, 0, 0), is_local=is_local,
        )

    def test_wxyz_inputs_update_only_the_selected_node(self):
        quaternion = np.array([1, 2, 3, 4], dtype=np.float32)
        quaternion /= np.linalg.norm(quaternion)
        inputs = (
            quaternion, quaternion.tolist(), torch.from_numpy(quaternion),
            ke.Quat.from_wxyz(quaternion),
        )
        for is_local in (True, False):
            for index in (0, 1):
                for value in inputs:
                    with self.subTest(is_local=is_local, index=index, type=type(value)):
                        state = self.make_state(is_local=is_local)
                        state.set_joint_quat_wxyz(index=index, quat_wxyz=value)
                        np.testing.assert_allclose(
                            np.asarray(state.get_joint_quat(index)), quaternion, atol=1e-6,
                        )
                        np.testing.assert_array_equal(
                            np.asarray(state.get_joint_quat(1 - index)), [1, 0, 0, 0],
                        )
                        self.assertEqual(state.is_local(), is_local)

    def test_local_joint_rotation_composes_with_parent(self):
        state = self.make_state()
        half = 2 ** -0.5
        state.set_joint_quat_wxyz(index=0, quat_wxyz=(half, 0, 0, half))
        state.set_joint_quat_wxyz(index=1, quat_wxyz=(half, half, 0, 0))

        child = state.compute_global_matrices()[1]
        np.testing.assert_allclose(child[:3, 3], [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(
            child[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-6,
        )

    def test_ambiguous_setter_is_removed(self):
        self.assertFalse(hasattr(SkeletonState, "set_rotation"))


if __name__ == "__main__":
    unittest.main()
