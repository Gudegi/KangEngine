import unittest

import numpy as np

from kangengine.animation import (
    CoordinateSystem,
    SkeletonMotion,
    SkeletonState,
    SkeletonTree,
    convert_motion_coordinates,
    convert_state_coordinates,
)


class AnimationCoordinateTests(unittest.TestCase):
    def test_z_up_x_forward_to_y_up_z_forward(self) -> None:
        tree = SkeletonTree(
            ["root", "child"],
            [-1, 0],
            ((0, 0, 0), (1, 2, 3)),
            ((1, 0, 0, 0), (1, 0, 0, 0)),
        )
        motion = SkeletonMotion.from_arrays(
            tree,
            ((1, 2, 3),),
            (((1, 0, 0, 0), (1, 0, 0, 0)),),
            30.0,
        )
        converted = convert_motion_coordinates(
            motion,
            source=CoordinateSystem.Z_UP_X_FORWARD,
            target=CoordinateSystem.Y_UP_Z_FORWARD,
        )
        child = converted.skeleton_tree.local_translation(1)
        np.testing.assert_allclose((child.x, child.y, child.z), (2, 3, 1))
        np.testing.assert_allclose(converted.root_translations()[0], (2, 3, 1))

    def test_round_trip(self) -> None:
        tree = SkeletonTree(
            ["root"], [-1], ((0.2, -0.4, 0.8),), ((0.5, 0.5, 0.5, 0.5),)
        )
        motion = SkeletonMotion.from_arrays(
            tree,
            ((1, 2, 3),),
            (((0.5, 0.5, 0.5, 0.5),),),
            60.0,
        )
        converted = convert_motion_coordinates(
            motion,
            source=CoordinateSystem.Y_UP_Z_FORWARD,
            target=CoordinateSystem.Z_UP_X_FORWARD,
        )
        restored = convert_motion_coordinates(
            converted,
            source=CoordinateSystem.Z_UP_X_FORWARD,
            target=CoordinateSystem.Y_UP_Z_FORWARD,
        )
        np.testing.assert_allclose(
            restored.root_translations(), motion.root_translations(), atol=1e-6
        )
        np.testing.assert_allclose(
            restored.local_rotations_wxyz(),
            motion.local_rotations_wxyz(),
            atol=1e-6,
        )

    def test_y_up_negative_z_forward(self) -> None:
        tree = SkeletonTree(
            ["root"], [-1], ((1, 2, 3),), ((1, 0, 0, 0),)
        )
        motion = SkeletonMotion.from_arrays(
            tree, ((1, 2, 3),), (((1, 0, 0, 0),),), 30.0
        )
        converted = convert_motion_coordinates(
            motion,
            source=CoordinateSystem.Y_UP_NEG_Z_FORWARD,
            target=CoordinateSystem.Y_UP_Z_FORWARD,
        )
        np.testing.assert_allclose(converted.root_translations()[0], (-1, 2, -3))

    def test_state_uses_converted_skeleton(self) -> None:
        tree = SkeletonTree(
            ["root"], [-1], ((0, 0, 0),), ((1, 0, 0, 0),)
        )
        state = SkeletonState.from_rotation_and_root_translation(
            tree, ((1, 0, 0, 0),), (1, 2, 3)
        )
        converted = convert_state_coordinates(
            state,
            source=CoordinateSystem.Z_UP_X_FORWARD,
            target=CoordinateSystem.Y_UP_Z_FORWARD,
        )
        root = converted.root_translation()
        np.testing.assert_allclose((root.x, root.y, root.z), (2, 3, 1))


if __name__ == "__main__":
    unittest.main()
