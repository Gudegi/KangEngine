from __future__ import annotations

import json
import unittest

import numpy as np

from kangengine.animation import (
    RetargetConfig,
    Retargeter,
    SkeletonMotion,
    SkeletonTree,
    retarget_motion,
)


def _tree(bind_rotations: np.ndarray | None = None) -> SkeletonTree:
    if bind_rotations is None:
        bind_rotations = np.array(((1, 0, 0, 0), (1, 0, 0, 0)), np.float32)
    return SkeletonTree(
        ["root", "child"],
        [-1, 0],
        np.array(((0, 0, 0), (0, 1, 0)), np.float32),
        bind_rotations,
    )


def _motion(tree: SkeletonTree) -> SkeletonMotion:
    rotations = np.array(
        (
            ((1, 0, 0, 0), (1, 0, 0, 0)),
            ((1, 0, 0, 0), (0.9238795, 0, 0, 0.3826834)),
        ),
        np.float32,
    )
    return SkeletonMotion.from_arrays(
        tree,
        np.array(((1, 2, 3), (2, 2, 3)), np.float32),
        rotations,
        30.0,
        "source",
    )


class RetargetTest(unittest.TestCase):
    def test_retargeter_handles_live_pose_and_state(self) -> None:
        tree = _tree()
        source = _motion(tree)
        retargeter = Retargeter(
            tree,
            tree,
            RetargetConfig(joint_map={"root": "root", "child": "child"}),
        )

        pose = retargeter.retarget_pose(
            source.root_translations()[1], source.local_rotations_wxyz()[1]
        )
        state = retargeter.retarget_state(source.frame(1))

        for result in (pose, state):
            root = result.root_translation()
            np.testing.assert_allclose((root.x, root.y, root.z), (2, 2, 3))
            child = result.rotation(1)
            np.testing.assert_allclose(
                (child.w, child.x, child.y, child.z),
                source.local_rotations_wxyz()[1, 1],
                atol=1.0e-6,
            )

    def test_identity_retarget_preserves_motion(self) -> None:
        tree = _tree()
        source = _motion(tree)
        config = RetargetConfig(joint_map={"root": "root", "child": "child"})

        result = retarget_motion(source, tree, config)

        np.testing.assert_allclose(
            result.root_translations(), source.root_translations()
        )
        np.testing.assert_allclose(
            result.local_rotations_wxyz(), source.local_rotations_wxyz(), atol=1.0e-6
        )

    def test_unmapped_joint_keeps_target_bind_rotation(self) -> None:
        half_angle = np.pi / 4.0
        target_bind = np.array(
            ((1, 0, 0, 0), (np.cos(half_angle), np.sin(half_angle), 0, 0)),
            np.float32,
        )
        source = _motion(_tree())
        target = _tree(target_bind)
        config = RetargetConfig(joint_map={"root": "root"})

        result = retarget_motion(source, target, config)

        np.testing.assert_allclose(
            result.local_rotations_wxyz()[:, 1],
            np.repeat(target_bind[None, 1], source.num_frames(), axis=0),
            atol=1.0e-6,
        )

    def test_source_bind_pose_maps_to_target_bind_pose(self) -> None:
        source_bind = np.array(((1, 0, 0, 0), (0.7071068, 0.7071068, 0, 0)), np.float32)
        target_bind = np.array(((1, 0, 0, 0), (0.7071068, 0, 0.7071068, 0)), np.float32)
        source_tree = _tree(source_bind)
        target_tree = _tree(target_bind)
        source = SkeletonMotion.from_arrays(
            source_tree,
            np.zeros((1, 3), np.float32),
            source_bind[None],
            30.0,
            "bind",
        )
        config = RetargetConfig(joint_map={"root": "root", "child": "child"})

        result = retarget_motion(source, target_tree, config)

        np.testing.assert_allclose(
            result.local_rotations_wxyz()[0], target_bind, atol=1.0e-6
        )

    def test_root_translation_is_bind_relative(self) -> None:
        tree = _tree()
        source = _motion(tree)
        config = RetargetConfig(
            joint_map={"root": "root"},
            source_bind_root=(1, 2, 3),
            target_bind_root=(10, 20, 30),
            translation_scale=2.0,
        )

        result = retarget_motion(source, tree, config)

        np.testing.assert_allclose(
            result.root_translations(), ((10, 20, 30), (12, 20, 30))
        )

    def test_config_round_trip_requires_retarget_suffix(self) -> None:
        config = RetargetConfig(
            joint_map={"root": "pelvis"},
            source_bind_local_wxyz={"root": (2, 0, 0, 0)},
            source_skeleton="source.fbx",
            target_skeleton="target.xml",
        )
        with self.subTest("round trip"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "source_to_target_retarget.json"
                self.assertEqual(config.save(path), path)
                self.assertEqual(RetargetConfig.load(path), config)
                self.assertEqual(json.loads(path.read_text())["version"], 1)
                with self.assertRaisesRegex(ValueError, "_retarget.json"):
                    config.save(Path(directory) / "wrong.json")


if __name__ == "__main__":
    unittest.main()
