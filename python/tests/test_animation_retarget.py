from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from kangengine.animation import (
    AngleRetargetConfig,
    AngleRetargeter,
    AngleTargetProfile,
    MotionSourceProfile,
    SkeletonMotion,
    SkeletonTree,
    retarget_angle_motion,
    scale_skeleton_motion,
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
    def test_scale_skeleton_motion_scales_offsets_and_root_translation(self) -> None:
        source = _motion(_tree())

        result = scale_skeleton_motion(source, 2.0)

        child_offset = result.skeleton_tree.local_translation(1)
        np.testing.assert_allclose(
            (child_offset.x, child_offset.y, child_offset.z), (0, 2, 0)
        )
        np.testing.assert_allclose(
            result.root_translations(), source.root_translations() * 2.0
        )
        np.testing.assert_allclose(
            result.local_rotations_wxyz(), source.local_rotations_wxyz()
        )

    def test_retargeter_handles_live_pose_and_state(self) -> None:
        tree = _tree()
        source = _motion(tree)
        retargeter = AngleRetargeter(
            tree,
            tree,
            AngleRetargetConfig(joint_map={"root": "root", "child": "child"}),
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
        config = AngleRetargetConfig(joint_map={"root": "root", "child": "child"})

        result = retarget_angle_motion(source, tree, config)

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
        config = AngleRetargetConfig(joint_map={"root": "root"})

        result = retarget_angle_motion(source, target, config)

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
        config = AngleRetargetConfig(joint_map={"root": "root", "child": "child"})

        result = retarget_angle_motion(source, target_tree, config)

        np.testing.assert_allclose(
            result.local_rotations_wxyz()[0], target_bind, atol=1.0e-6
        )

    def test_root_translation_is_bind_relative(self) -> None:
        tree = _tree()
        source = _motion(tree)
        config = AngleRetargetConfig(
            joint_map={"root": "root"},
            source_bind_root=(1, 2, 3),
            target_bind_root=(10, 20, 30),
            translation_scale=2.0,
        )

        result = retarget_angle_motion(source, tree, config)

        np.testing.assert_allclose(
            result.root_translations(), ((10, 20, 30), (12, 20, 30))
        )

    def test_config_round_trip_requires_retarget_suffix(self) -> None:
        config = AngleRetargetConfig(
            joint_map={"root": "pelvis"},
            source_bind_local_wxyz={"root": (2, 0, 0, 0)},
            source_profile=MotionSourceProfile(
                "source",
                translation_unit_scale=0.01,
                reference_skeleton=Path("source.fbx"),
            ),
            target_profile=AngleTargetProfile("target", Path("target.xml")),
        )
        with self.subTest("round trip"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "source_to_target_angle_retarget.json"
                self.assertEqual(config.save(path), path)
                self.assertEqual(AngleRetargetConfig.load(path), config)
                data = json.loads(path.read_text())
                self.assertEqual(data["version"], 2)
                self.assertEqual(data["source_profile"], "motions/source_motion.json")
                self.assertEqual(
                    data["target_profile"], "targets/target_angle_target.json"
                )
                with self.assertRaisesRegex(ValueError, "_retarget.json"):
                    config.save(Path(directory) / "wrong.json")


if __name__ == "__main__":
    unittest.main()
