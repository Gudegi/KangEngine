from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kangengine.animation import SkeletonTree, datasets
from kangengine.utils import CoordinateSystem


def _tree(joints: int = 24) -> SkeletonTree:
    parents = [-1] + list(range(joints - 1))
    translations = np.zeros((joints, 3), np.float32)
    rotations = np.zeros((joints, 4), np.float32)
    rotations[:, 0] = 1.0
    return SkeletonTree(
        [f"joint_{index}" for index in range(joints)],
        parents,
        translations,
        rotations,
    )


class AscendMotionTest(unittest.TestCase):
    def test_resolves_relative_path_from_motion_datasets_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "ascend" / "clip.npy"
            expected.parent.mkdir()
            expected.touch()
            with patch.dict(
                "os.environ", {"KE_MOTION_DATASETS_DIR": str(root)}, clear=False
            ):
                resolved = datasets.resolve_dataset_path("ascend/clip.npy")

        self.assertEqual(resolved, expected.resolve())

    def test_unresolved_relative_path_explains_dataset_environment(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "KE_MOTION_DATASETS_DIR"):
                datasets.resolve_dataset_path("missing/clip.npy")

    def test_dataset_loader_preserves_native_or_converts_world(self) -> None:
        poses = np.zeros((1, 72), np.float32)
        poses[0, 3:6] = (0.0, 0.0, np.pi / 2.0)
        person = {
            "opt_pose": poses,
            "opt_trans": np.asarray(((1.0, 2.0, 3.0),), np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ascend.npy"
            np.save(path, {"second_person": person}, allow_pickle=True)
            config = datasets.AscendConfig(skeleton=_tree())
            native = datasets.load_motion(
                path,
                dataset=datasets.DatasetType.ASCEND,
                config=config,
            )
            y_up = datasets.load_motion(
                path,
                dataset="ascend",
                target_world=CoordinateSystem.Y_UP_Z_FORWARD,
                config=config,
            )

        np.testing.assert_allclose(native.root_translations(), ((1, 2, 3),))
        np.testing.assert_allclose(
            native.local_rotations_wxyz()[0, 0], (1, 0, 0, 0), atol=1.0e-6
        )
        np.testing.assert_allclose(y_up.root_translations(), ((1, 3, -2),))
        np.testing.assert_allclose(
            y_up.local_rotations_wxyz()[0, 0],
            (np.sqrt(0.5), -np.sqrt(0.5), 0, 0),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            y_up.local_rotations_wxyz()[0, 1],
            native.local_rotations_wxyz()[0, 1],
            atol=1.0e-6,
        )

    def test_loads_pickled_dictionary_and_reexpresses_root(self) -> None:
        person = {
            "opt_pose": np.zeros((2, 72), np.float32),
            "opt_trans": np.asarray(((1, 2, 3), (4, 5, 6)), np.float32),
            "mocap_frame_rate": 60.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ascend.pkl"
            with path.open("wb") as stream:
                np.save(stream, {"second_person": person}, allow_pickle=True)
            motion = datasets.load_motion(
                path,
                dataset="ascend",
                target_world=CoordinateSystem.Y_UP_Z_FORWARD,
                config=datasets.AscendConfig(
                    skeleton=_tree(),
                    root_translation_offset=(10, 20, 30),
                ),
            )

        self.assertEqual(motion.num_frames(), 2)
        self.assertEqual(motion.num_joints(), 24)
        self.assertEqual(motion.fps(), 60.0)
        np.testing.assert_allclose(
            motion.root_translations(), ((11, 23, 28), (14, 26, 25))
        )
        expected = np.zeros((2, 24, 4), np.float32)
        expected[..., 0] = 1.0
        expected[:, 0] = (
            np.sqrt(0.5),
            -np.sqrt(0.5),
            0.0,
            0.0,
        )
        np.testing.assert_allclose(motion.local_rotations_wxyz(), expected, atol=1.0e-6)

    def test_converts_root_rotation_but_preserves_child_local_rotation(self) -> None:
        poses = np.zeros((1, 72), np.float32)
        poses[0, 3:6] = (0.0, 0.0, np.pi / 2.0)
        person = {
            "opt_pose": poses,
            "opt_trans": np.zeros((1, 3), np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ascend.npy"
            np.save(path, {"second_person": person}, allow_pickle=True)
            motion = datasets.load_motion(
                path,
                dataset="ascend",
                target_world=CoordinateSystem.Y_UP_Z_FORWARD,
                config=datasets.AscendConfig(skeleton=_tree()),
            )

        rotations = motion.local_rotations_wxyz()[0]
        np.testing.assert_allclose(
            rotations[0],
            (np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            rotations[1],
            (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
            atol=1.0e-6,
        )

    def test_loads_npz_object_person(self) -> None:
        person = {
            "opt_pose": np.zeros((1, 72), np.float32),
            "opt_trans": np.zeros((1, 3), np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ascend.npz"
            np.savez(path, second_person=np.asarray(person, dtype=object))
            motion = datasets.load_motion(
                path,
                dataset="ascend",
                config=datasets.AscendConfig(skeleton=_tree(), default_fps=24.0),
            )
        self.assertEqual(motion.num_frames(), 1)
        self.assertEqual(motion.fps(), 24.0)


if __name__ == "__main__":
    unittest.main()
