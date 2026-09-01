from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from kangengine.animation import SkeletonTree, datasets
from kangengine.asset.smpl import SMPL_JOINT_NAMES
from kangengine.asset import AMASSLoader
from kangengine._core import _ke
from kangengine.utils import CoordinateSystem


def _smpl_tree() -> SkeletonTree:
    rotations = np.zeros((24, 4), np.float32)
    rotations[:, 0] = 1.0
    return SkeletonTree(
        [f"joint_{index}" for index in range(24)],
        [-1] + list(range(23)),
        np.zeros((24, 3), np.float32),
        rotations,
    )


def test_loads_amass_from_dataset_uri_and_converts_world():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "amass" / "walk.npz"
        path.parent.mkdir()
        poses = np.zeros((2, 156), np.float32)
        np.savez(
            path,
            poses=poses,
            trans=np.asarray(((1, 2, 3), (4, 5, 6)), np.float32),
            mocap_framerate=np.asarray(60.0),
            gender=np.asarray("neutral"),
            betas=np.zeros(16, np.float32),
        )
        assert datasets.detect_dataset_type(path) is datasets.DatasetType.AMASS
        import os

        previous = os.environ.get("KE_MOTION_DATASETS_DIR")
        os.environ["KE_MOTION_DATASETS_DIR"] = str(root)
        try:
            motion = datasets.load_motion(
                "dataset://amass/walk.npz",
                dataset="amass",
                target_world=CoordinateSystem.Y_UP_Z_FORWARD,
                config=datasets.AMASSConfig(
                    model_type="smpl",
                    skeleton=_smpl_tree(),
                    scale=2.0,
                    root_translation_offset=(10, 20, 30),
                ),
            )
        finally:
            if previous is None:
                os.environ.pop("KE_MOTION_DATASETS_DIR", None)
            else:
                os.environ["KE_MOTION_DATASETS_DIR"] = previous

    assert motion.num_frames() == 2
    assert motion.num_joints() == 24
    assert motion.fps() == 60.0
    np.testing.assert_allclose(motion.root_translations(), ((12, 26, 26), (18, 32, 20)))


def test_amass_rejects_wrong_dataset_config(tmp_path: Path):
    path = tmp_path / "motion.npz"
    try:
        datasets.load_motion(
            path,
            dataset=datasets.DatasetType.AMASS,
            config=datasets.AscendConfig(),
        )
    except TypeError as error:
        assert "AMASSConfig" in str(error)
    else:
        raise AssertionError("AMASS accepted an AscendConfig")


def test_amass_reorders_rotations_to_match_skeleton_joint_names(tmp_path: Path):
    path = tmp_path / "motion.npz"
    poses = np.zeros((1, 156), np.float32)
    for joint in range(22):
        poses[0, joint * 3 + 2] = 0.01 * (joint + 1)
    np.savez(
        path,
        poses=poses,
        trans=np.zeros((1, 3), np.float32),
        mocap_framerate=np.asarray(30.0),
    )

    dfs_names = (
        "pelvis", "left_hip", "left_knee", "left_ankle", "left_foot",
        "right_hip", "right_knee", "right_ankle", "right_foot", "spine1",
        "spine2", "spine3", "neck", "head", "left_collar",
        "left_shoulder", "left_elbow", "left_wrist", "left_hand",
        "right_collar", "right_shoulder", "right_elbow", "right_wrist",
        "right_hand",
    )
    rotations = np.zeros((24, 4), np.float32)
    rotations[:, 0] = 1.0
    canonical_tree = SkeletonTree(
        list(SMPL_JOINT_NAMES), [-1] + [0] * 23, np.zeros((24, 3)), rotations
    )
    dfs_tree = SkeletonTree(
        list(dfs_names), [-1] + [0] * 23, np.zeros((24, 3)), rotations
    )
    canonical = AMASSLoader.load_motion(
        path, canonical_tree, model_type="smpl", up_axis=_ke.UpAxis.Z
    )
    reordered = AMASSLoader.load_motion(
        path, dfs_tree, model_type="smpl", up_axis=_ke.UpAxis.Z
    )

    expected = dict(zip(SMPL_JOINT_NAMES, canonical.local_rotations_wxyz()[0]))
    for index, name in enumerate(dfs_names):
        np.testing.assert_allclose(
            reordered.local_rotations_wxyz()[0, index], expected[name]
        )
