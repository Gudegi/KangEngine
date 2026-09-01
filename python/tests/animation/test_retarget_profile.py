import json
from pathlib import Path
from unittest.mock import patch

import kangengine as ke
import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_mixamo_angle_config_uses_profile_units_for_bind_root():
    motion = json.loads(
        (ROOT / "assets/retarget/motions/mixamo_motion.json").read_text()
    )
    pair = json.loads(
        (
            ROOT
            / "assets/retarget/angle/pairs/mixamo_to_kw5_angle_retarget.json"
        ).read_text()
    )

    assert motion["translation_unit_scale"] == pytest.approx(0.01)
    assert pair["translation_scale"] == pytest.approx(0.95109008)
    assert pair["source_bind_root"] == pytest.approx(
        [6.0e-8, 1.0427047, 0.01554256]
    )


def test_angle_and_ik_share_smpl_motion_profile():
    angle = ke.animation.AngleRetargetConfig.load(
        ROOT / "assets/retarget/angle/pairs/smpl_to_kw5_angle_retarget.json"
    )
    ik = ke.animation.IKRetargetConfig.load(
        ROOT / "assets/retarget/ik/pairs/smpl_to_g1_ik_retarget.json"
    )

    assert angle.source_profile_path == str(ik.source_profile_path)
    assert angle.source_profile == ik.source_profile


def test_skeleton_profile_round_trip(tmp_path: Path):
    reference = tmp_path / "robot.xml"
    profile = ke.animation.AngleTargetProfile("robot", reference, "z_up_x_forward")
    path = tmp_path / "robot_angle_target.json"

    profile.save(path)

    assert ke.animation.AngleTargetProfile.load(path) == profile
    with pytest.raises(ValueError, match="_angle_target.json"):
        profile.save(tmp_path / "robot.json")


def test_motion_profile_preserves_dataset_reference(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    reference = dataset_root / "soma" / "reference.bvh"
    reference.parent.mkdir(parents=True)
    reference.touch()
    profile_path = tmp_path / "soma_motion.json"
    profile_path.write_text(
        '{"version": 1, "name": "soma", '
        '"reference_skeleton": "dataset://soma/reference.bvh"}',
        encoding="utf-8",
    )

    with patch.dict(
        "os.environ", {"KE_MOTION_DATASETS_DIR": str(dataset_root)}, clear=False
    ):
        profile = ke.animation.MotionSourceProfile.load(profile_path)
        profile.save(profile_path)
        reloaded = ke.animation.MotionSourceProfile.load(profile_path)

    assert profile.reference_skeleton == reference.resolve()
    assert reloaded == profile
    assert "dataset://soma/reference.bvh" in profile_path.read_text(encoding="utf-8")
