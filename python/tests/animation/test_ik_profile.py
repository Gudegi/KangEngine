from pathlib import Path

import kangengine as ke
import pytest


ROOT = Path(__file__).resolve().parents[3]
SMPL_G1 = ROOT / "assets/retarget/ik/pairs/smpl_to_g1_ik_retarget.json"


def test_shared_profiles_and_pair_config_load():
    config = ke.animation.IKRetargetConfig.load(SMPL_G1)

    assert config.source_profile.name == "smpl"
    assert config.target_profile.name == "g1_29dof"
    assert config.urdf_path == (ROOT / "assets/characters/g1/g1_29dof.urdf").resolve()
    assert len(config.joint_order) == 29
    assert len(config.link_mappings) == 15
    assert config.region_scales["lower_body"] == (0.9, 0.9, 0.85)


def test_pair_config_round_trip_preserves_shared_references(tmp_path: Path):
    config = ke.animation.IKRetargetConfig.load(SMPL_G1)
    output = tmp_path / "copy_ik_retarget.json"

    config.save(output)
    loaded = ke.animation.IKRetargetConfig.load(output)

    assert loaded.to_dict() == config.to_dict()
    assert loaded.source_profile_path == config.source_profile_path
    assert loaded.target_profile_path == config.target_profile_path


def test_profile_suffixes_are_distinct(tmp_path: Path):
    with pytest.raises(ValueError, match="_motion.json"):
        ke.animation.MotionSourceProfile("smpl").save(tmp_path / "smpl.json")
    with pytest.raises(ValueError, match="_ik_target.json"):
        ke.animation.IKTargetProfile("robot", Path("robot.urdf"), ("joint",)).save(
            tmp_path / "robot.json"
        )
    with pytest.raises(ValueError, match="_ik_retarget.json"):
        ke.animation.IKRetargetConfig.load(tmp_path / "pair.json")


def test_pair_scale_is_separate_from_motion_unit_scale():
    config = ke.animation.IKRetargetConfig.load(
        ROOT / "assets/retarget/ik/pairs/mixamo_to_g1_ik_retarget.json"
    )

    assert config.source_profile.translation_unit_scale == 0.01
    assert config.translation_scale == pytest.approx(0.868)
    assert config.source_translation_scale == pytest.approx(0.00868)
