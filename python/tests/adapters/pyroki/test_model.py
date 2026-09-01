"""PyRoki URDF model adapter tests without importing the optional backend."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kangengine.animation import (
    IKEffectorMapping,
    MotionSourceProfile,
    IKRetargetConfig,
    IKTargetProfile,
)
from kangengine.adapters.pyroki import PyrokiRobotModel, validate_profile


def _profile(**changes) -> IKRetargetConfig:
    values = {
        "link_mappings": (
            IKEffectorMapping(source_joint="pelvis", target_link="base"),
            IKEffectorMapping(source_joint="left_foot", target_link="foot"),
        ),
        "contact_links": ("foot",),
        "target_root_link": "base",
    }
    joint_order = changes.pop("joint_order", ("hip", "knee"))
    values.update(changes)
    return IKRetargetConfig(
        source_profile=MotionSourceProfile("source"),
        target_profile=IKTargetProfile("robot", Path("robot.urdf"), joint_order),
        source_root_joint="pelvis",
        **values,
    )


def _model() -> PyrokiRobotModel:
    robot = SimpleNamespace(
        forward_kinematics=lambda configuration: np.zeros((3, 7), dtype=np.float32)
    )
    return PyrokiRobotModel(
        urdf=object(),
        robot=robot,
        link_names=("base", "thigh", "foot"),
        actuated_joint_names=("hip", "knee"),
    )


def test_validate_profile_accepts_reordered_canonical_joints():
    validate_profile(_profile(joint_order=("knee", "hip")), _model())


def test_validate_profile_reports_missing_target_link():
    profile = _profile(
        link_mappings=(IKEffectorMapping(source_joint="pelvis", target_link="missing"),)
    )
    with pytest.raises(ValueError, match="missing"):
        validate_profile(profile, _model())


def test_validate_profile_requires_every_actuated_joint():
    with pytest.raises(ValueError, match="omitted: knee"):
        validate_profile(_profile(joint_order=("hip",)), _model())


def test_forward_kinematics_uses_zero_rest_configuration():
    poses = _model().forward_kinematics()
    assert poses.shape == (3, 7)
    assert poses.dtype == np.float32


def test_forward_kinematics_preserves_configuration_batch_axes():
    robot = SimpleNamespace(
        forward_kinematics=lambda configuration: np.zeros(
            (*configuration.shape[:-1], 3, 7), dtype=np.float32
        )
    )
    model = PyrokiRobotModel(
        urdf=object(),
        robot=robot,
        link_names=("base", "thigh", "foot"),
        actuated_joint_names=("hip", "knee"),
    )
    poses = model.forward_kinematics(np.zeros((4, 2), dtype=np.float32))
    assert poses.shape == (4, 3, 7)
