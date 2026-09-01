"""Backend-independent articulation IK retarget contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch

from kangengine.animation import (
    IKEffectorMapping,
    MotionSourceProfile,
    IKRetargetConfig,
    IKTargetProfile,
)
from kangengine.animation.retarget import build_ik_target_motion, detect_foot_contacts


def _mapping() -> IKEffectorMapping:
    return IKEffectorMapping(source_joint="left_foot", target_link="left_ankle")


def _profile(*, link_mappings=(_mapping(),), joint_order=("joint",), **values):
    robot_values = {}
    for name in ("rest_configuration", "joint_rest_weights"):
        if name in values:
            robot_values[name] = values.pop(name)
    return IKRetargetConfig(
        source_profile=MotionSourceProfile("source"),
        target_profile=IKTargetProfile(
            "robot", Path("robot.urdf"), joint_order, **robot_values
        ),
        link_mappings=link_mappings,
        source_root_joint=values.pop("source_root_joint", "pelvis"),
        target_root_link=values.pop("target_root_link", "pelvis"),
        **values,
    )


def test_robot_profile_rejects_rest_configuration_with_wrong_size():
    with pytest.raises(ValueError, match="rest_configuration must contain"):
        _profile(
            joint_order=("hip", "knee"),
            rest_configuration=(0.0,),
        )


def test_robot_profile_copies_region_scale_mapping():
    scales = {"leg": (0.9, 0.9, 0.85)}
    profile = _profile(
        region_scales=scales,
    )
    scales["leg"] = (1.0, 1.0, 1.0)
    assert profile.region_scales["leg"] == (0.9, 0.9, 0.85)


class _FakeMotion:
    def node_names(self):
        return ["pelvis", "knee", "foot"]

    def global_positions(self):
        return np.asarray(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, -1.0], [1.0, 0.0, -2.0]],
                [[2.0, 0.0, 0.0], [2.0, 0.0, -1.0], [2.0, 0.0, -2.0]],
            ],
            dtype=np.float32,
        )

    def global_rotations_wxyz(self):
        rotations = np.zeros((2, 3, 4), dtype=np.float32)
        rotations[..., 0] = 1.0
        return rotations

    def fps(self):
        return 50.0


def test_target_builder_preserves_root_and_scales_root_relative_positions():
    profile = _profile(
        link_mappings=(
            IKEffectorMapping(source_joint="pelvis", target_link="base"),
            IKEffectorMapping(source_joint="foot", target_link="ankle", region="leg"),
        ),
        region_scales={"leg": (1.0, 1.0, 0.5)},
        target_root_link="base",
    )
    contacts = torch.tensor([[False, True], [False, False]])

    targets = build_ik_target_motion(_FakeMotion(), profile, contact_mask=contacts)

    assert targets.fps == 50.0
    assert targets.positions.shape == (2, 2, 3)
    assert targets.target_offsets.shape == (2, 3)
    assert targets.primary_effector_count == 2
    assert targets.initial_root_rotations_wxyz.shape == (2, 4)
    assert torch.allclose(
        targets.positions[:, 0], torch.tensor([[1.0, 0, 0], [2.0, 0, 0]])
    )
    assert torch.allclose(
        targets.positions[:, 1], torch.tensor([[1.0, 0, -1], [2.0, 0, -1]])
    )
    assert targets.contact_mask is contacts


def test_target_builder_reports_missing_source_joint():
    profile = _profile(
        link_mappings=(IKEffectorMapping("missing", "base"),),
    )
    with pytest.raises(ValueError, match="missing"):
        build_ik_target_motion(_FakeMotion(), profile)


def test_target_builder_applies_local_rotation_and_position_offsets():
    profile = _profile(
        link_mappings=(
            IKEffectorMapping(
                "foot",
                "ankle",
                position_offset=(1.0, 0.0, 0.0),
                rotation_offset_wxyz=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )
    targets = build_ik_target_motion(_FakeMotion(), profile)
    assert torch.allclose(
        targets.positions[0, 0], torch.tensor([0.0, 0.0, -2.0]), atol=1e-6
    )
    assert torch.allclose(
        targets.rotations_wxyz[0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0])
    )


def test_contact_detector_marks_low_stationary_mapped_foot():
    class ContactMotion:
        def node_names(self):
            return ["pelvis", "foot"]

        def global_positions(self):
            return np.asarray(
                [
                    [[0, 0, 1], [0, 0, 0.01]],
                    [[0, 0, 1], [0, 0, 0.01]],
                    [[0, 0, 1], [0.2, 0, 0.20]],
                ],
                dtype=np.float32,
            )

        def fps(self):
            return 30.0

    profile = _profile(
        link_mappings=(
            IKEffectorMapping("pelvis", "base"),
            IKEffectorMapping("foot", "ankle"),
        ),
        target_root_link="base",
        contact_links=("ankle",),
    )
    contacts = detect_foot_contacts(ContactMotion(), profile)
    assert contacts.shape == (3, 2)
    assert contacts[:, 0].tolist() == [0.0, 0.0, 0.0]
    assert contacts[0, 1] > contacts[2, 1] > 0.0
