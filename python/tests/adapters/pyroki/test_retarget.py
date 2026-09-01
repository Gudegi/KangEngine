"""Canonical packing tests for PyRoki offline retargeting."""

from pathlib import Path
from importlib import import_module
from dataclasses import replace

import numpy as np
import torch

import kangengine as ke
from kangengine.animation.retarget import IKTargetMotion
from kangengine.adapters.pyroki import (
    PyrokiPoseIKConfig,
    PyrokiTrajectoryIKConfig,
    PyrokiTrajectoryIKResult,
    PyrokiTrajectoryState,
    solve_trajectory_ik_windowed,
    trajectory_result_to_articulation_motion,
)


ROOT = Path(__file__).resolve().parents[4]


def _profile(joint_order: tuple[str, ...]):
    return ke.animation.IKRetargetConfig(
        source_profile=ke.animation.MotionSourceProfile("source"),
        target_profile=ke.animation.IKTargetProfile(
            "robot", Path("robot.urdf"), joint_order
        ),
        link_mappings=(ke.animation.IKEffectorMapping("root", "root"),),
        source_root_joint="root",
        target_root_link="root",
    )


def test_ik_configs_expose_orientation_weight():
    assert PyrokiPoseIKConfig().orientation_weight == 0.0
    trajectory = PyrokiTrajectoryIKConfig()
    assert trajectory.orientation_weight == 0.0
    assert trajectory.whole_trajectory_max_frames == 1000
    assert trajectory.window_size == 512
    assert trajectory.window_overlap == 32
    assert trajectory.max_iterations == 800
    assert trajectory.joint_rest_weight == 0.02
    assert trajectory.foot_skating_weight == 30.0
    assert trajectory.foot_tilt_weight == 1.0


def test_g1_profile_maps_one_human_ankle_target_per_robot_foot():
    profile = ke.animation.IKRetargetConfig.load(
        ROOT / "assets/retarget/ik/pairs/smpl_to_g1_ik_retarget.json"
    )
    mappings = {mapping.target_link: mapping for mapping in profile.link_mappings}

    assert mappings["left_ankle_roll_link"].source_joint == "left_ankle"
    assert mappings["right_ankle_roll_link"].source_joint == "right_ankle"
    assert mappings["left_foot_link"].source_joint == "left_ankle"
    assert mappings["right_foot_link"].source_joint == "right_ankle"
    assert "left_ankle_pitch_link" not in mappings
    assert "right_ankle_pitch_link" not in mappings
    assert profile.root_height_offset == 0.02
    assert profile.target_root_link == "pelvis_contour_link"
    assert profile.joint_rest_weights["waist_roll_joint"] == 1.0
    assert profile.joint_rest_weights["left_wrist_pitch_joint"] == 1.0
    assert profile.joint_rest_weights["right_wrist_pitch_joint"] == 1.0
    assert [point.name for point in profile.auxiliary_points] == [
        "left_hand_aux",
        "right_hand_aux",
        "pelvis_aux",
    ]
    assert len(profile.local_alignment_pairs) == 10
    assert ("left_ankle_roll_link", "left_foot_link") in profile.local_alignment_pairs
    assert profile.rest_configuration == (0.0,) * len(profile.joint_order)


def test_g1_pair_config_accepts_scale_override():
    profile = ke.animation.IKRetargetConfig.load(
        ROOT / "assets/retarget/ik/pairs/smpl_to_g1_ik_retarget.json"
    )
    profile = replace(
        profile,
        translation_scale=0.75,
        region_scales={"lower_body": (0.8, 0.8, 0.75)},
    )

    assert profile.rest_configuration == (0.0,) * len(profile.joint_order)
    assert profile.source_translation_scale == 0.75
    assert profile.region_scales["lower_body"] == (0.8, 0.8, 0.75)


def test_trajectory_result_packs_canonical_q_and_qd(tmp_path):
    asset_path = Path(__file__).parents[4] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(asset_path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=True
    )
    joint_names = tuple(
        block.joint_name
        for block in layout.blocks
        if block.type
        in (
            ke.animation.ArticulationCoordinateType.REVOLUTE,
            ke.animation.ArticulationCoordinateType.PRISMATIC,
        )
    )
    profile = _profile(joint_names)
    joints = np.zeros((2, len(joint_names)), dtype=np.float32)
    joints[1] = 0.1
    solve = PyrokiTrajectoryIKResult(
        root_positions=np.asarray([[0, 0, 1], [0.2, 0, 1]], dtype=np.float32),
        root_rotations_wxyz=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        joint_configurations=joints,
        initial_position_rmse=1.0,
        position_rmse=0.1,
        iterations=5,
        converged=True,
    )

    motion = trajectory_result_to_articulation_motion(solve, profile, layout, fps=20.0)

    assert motion.q.shape == (2, layout.nq)
    assert motion.qd.shape == (2, layout.nv)
    np.testing.assert_allclose(motion.q[:, :3], [[0, 0, 1], [0.2, 0, 1]])
    np.testing.assert_allclose(motion.qd[:, 0], [4.0, 4.0])
    assert np.isfinite(motion.q).all()
    assert np.isfinite(motion.qd).all()

    saved = ke.animation.save_articulation_motion_npz(motion, tmp_path / "motion.npz")
    restored = ke.animation.load_articulation_motion_npz(saved, layout)
    np.testing.assert_allclose(restored.q, motion.q)
    np.testing.assert_allclose(restored.qd, motion.qd)
    assert restored.fps() == motion.fps()


def test_windowed_solver_blends_preallocated_output(monkeypatch):
    module = import_module("kangengine.adapters.pyroki.retarget")
    calls = []

    def fake_solve(model, profile, targets, **kwargs):
        del model, profile
        calls.append(kwargs)
        value = float(len(calls))
        frames = targets.num_frames
        return PyrokiTrajectoryIKResult(
            root_positions=np.full((frames, 3), value, dtype=np.float32),
            root_rotations_wxyz=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (frames, 1),
            ),
            joint_configurations=np.full((frames, 1), value, dtype=np.float32),
            initial_position_rmse=value,
            position_rmse=value * 0.1,
            iterations=1,
            converged=True,
        )

    monkeypatch.setattr(module, "solve_trajectory_ik", fake_solve)
    targets = IKTargetMotion(
        fps=30.0,
        source_root_joint="root",
        target_root_link="root",
        target_link_names=("root",),
        target_offsets=torch.zeros((1, 3)),
        primary_effector_count=1,
        positions=torch.zeros((6, 1, 3)),
        rotations_wxyz=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 6),
        position_weights=torch.ones(1),
        rotation_weights=torch.zeros(1),
    )
    profile = _profile(("joint",))

    result = solve_trajectory_ik_windowed(
        object(),
        profile,
        targets,
        initial_state=PyrokiTrajectoryState(np.zeros((6, 1), dtype=np.float32)),
        config=PyrokiTrajectoryIKConfig(
            whole_trajectory_max_frames=1,
            window_size=4,
            window_overlap=2,
        ),
    )

    np.testing.assert_allclose(
        result.joint_configurations[:, 0], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    )
    continuity = calls[1]["continuity_reference"]
    np.testing.assert_allclose(continuity.joint_configurations, [[1.0], [1.0]])
