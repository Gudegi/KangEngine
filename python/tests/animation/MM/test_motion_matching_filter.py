"""Offline filtering, reference-frame consistency, and cached runtime playback."""

import builtins
import json

import numpy as np
import pytest

from kangengine import animation
from kangengine.animation.matching import make_walking_query
from kangengine.utils.math import (
    quat_wxyz_multiply_numpy,
    quat_wxyz_rotate_numpy,
    quat_wxyz_to_rotation_vector_numpy,
)
from test_motion_matching import make_config, make_motion


@pytest.mark.parametrize("filtered", [False, True])
def test_walking_query_preserves_output_pose_and_velocity_across_transition(filtered):
    motions = [
        make_motion(animated=True, velocity=(0.7, 0, 0)),
        make_motion(animated=True, velocity=(0, 1.5, 0)),
    ]
    db = (
        filtered_database(motions)
        if filtered
        else animation.MotionMatchingDatabase.from_motions(motions, config=make_config())
    )
    controller = animation.MotionMatchingController(db, transition_half_life=0.2)
    controller.reset(time=0.23)

    def pose_query():
        query = make_walking_query(
            controller, desired_velocity=(1, 0, 0), facing_angle=0, half_life=0.2
        )
        return (query / db.weights * db.scale + db.mean)[:15]

    before = pose_query()
    controller._place(1, 0.61, reset=False)
    np.testing.assert_allclose(pose_query(), before, atol=2e-5)
    cached = db._clips[1].pose_features[round(controller.time * motions[1].fps())]
    assert np.linalg.norm(pose_query() - cached) > 0.1

    previous_positions = controller.state.compute_global_matrices()[:, :3, 3]
    controller._until_search = 1.0
    controller.step(dt=0.07, desired_velocity=(1, 0, 0))
    query = pose_query()
    heading = controller.reference_rotation_wxyz
    positions = controller.state.compute_global_matrices()[:, :3, 3]
    origin = controller.reference_position.copy()
    origin[2] = 0
    np.testing.assert_allclose(
        query[:6].reshape(2, 3),
        quat_wxyz_rotate_numpy(heading, positions[[1, 2]] - origin, inverse=True),
        atol=2e-6,
    )
    measured = (positions - previous_positions) / 0.07
    np.testing.assert_allclose(
        query[6:].reshape(3, 3),
        quat_wxyz_rotate_numpy(heading, measured[[1, 2, 0]], inverse=True),
        atol=2e-6,
    )


def test_external_world_positions_and_velocities_encode_in_joint_order():
    tree = animation.SkeletonTree(
        names=["root", "joint", "tip"],
        parents=[-1, 0, 1],
        local_translations=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
        local_rotations_wxyz=np.tile((1, 0, 0, 0), (3, 1)),
    )
    quarter_turn = (np.sqrt(0.5), 0, 0, np.sqrt(0.5))
    state = animation.SkeletonState.from_rotation_and_root_translation(
        tree, [quarter_turn, quarter_turn, (1, 0, 0, 0)], (0, 0, 1)
    )
    encoded = animation.WalkingFeatures.encode_pose(
        state.compute_global_matrices()[:, :3, 3],
        [(1, 0, 0), (0, 0, 0), (0, -1, 2)],
        skeleton_tree=tree,
        config=make_config(
            position_joints=("tip",), velocity_joints=("joint", "tip", "root")
        ),
        reference_position=(0, 0, 0),
        reference_rotation_wxyz=(1, 0, 0, 0),
    )
    np.testing.assert_allclose(
        encoded.reshape(-1, 3),
        [(-1, 1, 1), (0, 0, 0), (0, -1, 2), (1, 0, 0)],
        atol=1e-6,
    )


def noisy_motion(fps=60):
    motion = make_motion(fps=fps, duration=4)
    t = np.arange(motion.num_frames()) / fps
    roots = motion.root_translations()
    roots[:, 1] += 0.08 * np.sin(2 * np.pi * 6 * t)
    roots[:, 2] += 0.04 * np.sin(2 * np.pi * 3 * t)
    rotations = motion.local_rotations_wxyz()
    # Cross the +/-pi wrap while retaining fast gait-related yaw oscillation.
    yaw = 2.8 + 0.3 * t + 0.12 * np.sin(2 * np.pi * 6 * t)
    rotations[:, 0, 0] = np.cos(yaw / 2)
    rotations[:, 0, 3] = np.sin(yaw / 2)
    return animation.SkeletonMotion.from_arrays(
        motion.skeleton_tree, roots, rotations, fps, "noisy"
    )


def filtered_database(motions=None):
    return animation.MotionMatchingDatabase.from_motions(
        [noisy_motion()] if motions is None else motions,
        config=make_config(),
        feature_builder=animation.WalkingFeatures(
            position_filter_window=0.5, rotation_filter_window=0.5
        ),
    )


def test_filter_removes_reference_jitter_but_preserves_source_motion():
    motion = noisy_motion()
    roots, rotations = motion.root_translations(), motion.local_rotations_wxyz()
    db = filtered_database([motion])
    positions = db.reference_positions(0)
    yaw = np.unwrap(
        2 * np.arctan2(db.reference_quats_wxyz(0)[:, 3], db.reference_quats_wxyz(0)[:, 0])
    )
    t = np.arange(len(roots)) / motion.fps()
    assert np.std(positions[30:-30, 1]) < np.std(roots[30:-30, 1]) * 0.3
    assert np.std((yaw - (2.8 + 0.3 * t))[30:-30]) < 0.03
    np.testing.assert_array_equal(positions[:, 2], 0)
    np.testing.assert_array_equal(db.motions[0].root_translations(), roots)
    np.testing.assert_array_equal(db.motions[0].local_rotations_wxyz(), rotations)
    assert not positions.flags.writeable


def test_reference_frame_query_reproduces_database_row():
    db = filtered_database()
    frame = 43
    time = frame / db.motions[0].fps()
    times = time + np.asarray(db.config.trajectory_times)
    heading = db.reference_quat_wxyz(0, time)
    origin = db.reference_position(0, time)
    positions = np.array([db.reference_position(0, t) for t in times])
    headings = np.array([db.reference_quat_wxyz(0, t) for t in times])
    query = db.make_query(
        0,
        time,
        trajectory_positions=quat_wxyz_rotate_numpy(
            heading, positions - origin, inverse=True
        ),
        trajectory_directions=quat_wxyz_rotate_numpy(
            heading, quat_wxyz_rotate_numpy(headings, (1, 0, 0)), inverse=True
        ),
    )
    # Low-variance trajectory scaling amplifies float32 quaternion roundoff.
    np.testing.assert_allclose(query, db.features[frame], atol=2e-5)


def test_placement_preserves_hips_residual_and_reference_derivatives():
    db = filtered_database()
    controller = animation.MotionMatchingController(db, transition_half_life=0)
    controller.reset(time=0.231)
    controller._player.advance(dt=0.2)
    time = controller.time
    source = db.motions[0].frame(0)
    db.motions[0].sample_into(time=time, state=source, loop=False)
    alignment = quat_wxyz_multiply_numpy(
        controller.reference_rotation_wxyz,
        db.reference_quat_wxyz(0, time) * (1, -1, -1, -1),
    )
    # Every original joint retains its position relative to the filtered origin.
    expected = controller.reference_position + quat_wxyz_rotate_numpy(
        alignment,
        source.compute_global_matrices()[:, :3, 3] - db.reference_position(0, time),
    )
    np.testing.assert_allclose(
        controller.state.compute_global_matrices()[:, :3, 3], expected, atol=2e-6
    )
    assert abs(controller.root_position[2] - controller.reference_position[2]) > 0.9
    position, rotation = (
        controller.reference_position,
        controller.reference_rotation_wxyz,
    )
    linear, angular = (
        controller.reference_linear_velocity,
        controller.reference_angular_velocity,
    )
    dt = 0.001
    controller._player.advance(dt=dt)
    np.testing.assert_allclose(
        (controller.reference_position - position) / dt, linear, atol=3e-4
    )
    delta = quat_wxyz_multiply_numpy(
        controller.reference_rotation_wxyz, rotation * (1, -1, -1, -1)
    )
    np.testing.assert_allclose(
        quat_wxyz_to_rotation_vector_numpy(delta) / dt, angular, atol=3e-4
    )


def test_transition_aligns_reference_and_smooths_original_pose_independently():
    db = filtered_database([noisy_motion(), make_motion(heading=-0.8, animated=True)])
    first = animation.MotionMatchingController(db)
    second = animation.MotionMatchingController(db)
    first.reset(time=0.237)
    position, rotation = first.reference_position, first.reference_rotation_wxyz
    pose = first.state.compute_global_matrices()
    other = second.state.compute_global_matrices()
    first._place(1, 0.743, reset=False)
    np.testing.assert_allclose(first.reference_position, position, atol=2e-6)
    assert abs(np.dot(first.reference_rotation_wxyz, rotation)) > 1 - 1e-6
    np.testing.assert_allclose(first.state.compute_global_matrices(), pose, atol=3e-6)
    np.testing.assert_array_equal(second.state.compute_global_matrices(), other)
    first.reset(time=(db.motions[0].num_frames() - 1) / db.motions[0].fps())
    np.testing.assert_array_equal(first.reference_linear_velocity, 0)
    np.testing.assert_array_equal(first.reference_angular_velocity, 0)


def test_npz_runtime_does_not_import_scipy_or_recompute_filter(tmp_path, monkeypatch):
    db = filtered_database()
    path = db.save_npz(tmp_path / "filtered.npz")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
    assert metadata["version"] == 1
    assert metadata["has_reference_track"] is True
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise AssertionError("runtime must not import SciPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    restored = animation.MotionMatchingDatabase.from_npz(path)
    np.testing.assert_array_equal(
        restored.reference_positions(0), db.reference_positions(0)
    )
    assert restored.schema == db.schema
    a, b = [animation.MotionMatchingController(d) for d in (db, restored)]
    for _ in range(40):
        for controller in (a, b):
            controller.step(1 / 60, desired_velocity=(1, 0, 0))
        np.testing.assert_array_equal(
            a.state.compute_global_matrices(), b.state.compute_global_matrices()
        )


@pytest.mark.parametrize("damage", ("missing", "nan", "shape", "reference_flag"))
def test_filtered_archive_rejects_invalid_tracks(tmp_path, damage):
    path = filtered_database().save_npz(tmp_path / "filtered.npz")
    with np.load(path, allow_pickle=False) as archive:
        payload = dict(archive)
    key = "clip_0_reference_positions"
    if damage == "missing":
        del payload[key]
    elif damage == "nan":
        payload[key][0, 0] = np.nan
    elif damage == "shape":
        payload[key] = payload[key][:-1]
    else:
        metadata = json.loads(payload["metadata"].tobytes())
        metadata["has_reference_track"] = False
        payload["metadata"] = np.frombuffer(
            json.dumps(metadata).encode(), dtype=np.uint8
        )
    np.savez(path, **payload)
    with pytest.raises(KeyError if damage == "missing" else ValueError):
        animation.MotionMatchingDatabase.from_npz(path)


def test_windows_use_seconds_and_do_not_cross_clip_boundaries():
    a, b = [filtered_database([noisy_motion(fps)]) for fps in (30, 60)]
    np.testing.assert_allclose(
        a.reference_positions(0)[15:-15, :2],
        b.reference_positions(0)[30:-30:2, :2],
        atol=0.007,
    )
    motions = [make_motion(origin=(10, 20, 1)), make_motion(origin=(-100, -200, 1))]
    db = filtered_database(motions)
    for i, motion in enumerate(motions):
        np.testing.assert_allclose(
            db.reference_positions(i)[:, :2],
            motion.root_translations()[:, :2],
            atol=1e-5,
        )
    for length in (2, 5, 6, 7):
        values = np.arange(length, dtype=float)
        np.testing.assert_allclose(
            animation.WalkingFeatures._filter(values, 60, 10, order=5),
            values,
            atol=1e-10,
        )


@pytest.mark.parametrize("window", (-1, np.inf, np.nan))
def test_invalid_filter_settings(window):
    with pytest.raises(ValueError):
        animation.WalkingFeatures(position_filter_window=window)
