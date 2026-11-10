"""Filtered reference placement shared by native batch adapters."""

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_rotate,
)
from test_motion_matching_batch import (
    assert_states_equal,
    device as device,
    make_playback,
    root_fields,
)


@pytest.fixture(params=["physx", "newton", "mujoco"])
def filtered_batch(tmp_path, device, request):
    original = make_playback(tmp_path, device, request.param)
    motions = []
    for idx, motion in enumerate(original.search.database.motions):
        roots = motion.root_translations().copy()
        rotations = motion.local_rotations_wxyz().copy()
        t = np.arange(len(roots)) / motion.fps()
        roots[:, 1] += 0.12 * np.sin(t * 25)
        roots[:, 2] += 0.04 * np.sin(t * 18)
        yaw = 0.15 * np.sin(t * 25)
        sway = torch.zeros((len(t), 4), dtype=torch.float64)
        sway[:, 0] = torch.from_numpy(np.cos(yaw / 2))
        sway[:, 3] = torch.from_numpy(np.sin(yaw / 2))
        rotations[:, 0] = quat_wxyz_multiply(
            sway, torch.from_numpy(rotations[:, 0])
        ).numpy()
        motions.append(
            ke.animation.SkeletonMotion.from_arrays(
                motion.skeleton_tree, roots, rotations, motion.fps(), f"sway{idx}"
            )
        )
    database = ke.animation.MotionMatchingDatabase.from_motions(
        motions,
        config=original.search.database.config,
        feature_builder=ke.animation.WalkingFeatures(
            position_filter_window=0.6, rotation_filter_window=0.6
        ),
    )
    molib = ke.animation.MotionLibrary(
        motions, adapter=original._motion_lib.adapter, device=device
    )
    search = ke.animation.MotionMatchingBatchSearch(database, device=device)
    return ke.animation.MotionMatchingBatchController(search, molib, num_envs=3)


def assert_reference_matches_scalar(batch, scalar):
    reference = batch.sample_reference_state()
    expected = (
        scalar.reference_position,
        scalar.reference_rotation_wxyz,
        scalar.reference_linear_velocity,
        scalar.reference_angular_velocity,
    )
    for actual, value in zip(reference, expected):
        torch.testing.assert_close(
            actual, actual.new_tensor(value).expand_as(actual), atol=3e-5, rtol=3e-5
        )


@pytest.mark.parametrize("half_life", [0.0, 0.1])
def test_reference_matches_scalar_and_preserves_original_body_motion(
    filtered_batch, half_life
):
    batch = ke.animation.MotionMatchingBatchController(
        filtered_batch.search,
        filtered_batch._motion_lib,
        num_envs=3,
        transition_half_life=half_life,
    )
    scalar = ke.animation.MotionMatchingController(
        batch.search.database, transition_half_life=half_life
    )
    batch.reset(times=0.237)
    scalar.reset(time=0.237)
    for motion_id, time in ((1, 0.743), (0, 0.421)):
        batch.advance(0.121)
        scalar._player.advance(dt=0.121)
        assert_reference_matches_scalar(batch, scalar)
        before = batch.sample_robot_state()
        reference_before = batch.sample_reference_state()
        batch.transition(motion_ids=motion_id, times=time)
        scalar._place(motion_id, time, reset=False)
        assert_reference_matches_scalar(batch, scalar)
        if half_life > 0:
            assert_states_equal(
                before, batch.sample_robot_state(), atol=3e-5, rtol=3e-5
            )
        for before_value, after in zip(
            reference_before[:2], batch.sample_reference_state()[:2]
        ):
            torch.testing.assert_close(before_value, after, atol=3e-5, rtol=3e-5)

    batch.reset(times=0.237)
    batch.advance(0.121)
    _, source = batch._motion_lib.sample_robot_state(batch.motion_ids, batch.times)
    _, placed = batch.sample_robot_state()
    pos, quat, _, _ = batch.sample_reference_state()
    time = float(batch.times[0])
    source_pos = pos.new_tensor(batch.search.database.reference_position(0, time))
    source_quat = quat.new_tensor(batch.search.database.reference_quat_wxyz(0, time))
    rotation = quat_wxyz_multiply(quat, quat_wxyz_conjugate(source_quat))
    expected = pos[:, None] + quat_wxyz_rotate(
        rotation[:, None], source.body_pos_w - source_pos
    )
    torch.testing.assert_close(placed.body_pos_w, expected, atol=3e-5, rtol=3e-5)
    # The reference is on the ground; the actual root retains height and sway.
    assert torch.all(placed.body_pos_w[:, 0, 2] - pos[:, 2] > 0.9)
    assert torch.all((placed.body_pos_w[:, 0, 1] - pos[:, 1]).abs() > 0.01)


def test_reference_selection_reset_and_clip_end(filtered_batch):
    batch = filtered_batch
    selected = torch.tensor([0, 2], device=batch.search.device)
    before = batch.sample_reference_state()
    position = batch.times.new_tensor([[3, -2, 0.2], [-4, 6, 0.1]])
    rotation = batch.times.new_tensor([[0.8, 0, 0, 0.6]]).expand(2, -1)
    batch.reset(
        selected, motion_ids=1, times=0.377, root_pos=position, root_quat_wxyz=rotation
    )
    placed = batch.sample_reference_state(selected)
    torch.testing.assert_close(placed[0], position, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(placed[1], rotation, atol=3e-6, rtol=1e-5)
    for old, new in zip(before, batch.sample_reference_state()):
        torch.testing.assert_close(old[1], new[1])

    initial = batch.sample_robot_state()
    # Masked-out environments must not be realigned by select().
    batch.select(
        batch.search.features[3].expand(3, -1),
        allowed_motion_mask=torch.zeros(
            2, dtype=torch.bool, device=batch.search.device
        ),
    )
    assert_states_equal(initial, batch.sample_robot_state(), atol=3e-5, rtol=3e-5)
    batch.advance(10)
    _, _, velocity, angular = batch.sample_reference_state()
    torch.testing.assert_close(velocity, torch.zeros_like(velocity))
    torch.testing.assert_close(angular, torch.zeros_like(angular))


def test_reference_velocities_are_segment_derivatives(filtered_batch):
    batch = filtered_batch
    batch.reset(times=0.237)
    pos, _, vel, _ = batch.sample_reference_state()
    batch.advance(0.001)
    following = batch.sample_reference_state()[0]
    torch.testing.assert_close((following - pos) / 0.001, vel, atol=1e-4, rtol=1e-4)
    # The future body target uses the same placement as subsequent playback.
    _, future = batch.sample_robot_state(future_offsets=batch.times.new_tensor([0.1]))
    batch.advance(0.1)
    _, current = batch.sample_robot_state()
    torch.testing.assert_close(future.body_pos_w[:, 0], current.body_pos_w)


def test_reference_frame_reproduces_database_pose_features(filtered_batch):
    batch = filtered_batch
    database = batch.search.database
    frame = 12
    batch.reset(times=frame / database.motions[0].fps())
    _, bodies = batch.sample_robot_state()
    origin, rotation, _, _ = batch.sample_reference_state()
    origin[:, 2] = 0
    inverse = quat_wxyz_conjugate(rotation)[:, None]
    positions = quat_wxyz_rotate(inverse, bodies.body_pos_w[:, 1:] - origin[:, None])
    velocities = quat_wxyz_rotate(inverse, bodies.body_vel_w)
    raw = torch.cat((positions.flatten(1), velocities.flatten(1)), dim=-1)
    expected = database._clip(0).pose_features[frame]
    torch.testing.assert_close(
        raw, raw.new_tensor(expected).expand_as(raw), atol=3e-5, rtol=3e-5
    )


def test_unfiltered_reference_uses_smoothed_native_root(filtered_batch):
    original = filtered_batch
    database = ke.animation.MotionMatchingDatabase.from_motions(
        original.search.database.motions, config=original.search.database.config
    )
    search = ke.animation.MotionMatchingBatchSearch(
        database, device=original.search.device
    )
    batch = ke.animation.MotionMatchingBatchController(
        search, original._motion_lib, num_envs=3
    )
    batch.advance(0.237)
    batch.transition(motion_ids=1, times=0.4)
    batch.advance(0.03)
    native, _ = batch.sample_robot_state()
    pos, _, vel = root_fields(native)
    reference = batch.sample_reference_state()
    torch.testing.assert_close(reference[0], pos)
    torch.testing.assert_close(reference[2], vel)
    if isinstance(native, ke.adapters.physx.PhysXMotionState):
        angular = native.root_ang_vel
    elif isinstance(native, ke.adapters.newton.NewtonMotionState):
        angular = native.joint_qd[:, 3:6]
    else:
        angular = quat_wxyz_rotate(native.qpos[:, 3:7], native.qvel[:, 3:6])
    torch.testing.assert_close(reference[3], angular)


def test_filtered_runtime_does_not_sync_or_refilter(filtered_batch, monkeypatch):
    batch = filtered_batch

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime must use the uploaded reference track")

    monkeypatch.setattr(type(batch.search.database), "reference_positions", forbidden)
    monkeypatch.setattr(ke.animation.WalkingFeatures, "_filter", forbidden)
    query = batch.search.features[5].expand(3, -1)
    batch.select(query)
    previous = (
        torch.cuda.get_sync_debug_mode() if batch.search.device.type == "cuda" else None
    )
    try:
        if previous is not None:
            torch.cuda.set_sync_debug_mode("error")
        batch.select(query)
        batch.advance(1 / 60)
        batch.sample_reference_state()
        batch.sample_robot_state()
        batch.reset(times=0.2)
    finally:
        if previous is not None:
            torch.cuda.set_sync_debug_mode(previous)
