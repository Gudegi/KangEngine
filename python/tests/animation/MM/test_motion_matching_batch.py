"""Batch matching, placement, and continuous target transitions without a world."""

from dataclasses import fields
from importlib import import_module

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.adapters._motion_transition import (
    _left_jacobian,
    _rotation_quat,
)
from kangengine.utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_to_rotation_vector,
)
from test_motion_matching import make_config, make_motion


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("distance_mode", ["direct", "matrix"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_search_matches_scalar_and_filters(device, distance_mode, dtype):
    database = ke.animation.MotionMatchingDatabase.from_motions(
        [make_motion(animated=True), make_motion(animated=True, fps=60)],
        config=make_config(),
    )
    search = ke.animation.MotionMatchingBatchSearch(
        database,
        device=device,
        dtype=dtype,
        distance_mode=distance_mode,
        query_chunk_size=2,
        frame_chunk_size=7,
    )
    queries = database.features[[2, 7, 15, 36, 50]].copy()
    queries += np.random.default_rng(42).normal(0, 0.01, queries.shape)
    ids = torch.tensor([0, 0, 1, 1, 0], device=device)
    times = torch.tensor([0.1, 0.3, 0.7, 0.2, 1.7], device=device, dtype=torch.float64)
    allowed = torch.tensor(
        [[1, 1], [0, 1], [1, 0], [1, 1], [0, 0]], device=device, dtype=torch.bool
    )
    result = search.search(
        torch.tensor(queries, device=device, dtype=dtype),
        current_motion_ids=ids,
        current_times=times,
        continuation_bias=0.3,
        allowed_motion_mask=allowed,
    )
    for i in range(4):
        reference = database.search(
            queries[i],
            current_motion_index=int(ids[i]),
            current_time=float(times[i]),
            continuation_bias=0.3,
            allowed_motion_indices=torch.where(allowed[i])[0].cpu().tolist(),
        )
        assert result.row_ids[i].item() == reference.row
        assert result.motion_ids[i].item() == reference.motion_index
        assert result.frame_ids[i].item() == reference.frame_index
        assert result.times[i].item() == pytest.approx(reference.time)
        tolerance = 1e-4 if dtype == torch.float32 else 1e-10
        assert result.costs[i].item() == pytest.approx(reference.cost, abs=tolerance)
    assert result.row_ids[-1] == -1
    assert result.motion_ids[-1] == -1
    assert result.frame_ids[-1] == -1
    assert result.times[-1].isnan()
    assert result.costs[-1].isinf()


def test_ties_normalization_and_empty_batch(device):
    motion = make_motion(animated=True)
    database = ke.animation.MotionMatchingDatabase.from_motions(
        [motion, motion], config=make_config()
    )
    search = ke.animation.MotionMatchingBatchSearch(
        database, device=device, frame_chunk_size=3, distance_mode="direct"
    )
    query = search.features[7:8]
    result = search.search(query)
    assert result.row_ids.item() == 7
    raw = query / search.weights * search.scale + search.mean
    torch.testing.assert_close(search.normalize(raw), query)
    assert search.search(query[:0]).motion_ids.shape == (0,)
    allowed = torch.tensor([False, True], device=device)
    assert search.search(query, allowed_motion_mask=allowed).motion_ids.item() == 1


@pytest.fixture(params=["physx", "newton", "mujoco"])
def playback(tmp_path, device, request):
    return make_playback(tmp_path, device, request.param)


def make_playback(tmp_path, device, engine="physx"):
    path = tmp_path / "robot.xml"
    path.write_text(
        '<mujoco><worldbody><body name="root"><geom type="sphere" size="0.1"/>'
        '<body name="limb" pos="0 0 -0.4"><joint name="hinge" axis="0 1 0"/>'
        '<geom type="capsule" size="0.05 0.2"/>'
        "</body></body></worldbody></mujoco>"
    )
    data = ke.asset.MJCFLoader.load(str(path))
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=True
    )
    mapper = ke.animation.ArticulationMotionMapper(layout)
    motions = []
    for clip, fps in enumerate((30, 60)):
        t = np.arange(2 * fps + 1) / fps
        q = np.zeros((len(t), layout.nq), dtype=np.float32)
        qd = np.zeros((len(t), layout.nv), dtype=np.float32)
        yaw = clip * 1.2 + t * 0.3
        q[:, :3] = np.column_stack(
            (t * (clip + 1) + clip * 20, t * clip, np.ones_like(t))
        )
        q[:, 3] = np.cos(yaw / 2)
        q[:, 6] = np.sin(yaw / 2)
        q[:, 7] = np.sin(t * 2) * 0.4 + clip * 0.8
        qd[:, 0] = clip + 1
        qd[:, 1] = clip
        qd[:, 5] = 0.3
        qd[:, 6] = np.cos(t * 2) * 0.8
        motions.append(
            ke.animation.ArticulationMotion.from_arrays(
                layout=layout, q=q, qd=qd, fps=fps, motion_name=f"clip{clip}"
            )
        )
    database = ke.animation.MotionMatchingDatabase.from_motions(
        [mapper.to_skeleton_motion(m) for m in motions],
        config=ke.animation.MotionMatchingConfig(
            position_joints=("limb",),
            velocity_joints=("root", "limb"),
        ),
    )
    molib = ke.animation.MotionLibrary(
        motions,
        adapter={
            "physx": ke.adapters.physx.PhysXMotionAdapter,
            "newton": ke.adapters.newton.NewtonMotionAdapter,
            "mujoco": ke.adapters.mujoco.MuJoCoMotionAdapter,
        }[engine](layout),
        device=device,
    )
    search = ke.animation.MotionMatchingBatchSearch(database, device=device)
    return ke.animation.MotionMatchingBatchController(search, molib, num_envs=3)


def root_fields(state):
    if isinstance(state, ke.adapters.physx.PhysXMotionState):
        return state.root_pos, state.root_quat_xyzw[..., [3, 0, 1, 2]], state.root_vel
    if isinstance(state, ke.adapters.newton.NewtonMotionState):
        return (
            state.joint_q[..., :3],
            state.joint_q[..., [6, 3, 4, 5]],
            state.joint_qd[..., :3],
        )
    return state.qpos[..., :3], state.qpos[..., 3:7], state.qvel[..., :3]


def assert_states_equal(first, second, **kwargs):
    for left, right in zip(first, second):
        for field in fields(left):
            torch.testing.assert_close(
                getattr(left, field.name), getattr(right, field.name), **kwargs
            )


def test_interrupted_transition_preserves_pose_and_velocity(playback):
    batch = playback
    batch.advance(0.23)
    before = batch.sample_robot_state()
    selected = torch.tensor([0, 2], device=batch.search.device)
    batch.transition(selected, motion_ids=1, times=0.4)
    assert_states_equal(before, batch.sample_robot_state(), atol=3e-5, rtol=1e-5)
    batch.advance(0.07)
    before = batch.sample_robot_state()
    batch.transition(selected, motion_ids=0, times=0.7)
    assert_states_equal(before, batch.sample_robot_state(), atol=3e-5, rtol=1e-5)
    assert batch.motion_ids.tolist() == [0, 0, 0]


def test_future_targets_equal_advance_and_snapshots_are_independent(playback):
    batch = playback
    batch.advance(0.2)
    batch.transition(motion_ids=1, times=0.3)
    future = batch.sample_robot_state(
        future_offsets=batch.times.new_tensor([0, 0.07, 0.2])
    )
    assert root_fields(future[0])[0].shape == (3, 3, 3)
    assert future[1].body_pos_w.shape == (3, 3, 2, 3)
    for step, column in ((0.0, 0), (0.07, 1), (0.13, 2)):
        batch.advance(step)
        current = batch.sample_robot_state()
        for values, expected in zip(current, future):
            for field in fields(values):
                torch.testing.assert_close(
                    getattr(values, field.name),
                    getattr(expected, field.name)[:, column],
                    atol=1e-5,
                    rtol=1e-5,
                )
    root_fields(current[0])[0].zero_()
    assert root_fields(batch.sample_robot_state()[0])[0].abs().sum() > 0


def test_subset_reset_placement_and_raw_velocity(playback):
    batch = playback
    batch.advance(0.25)
    before = batch.sample_robot_state()
    selected = torch.tensor([1], device=batch.search.device)
    position = batch.times.new_tensor([[5, -4, 2]])
    heading = batch.times.new_tensor([[2**-0.5, 0, 0, 2**-0.5]])
    batch.reset(
        selected, motion_ids=1, times=0.1, root_pos=position, root_quat_wxyz=heading
    )
    state, bodies = batch.sample_robot_state()
    root_pos, root_quat, root_vel = root_fields(state)
    torch.testing.assert_close(root_pos[1:2], position)
    torch.testing.assert_close(root_quat[1:2], heading)
    torch.testing.assert_close(bodies.body_pos_w[1, 0], root_pos[1])
    assert torch.linalg.vector_norm(root_vel[1]).item() == pytest.approx(
        5**0.5, abs=5e-5
    )
    for current, original in zip((state, bodies), before):
        for field in fields(current):
            torch.testing.assert_close(
                getattr(current, field.name)[[0, 2]],
                getattr(original, field.name)[[0, 2]],
            )


def test_selection_continuation_empty_filter_and_end(playback):
    batch = playback
    batch.advance(0.217)
    before = batch.sample_robot_state()
    times = batch.times.clone()
    query = batch.search.features[7].expand(3, -1)
    batch.select(query, continuation_bias=1e6)
    torch.testing.assert_close(times, batch.times)
    assert_states_equal(before, batch.sample_robot_state())
    batch.select(
        query,
        allowed_motion_mask=torch.zeros(
            2, dtype=torch.bool, device=batch.search.device
        ),
    )
    assert_states_equal(before, batch.sample_robot_state())
    batch.advance(5)
    assert batch.finished.all()
    batch.select(query)
    assert not batch.finished.any()


def test_raw_placement_matches_native_player(playback):
    batch = ke.animation.MotionMatchingBatchController(
        playback.search, playback._motion_lib, num_envs=1, transition_half_life=0
    )
    database = batch.search.database
    player = ke.animation.SkeletonMotionPlayer(
        database.motions, up_axis="Z", forward_axis="X", transition_half_life=0
    )
    batch.reset(times=0.2)
    player.reset(motion_index=0, time=0.2)
    batch.transition(motion_ids=1, times=0.4)
    player.transition(motion_index=1, time=0.4)
    batch.advance(0.1)
    player.advance(dt=0.1)
    state, bodies = batch.sample_robot_state()
    matrices = torch.tensor(
        np.asarray(player.state.compute_global_matrices()), device=batch.search.device
    )
    torch.testing.assert_close(
        root_fields(state)[0][0], matrices[0, :3, 3], atol=2e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        bodies.body_pos_w[0], matrices[:, :3, 3], atol=2e-5, rtol=1e-5
    )


def test_smoothing_converges_to_new_target(playback):
    batch = playback
    batch.advance(0.2)
    batch.transition(motion_ids=1, times=0.3)
    raw = ke.animation.MotionMatchingBatchController(
        batch.search, batch._motion_lib, num_envs=3, transition_half_life=0
    )
    raw.advance(0.2)
    raw.transition(motion_ids=1, times=0.3)
    batch.advance(1.5)
    raw.advance(1.5)
    assert_states_equal(
        batch.sample_robot_state(), raw.sample_robot_state(), atol=1e-5, rtol=1e-5
    )


def test_raw_playback_needs_no_offsets_or_transition_body_sampling(
    playback, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("raw transition must only query articulation state")

    def forbidden_offsets(*args, **kwargs):
        raise AssertionError("raw playback must not allocate smoothing offsets")

    with monkeypatch.context() as patch:
        module = import_module(type(playback._native_transition).__module__)
        patch.setattr(module, "TransitionOffsets", forbidden_offsets)
        batch = ke.animation.MotionMatchingBatchController(
            playback.search, playback._motion_lib, num_envs=3, transition_half_life=0
        )
        patch.setattr(batch._motion_lib, "sample_robot_state", forbidden)
        batch.reset(times=0.1)
        batch.transition(motion_ids=1, times=0.2)
        batch.select(batch.search.features[0].expand(3, -1))
    state, _ = batch.sample_robot_state()
    assert torch.isfinite(root_fields(state)[0]).all()


def test_reset_clears_transition_decay(playback):
    batch = playback
    selected = torch.tensor([0, 2], device=batch.search.device)
    batch.transition(motion_ids=1, times=0.4)
    batch.advance(0.05)
    batch.reset(selected, motion_ids=0, times=0.1)
    reference = ke.animation.MotionMatchingBatchController(
        batch.search, batch._motion_lib, num_envs=2
    )
    reference.reset(times=0.1)
    # Stale per-transition coefficients would reappear after the reset instant.
    batch.advance(0.1)
    reference.advance(0.1)
    assert_states_equal(
        batch.sample_robot_state(selected), reference.sample_robot_state()
    )


def test_native_playback_never_changes_library_buffers(playback):
    batch = playback
    native = batch._motion_lib.adapted_buffers()
    saved = {
        field.name: getattr(native, field.name).clone() for field in fields(native)
    }
    body_pos_w = batch._motion_lib.source_buffers().body_pos_w.clone()
    batch.advance(0.2)
    batch.transition(motion_ids=1, times=0.3)
    state, bodies = batch.sample_robot_state(
        future_offsets=batch.times.new_tensor([0, 0.1])
    )
    for field in fields(state):
        getattr(state, field.name).zero_()
    bodies.body_pos_w.zero_()
    for name, original in saved.items():
        torch.testing.assert_close(getattr(native, name), original, atol=0, rtol=0)
    torch.testing.assert_close(
        batch._motion_lib.source_buffers().body_pos_w, body_pos_w, atol=0, rtol=0
    )


def test_rotation_derivative_and_inverse(device):
    # Noncommuting rotations expose missing SO(3) Jacobian terms.
    rotation = torch.tensor(
        [[0.7, -0.3, 1.4], [1e-7, 0, 0]], dtype=torch.float64, device=device
    )
    derivative = torch.tensor(
        [[0.2, 0.9, -0.4], [0, 0.2, 0.5]], dtype=torch.float64, device=device
    )
    velocity = _left_jacobian(rotation, derivative)
    torch.testing.assert_close(
        _left_jacobian(rotation, velocity, inverse=True), derivative
    )
    eps = 1e-5
    difference = quat_wxyz_multiply(
        _rotation_quat(rotation + derivative * eps),
        quat_wxyz_conjugate(_rotation_quat(rotation - derivative * eps)),
    )
    measured = quat_wxyz_to_rotation_vector(difference) / (2 * eps)
    torch.testing.assert_close(measured, velocity, atol=1e-7, rtol=1e-6)


def test_cuda_queries_and_transitions_do_not_sync(playback):
    batch = playback
    if batch.search.device.type != "cuda":
        pytest.skip("CUDA-only synchronization check")
    query = batch.search.features[5].expand(3, -1)
    ids = torch.tensor([1, 0, 1], device=batch.search.device)
    times = batch.times.new_full((3,), 0.2)
    # Warm up kernels before enabling Torch's host-sync diagnostics.
    batch.select(query)
    batch.transition(motion_ids=ids, times=times)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        batch.select(query)
        batch.transition(motion_ids=ids, times=times)
        batch.advance(1 / 30)
        batch.sample_robot_state(future_offsets=times)
        batch.reset(motion_ids=ids, times=times)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
