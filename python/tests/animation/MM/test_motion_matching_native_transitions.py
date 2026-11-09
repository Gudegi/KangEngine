"""Native quaternion transitions, including local MuJoCo angular rates."""

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.adapters._motion_transition import _rotation_quat
from kangengine.utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_rotate,
    quat_wxyz_to_rotation_vector,
)
from test_motion_matching_batch import assert_states_equal


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture
def native_batches(tmp_path, device):
    path = tmp_path / "ball.xml"
    path.write_text(
        '<mujoco><worldbody><body name="root"><geom type="sphere" size="0.1"/>'
        '<body name="ball" pos="0 0 -0.4"><joint type="ball"/>'
        '<geom type="sphere" size="0.1"/>'
        '<body name="tip" pos="0 0 -0.3"><joint axis="0 1 0"/>'
        '<geom type="sphere" size="0.1"/>'
        "</body></body></body></worldbody></mujoco>"
    )
    data = ke.asset.MJCFLoader.load(str(path))
    # The MJCF loader expands balls into scalar axes. Exercise the native
    # Spherical enum (2), which currently has no named Python enum export.
    joints = data.joints
    joints[1] = [
        ke.asset.JointDesc(
            "ball",
            type=ke.asset.JointDescType(2),
            axis=(1, 0, 0),
            joint_offset=(0, 0, 0),
            lo_limit=-3.14,
            hi_limit=3.14,
        )
    ]
    data = ke.asset.ArticulationDesc(
        data.skeleton_tree, joints=joints, collision_geoms={}, inertials={}
    )
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=True
    )
    assert layout.blocks[1].type == ke.animation.ArticulationCoordinateType.SPHERICAL
    mapper = ke.animation.ArticulationMotionMapper(layout)
    motions = []
    for clip in range(2):
        t = torch.arange(121, dtype=torch.float64) / 60
        q = torch.zeros((len(t), layout.nq), dtype=t.dtype)
        qd = torch.zeros((len(t), layout.nv), dtype=t.dtype)
        q[:, :3] = torch.stack((t + clip * 4, -0.3 * t, t * 0 + 1), -1)
        qd[:, :3] = torch.tensor([1, -0.3, 0])
        for offset, velocity_offset, base, angular in (
            (3, 3, [0.3, -0.2, clip * 0.8], [0.2, -0.1, 0.3]),
            (7, 6, [-0.4, clip * 0.7, 0.2], [-0.3, 0.2, 0.1]),
        ):
            base = q.new_tensor(base)
            angular = q.new_tensor(angular) * (clip + 1)
            q[:, offset : offset + 4] = quat_wxyz_multiply(
                _rotation_quat(t[:, None] * angular),
                _rotation_quat(base).expand(len(t), -1),
            )
            qd[:, velocity_offset : velocity_offset + 3] = angular
        q[:, -1] = 0.2 * t + clip * 0.6
        qd[:, -1] = 0.2
        motions.append(
            ke.animation.ArticulationMotion.from_arrays(
                layout=layout,
                q=q.numpy().astype(np.float32),
                qd=qd.numpy().astype(np.float32),
                fps=60,
                motion_name=f"clip{clip}",
            )
        )
    database = ke.animation.MotionMatchingDatabase.from_motions(
        [mapper.to_skeleton_motion(motion) for motion in motions],
        config=ke.animation.MotionMatchingConfig(
            position_joints=("tip",),
            velocity_joints=("root", "tip"),
        ),
    )
    search = ke.animation.MotionMatchingBatchSearch(database, device=device)
    batches = []
    for adapter in (
        ke.adapters.newton.NewtonMotionAdapter(layout),
        ke.adapters.mujoco.MuJoCoMotionAdapter(layout),
    ):
        molib = ke.animation.MotionLibrary(motions, adapter=adapter, device=device)
        batches.append(
            ke.animation.MotionMatchingBatchController(search, molib, num_envs=3)
        )
    return batches


def parent_frame_state(state):
    """Compare native states in WXYZ with parent-frame angular velocities."""
    if isinstance(state, ke.adapters.newton.NewtonMotionState):
        q, qd = state.joint_q.clone(), state.joint_qd.clone()
        q[..., 3:7] = state.joint_q[..., [6, 3, 4, 5]]
        q[..., 7:11] = state.joint_q[..., [10, 7, 8, 9]]
    else:
        q, qd = state.qpos.clone(), state.qvel.clone()
        qd[..., 3:6] = quat_wxyz_rotate(q[..., 3:7], qd[..., 3:6])
        qd[..., 6:9] = quat_wxyz_rotate(q[..., 7:11], qd[..., 6:9])
    return q, qd


def test_ball_transitions_match_native_frames_and_remain_continuous(native_batches):
    newton, mujoco = native_batches
    selected = torch.tensor([0, 2], device=newton.search.device)
    position = newton.times.new_tensor([[4, -2, 1], [2, 3, 1]])
    heading = newton.times.new_tensor([[0.8, 0, 0, 0.6]]).expand(2, -1)
    for batch in native_batches:
        batch.advance(0.23)
        before = batch.sample_robot_state()
        batch.transition(
            selected,
            motion_ids=1,
            times=0.4,
            root_pos=position,
            root_quat_wxyz=heading,
        )
        assert_states_equal(before, batch.sample_robot_state(), atol=2e-5, rtol=1e-5)
        batch.advance(0.07)
        before = batch.sample_robot_state()
        batch.transition(selected, motion_ids=0, times=0.7)
        assert_states_equal(before, batch.sample_robot_state(), atol=2e-5, rtol=1e-5)
        batch.advance(0.04)
    first, second = (batch.sample_robot_state() for batch in native_batches)
    for left, right in zip(parent_frame_state(first[0]), parent_frame_state(second[0])):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=1e-5)
    assert_states_equal((first[1],), (second[1],))


def test_ball_velocities_match_rotation_derivatives(native_batches):
    for batch in native_batches:
        batch.advance(0.23)
        batch.transition(motion_ids=1, times=0.4)
        batch.advance(0.07)
        # Avoid clip boundaries and verify noncommuting root/ball rotations.
        offsets = batch.times.new_tensor([0, 0.001, 0.002])
        state, _ = batch.sample_robot_state(future_offsets=offsets)
        q, qd = parent_frame_state(state)
        for rotation, angular in (
            (slice(3, 7), slice(3, 6)),
            (slice(7, 11), slice(6, 9)),
        ):
            delta = quat_wxyz_multiply(
                q[:, 2, rotation], quat_wxyz_conjugate(q[:, 0, rotation])
            )
            measured = quat_wxyz_to_rotation_vector(delta) / 0.002
            torch.testing.assert_close(
                measured, qd[:, 1, angular], atol=2e-3, rtol=2e-3
            )
            torch.testing.assert_close(
                q[..., rotation].norm(dim=-1),
                torch.ones_like(q[..., 0]),
                atol=2e-6,
                rtol=0,
            )


def test_raw_ball_alignment_preserves_local_rates(native_batches):
    for original in native_batches:
        batch = ke.animation.MotionMatchingBatchController(
            original.search, original._motion_lib, num_envs=3, transition_half_life=0
        )
        source = batch._motion_lib.sample_arti_state(
            batch.motion_ids, batch.times.new_full((3,), 0.3), loop=False
        )
        heading = batch.times.new_tensor([[0.8, 0, 0, 0.6]]).expand(3, -1)
        batch.reset(times=0.3, root_quat_wxyz=heading)
        result, _ = batch.sample_robot_state()
        source_q, source_qd = parent_frame_state(source)
        q, qd = parent_frame_state(result)
        alignment = quat_wxyz_multiply(
            q[..., 3:7], quat_wxyz_conjugate(source_q[..., 3:7])
        )
        torch.testing.assert_close(q[..., 7:], source_q[..., 7:])
        torch.testing.assert_close(qd[..., 6:], source_qd[..., 6:])
        torch.testing.assert_close(
            qd[..., 3:6], quat_wxyz_rotate(alignment, source_qd[..., 3:6])
        )
        if isinstance(result, ke.adapters.mujoco.MuJoCoMotionState):
            torch.testing.assert_close(result.qvel[..., 3:6], source.qvel[..., 3:6])


def test_native_ball_transitions_do_not_sync(native_batches):
    if native_batches[0].search.device.type != "cuda":
        pytest.skip("CUDA-only synchronization check")
    for batch in native_batches:
        ids = torch.tensor([1, 0, 1], device=batch.search.device)
        times = batch.times.new_full((3,), 0.3)
        batch.transition(motion_ids=ids, times=times)
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            batch.transition(motion_ids=ids, times=times)
            batch.advance(1 / 60)
            batch.sample_robot_state(future_offsets=times)
            batch.reset(motion_ids=ids, times=times)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
