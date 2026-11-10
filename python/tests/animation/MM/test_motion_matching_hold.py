"""Holding freezes final poses, including blends, across native adapters."""

from dataclasses import fields

import torch

import kangengine as ke
from test_motion_matching_batch import (
    assert_states_equal,
    device as device,
    playback as playback,
)
from test_motion_matching_batch_reference import filtered_batch as filtered_batch


def assert_zero_rates(states):
    native, bodies = states
    if hasattr(native, "joint_qd"):
        rates = [native.joint_qd]
    elif hasattr(native, "qvel"):
        rates = [native.qvel]
    else:
        rates = [native.joint_vel, native.root_vel, native.root_ang_vel]
    for value in (*rates, bodies.body_vel_w, bodies.body_ang_vel_w):
        assert torch.count_nonzero(value) == 0


def check_hold(batch):
    selected = torch.tensor([0, 2], device=batch.search.device)
    batch.advance(0.23)
    batch.transition(motion_ids=1, times=0.3)
    batch.advance(0.04)
    before = batch.sample_robot_state(selected)
    batch.hold(selected)
    held = batch.sample_robot_state(selected)
    # Holding changes velocities only, even halfway through a transition.
    for left, right in zip(before, held):
        for field in fields(left):
            if "vel" not in field.name and field.name != "joint_qd":
                torch.testing.assert_close(
                    getattr(left, field.name), getattr(right, field.name)
                )
    assert_zero_rates(held)
    reference = batch.sample_reference_state(selected)
    assert torch.count_nonzero(reference[2]) == 0
    assert torch.count_nonzero(reference[3]) == 0
    held_times = batch.times[selected].clone()
    held_ages = batch._ages[selected].clone()
    other_time = batch.times[1].clone()

    batch.advance(0.5)
    assert_states_equal(held, batch.sample_robot_state(selected), atol=0, rtol=0)
    torch.testing.assert_close(batch.times[selected], held_times)
    torch.testing.assert_close(batch._ages[selected], held_ages)
    assert batch.times[1] > other_time
    for left, right in zip(reference, batch.sample_reference_state(selected)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    future = batch.sample_robot_state(
        selected, future_offsets=batch.times.new_tensor([0, 0.1, 1])
    )
    for expected, actual in zip(held, future):
        for field in fields(expected):
            value = getattr(expected, field.name)
            torch.testing.assert_close(
                getattr(actual, field.name),
                value[:, None].expand_as(getattr(actual, field.name)),
                atol=2e-6,
                rtol=2e-6,
            )

    # select() may search held rows, but cannot change their clocks or poses.
    batch.select(batch.search.features[5].expand(3, -1))
    assert_states_equal(held, batch.sample_robot_state(selected), atol=0, rtol=0)
    batch.resume(selected)
    assert not batch.held.any()
    assert_states_equal(held, batch.sample_robot_state(selected), atol=3e-5, rtol=3e-5)
    batch.advance(0.1)
    assert torch.all(batch.times[selected] > held_times)
    assert torch.count_nonzero(batch.sample_robot_state(selected)[1].body_vel_w) > 0

    batch.hold(selected)
    held = batch.sample_robot_state(selected)
    batch.transition(selected, motion_ids=0, times=0.6)
    assert not batch.held.any()
    assert_states_equal(held, batch.sample_robot_state(selected), atol=3e-5, rtol=3e-5)
    batch.hold()
    batch.reset(selected)
    assert batch.held.tolist() == [False, True, False]


def test_hold_resume_native(playback):
    check_hold(playback)


def test_hold_resume_filtered_reference(filtered_batch):
    check_hold(filtered_batch)


def test_hold_without_smoothing(playback):
    batch = ke.animation.MotionMatchingBatchController(
        playback.search, playback._motion_lib, num_envs=3, transition_half_life=0
    )
    batch.advance(0.2)
    batch.hold()
    held = batch.sample_robot_state()
    batch.advance(1)
    assert_states_equal(held, batch.sample_robot_state(), atol=0, rtol=0)
    assert_zero_rates(held)
    batch.resume()
    assert torch.count_nonzero(batch.sample_robot_state()[1].body_vel_w) > 0


def test_hold_and_resume_do_not_read_tensor_values_on_host(playback, monkeypatch):
    batch = playback
    selected = torch.tensor([0, 2], device=batch.search.device)
    offsets = batch.times.new_tensor([0, 0.1])

    def host_read(*args, **kwargs):
        raise AssertionError("hold/resume attempted a host read")

    with monkeypatch.context() as patch:
        for name in ("item", "cpu", "tolist", "numpy", "nonzero", "__bool__"):
            patch.setattr(torch.Tensor, name, host_read)
        batch.hold(selected)
        batch.advance(0.2)
        batch.sample_robot_state(future_offsets=offsets)
        batch.sample_reference_state(selected)
        batch.resume(selected)
