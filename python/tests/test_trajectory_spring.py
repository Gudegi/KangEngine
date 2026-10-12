"""Analytic trajectory integration, physical derivatives, and horizon contracts."""

import math
from pathlib import Path

import numpy as np
import pytest

from kangengine.animation import trajectory_spring_position as position_spring
from kangengine.animation import trajectory_spring_rotation as rotation_spring


def quat(r):
    r = np.asarray(r, dtype=np.float64)
    angle = np.linalg.norm(r)
    return np.r_[np.cos(angle / 2), r * (np.sin(angle / 2) / angle if angle else 0.5)]


def multiply(a, b):
    return np.r_[
        a[0] * b[0] - a[1:] @ b[1:],
        a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]),
    ]


def relative_vector(after, before):
    q = multiply(after, before * (1, -1, -1, -1))
    if q[0] < 0:
        q = -q
    length = np.linalg.norm(q[1:])
    return (
        2 * q[1:] if length < 1e-12 else 2 * np.arctan2(length, q[0]) * q[1:] / length
    )


def test_position_matches_reference_equation():
    p, v, a, goal = np.array(((1, 2, -3), (0.4, -0.8, 1), (0.2, 0.3, -0.4), (2, 1, 0)))
    for half_life in (0.02, 0.2, 2):
        y = 2 * math.log(2) / half_life
        j0, j1 = v - goal, a + (v - goal) * y
        for dt in (0, 0.001, 0.1, 0.7):
            e = math.exp(-y * dt)
            expected = (
                p
                + e * (-j1 / y**2 + (-j0 - j1 * dt) / y)
                + j1 / y**2
                + j0 / y
                + goal * dt,
                e * (j0 + j1 * dt) + goal,
                e * (a - j1 * y * dt),
            )
            actual = position_spring(
                p, v, a, desired_velocity=goal, half_life=half_life, dt=dt
            )
            np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_position_substeps_and_derivatives():
    initial = np.array(((2, 3, 4), (0.5, -1, 2), (0.2, 0.1, -0.4)))
    kwargs = dict(desired_velocity=(2, 0, -1), half_life=0.2)
    whole = position_spring(*initial, dt=0.3, **kwargs)
    partial = initial
    for _ in range(30):
        partial = position_spring(*partial, dt=0.01, **kwargs)
    np.testing.assert_allclose(partial, whole, atol=1e-13)
    eps = 1e-5
    before = position_spring(*initial, dt=0.3 - eps, **kwargs)
    after = position_spring(*initial, dt=0.3 + eps, **kwargs)
    np.testing.assert_allclose((after[0] - before[0]) / (2 * eps), whole[1], atol=2e-9)
    np.testing.assert_allclose((after[1] - before[1]) / (2 * eps), whole[2], atol=2e-9)
    # Tiny steps avoid cancellation in integrated displacement.
    tiny = position_spring(*initial, dt=1e-10, **kwargs)
    np.testing.assert_allclose(tiny[0] - initial[0], initial[1] * 1e-10, atol=4e-16)


def test_rotation_matches_fixed_axis_reference():
    y, dt = 2 * math.log(2) / 0.2, 0.17
    e = math.exp(-y * dt)
    j0, j1 = 0.9, 0.7 + 0.9 * y
    q, w = rotation_spring(
        quat((0, 0, 0.6)),
        (0, 0, 0.7),
        desired_rotation_wxyz=quat((0, 0, -0.3)),
        half_life=0.2,
        dt=dt,
    )
    np.testing.assert_allclose(q, quat((0, 0, -0.3 + e * (j0 + j1 * dt))), atol=1e-14)
    np.testing.assert_allclose(w, (0, 0, e * (0.7 - j1 * y * dt)), atol=1e-14)


def test_arbitrary_rotation_preserves_spatial_velocity_and_substeps():
    initial = (quat((0.6, -0.3, 0.2)), np.array((0.5, 0.2, -0.7)))
    kwargs = dict(desired_rotation_wxyz=quat((-0.2, 0.3, 0.1)), half_life=0.2)
    eps = 1e-6
    at_start = rotation_spring(*initial, dt=eps, **kwargs)
    at_half = rotation_spring(*initial, dt=eps / 2, **kwargs)
    # Remove first-order acceleration error from the forward difference.
    np.testing.assert_allclose(
        (
            4 * relative_vector(at_half[0], initial[0])
            - relative_vector(at_start[0], initial[0])
        )
        / eps,
        initial[1],
        atol=2e-9,
    )
    whole = rotation_spring(*initial, dt=0.3, **kwargs)
    partial = initial
    for _ in range(30):
        partial = rotation_spring(*partial, dt=0.01, **kwargs)
    np.testing.assert_allclose(partial[0], whole[0], atol=1e-13)
    np.testing.assert_allclose(partial[1], whole[1], atol=1e-13)
    before = rotation_spring(*initial, dt=0.3 - eps, **kwargs)
    after = rotation_spring(*initial, dt=0.3 + eps, **kwargs)
    np.testing.assert_allclose(
        relative_vector(after[0], before[0]) / (2 * eps), whole[1], atol=1e-9
    )


def test_horizons_are_independent_and_do_not_mutate_inputs():
    initial = np.array(((1, 2, 3), (0.2, 0.3, 0.4), (0.5, 0.6, 0.7)))
    saved = initial.copy()
    times = np.array((0.7, 0, 0.1, 0.1))  # Unsorted and duplicate horizons are valid.
    kwargs = dict(desired_velocity=(1, -1, 0), half_life=0.2)
    batched = position_spring(*initial, dt=times, **kwargs)
    for i, time in enumerate(times):
        single = position_spring(*initial, dt=time, **kwargs)
        np.testing.assert_array_equal(np.asarray(batched)[:, i], single)
    np.testing.assert_array_equal(initial, saved)
    empty = position_spring(*initial, dt=[], **kwargs)
    assert all(value.shape == (0, 3) for value in empty)
    rotation_kwargs = dict(desired_rotation_wxyz=quat((0, 0, 0.4)), half_life=0.2)
    q, w = rotation_spring((1, 0, 0, 0), (0, 0, 0), dt=times, **rotation_kwargs)
    assert q.shape == (4, 4) and w.shape == (4, 3)
    for i, time in enumerate(times):
        single = rotation_spring(
            (1, 0, 0, 0), (0, 0, 0), dt=time, **rotation_kwargs
        )
        np.testing.assert_array_equal(q[i], single[0])
        np.testing.assert_array_equal(w[i], single[1])


def test_zero_half_life_zero_time_large_time_and_quaternion_signs():
    initial = np.array(((1, 2, 3), (0.2, 0.3, 0.4), (0.5, 0.6, 0.7)))
    for half_life in (0, 0.2):
        unchanged = position_spring(
            *initial, desired_velocity=(2, 0, 0), half_life=half_life, dt=0
        )
        np.testing.assert_array_equal(unchanged, initial)
    snapped = position_spring(
        *initial, desired_velocity=(2, 0, 0), half_life=0, dt=0.5
    )
    np.testing.assert_allclose(snapped, ((2, 2, 3), (2, 0, 0), (0, 0, 0)))
    for half_life in (0, 1e-300, 0.2):
        q, w = rotation_spring(
            quat((0.3, 0.2, 0.1)),
            (1, 2, 3),
            desired_rotation_wxyz=(1, 0, 0, 0),
            half_life=half_life,
            dt=1000,
        )
        np.testing.assert_array_equal(q, (1, 0, 0, 0))
        np.testing.assert_array_equal(w, (0, 0, 0))
    for q in ((0, 1, 0, 0), (0, -1, 0, 0)):
        result = rotation_spring(
            q,
            (0, 0, 0),
            desired_rotation_wxyz=(1, 0, 0, 0),
            half_life=0.2,
            dt=0.1,
        )
        negative_goal = rotation_spring(
            q,
            (0, 0, 0),
            desired_rotation_wxyz=(-1, 0, 0, 0),
            half_life=0.2,
            dt=0.1,
        )
        np.testing.assert_allclose(result[0], -negative_goal[0], atol=1e-14)
        np.testing.assert_allclose(result[1], negative_goal[1], atol=1e-14)
        assert result[0][1] > 0 and result[1][0] < 0


def test_invalid_inputs_are_rejected():
    p = dict(
        position=(0, 0, 0),
        velocity=(0, 0, 0),
        acceleration=(0, 0, 0),
        desired_velocity=(1, 0, 0),
        half_life=0.2,
        dt=0.1,
    )
    r = dict(
        rotation_wxyz=(1, 0, 0, 0),
        angular_velocity=(0, 0, 0),
        desired_rotation_wxyz=(1, 0, 0, 0),
        half_life=0.2,
        dt=0.1,
    )
    for function, arguments in ((position_spring, p), (rotation_spring, r)):
        for bad in (-1, np.nan, np.inf):
            with pytest.raises(ValueError):
                function(**(arguments | {"half_life": bad, "dt": []}))
        for key in ("half_life", "dt"):
            for bad in (-1, np.nan, np.inf):
                with pytest.raises(ValueError):
                    function(**(arguments | {key: bad}))
        with pytest.raises(ValueError):
            function(**(arguments | {"dt": [[0.1]]}))
    for bad in ((0, 0), (0, 0, np.nan), [[0, 0, 0]]):
        with pytest.raises(ValueError):
            position_spring(**(p | {"position": bad}))
        with pytest.raises(ValueError):
            position_spring(**(p | {"position": bad, "dt": []}))
    for bad in ((0, 0, 0, 0), (2, 0, 0, 0), (1, 0, np.inf, 0)):
        with pytest.raises(ValueError):
            rotation_spring(**(r | {"desired_rotation_wxyz": bad}))
        with pytest.raises(ValueError):
            rotation_spring(**(r | {"desired_rotation_wxyz": bad, "dt": []}))


@pytest.mark.parametrize("up_axis,forward_axis", [("Z", "X"), ("Y", "Z")])
def test_walking_prediction_coordinates_and_inputs(monkeypatch, up_axis, forward_axis):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "animation" / "MM"))
    from kangengine.animation.matching import (
        MotionMatchingController,
        MotionMatchingDatabase,
        predict_walking_trajectory,
    )
    from test_motion_matching import make_config, make_motion

    config = make_config(up_axis=up_axis, forward_axis=forward_axis)
    controller = MotionMatchingController(
        MotionMatchingDatabase.from_motions([make_motion()], config=config)
    )
    controller.reset(time=0.2)
    state = controller.state.compute_global_matrices().copy()
    up = np.eye(3)["XYZ".index(up_axis)]
    forward = np.eye(3)["XYZ".index(forward_axis)]
    facing = np.cross(up, forward)
    velocity = 1.4 * facing
    positions, directions = predict_walking_trajectory(
        controller, desired_velocity=velocity, facing_angle=math.pi / 2, half_life=0
    )
    times = np.array((0, *config.trajectory_times))
    np.testing.assert_allclose(
        positions, controller.reference_position + times[:, None] * velocity, atol=1e-7
    )
    np.testing.assert_allclose(directions[0], forward, atol=1e-7)
    np.testing.assert_allclose(
        directions[1:], np.tile(facing, (len(times) - 1, 1)), atol=2e-7
    )
    np.testing.assert_array_equal(controller.state.compute_global_matrices(), state)
    np.testing.assert_array_equal(velocity, 1.4 * facing)
    assert controller.time == pytest.approx(0.2)
    for bad in ((0, 0), (np.nan, 0, 0), up):
        with pytest.raises(ValueError, match="desired_velocity"):
            predict_walking_trajectory(
                controller, desired_velocity=bad, facing_angle=0, half_life=0.2
            )
    with pytest.raises(ValueError, match="facing_angle"):
        predict_walking_trajectory(
            controller, desired_velocity=velocity, facing_angle=np.nan, half_life=0.2
        )


def test_walking_query_zero_half_life_and_search_boundary_consistency(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "animation" / "MM"))
    from kangengine.animation.matching import make_walking_query
    from kangengine.animation import MotionMatchingController, MotionMatchingDatabase
    from test_motion_matching import make_config, make_motion

    db = MotionMatchingDatabase.from_motions(
        [make_motion(animated=True), make_motion(velocity=(2, 0, 0))],
        config=make_config(),
    )
    first, second = MotionMatchingController(db), MotionMatchingController(db)
    facing = np.array((math.cos(0.3), math.sin(0.3), 0), dtype=np.float32)
    expected = first._walking_query(1.2 * facing, facing)
    actual = make_walking_query(
        first, desired_velocity=1.2 * facing, facing_angle=0.3, half_life=0
    )
    np.testing.assert_allclose(actual, expected, atol=2e-6)

    def query(controller):
        return make_walking_query(
            controller, desired_velocity=1.2 * facing, facing_angle=0.3, half_life=0.2
        )

    first.step_with_query(0.45, query_provider=query)
    for _ in range(9):
        second.step_with_query(0.05, query_provider=query)
    assert first.motion_index == second.motion_index
    np.testing.assert_allclose(
        first.state.compute_global_matrices(),
        second.state.compute_global_matrices(),
        atol=2e-6,
    )
