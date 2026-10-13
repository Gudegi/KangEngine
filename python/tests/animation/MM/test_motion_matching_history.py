"""Output pose history and two-sample query velocities."""

import numpy as np
import pytest

from kangengine import animation
from kangengine.utils.math import quat_wxyz_rotate_numpy
from test_motion_matching import make_config, make_motion


def make_controller(**kwargs):
    db = animation.MotionMatchingDatabase.from_motions(
        [make_motion(animated=True), make_motion(velocity=(0, 1.5, 0))],
        config=make_config(),
    )
    return animation.MotionMatchingController(db, **kwargs)


def positions(state):
    return state.compute_global_matrices()[:, :3, 3]


def query_velocities(controller):
    db = controller.database
    horizons = len(db.config.trajectory_times)
    query = controller.make_query(
        trajectory_positions=np.zeros((horizons, 3)),
        trajectory_directions=np.tile((1, 0, 0), (horizons, 1)),
    )
    raw = query / db.weights * db.scale + db.mean
    return quat_wxyz_rotate_numpy(
        controller.reference_rotation_wxyz, raw[6:15].reshape(3, 3)
    )


@pytest.mark.parametrize("size", [2, 4, 10])
def test_history_retains_only_requested_number_of_public_outputs(size):
    controller = make_controller(history_size=size)
    outputs = [positions(controller.state)]
    for _ in range(12):
        output = controller.step(dt=0.02, desired_velocity=(1, 0, 0))
        outputs.append(positions(output))
    assert len(controller.history) == size
    for pose, expected in zip(controller.history, outputs[-size:]):
        np.testing.assert_array_equal(positions(pose), expected)
    output.set_root_translation((99, 99, 99))
    np.testing.assert_array_equal(positions(controller.history[-1]), outputs[-1])


def test_velocity_uses_only_previous_output_with_actual_dt_and_search_substeps():
    controller = make_controller()
    np.testing.assert_array_equal(query_velocities(controller), 0)
    previous = positions(controller.state)
    for dt in (0.02, 0.07, 0.31, 0.013):
        history_length = len(controller.history)
        output = controller.step(dt=dt, desired_velocity=(1, 0, 0))
        current = positions(output)
        np.testing.assert_allclose(
            query_velocities(controller), ((current - previous) / dt)[[1, 2, 0]],
            atol=2e-6,
        )
        assert len(controller.history) == history_length + 1
        previous = current
    saved_velocity = query_velocities(controller)
    count = len(controller.history)
    controller.step(dt=0, desired_velocity=(1, 0, 0))
    assert len(controller.history) == count
    np.testing.assert_array_equal(query_velocities(controller), saved_velocity)


def test_search_queries_use_partial_step_dt_without_adding_history_entries():
    controller = make_controller(search_interval=0.05)
    controller.step(dt=0.01, desired_velocity=(1, 0, 0))
    previous = positions(controller.state)
    elapsed = controller.elapsed_time
    count = len(controller.history)
    calls = []

    def query(current):
        partial_dt = current.elapsed_time - elapsed
        calls.append(partial_dt)
        assert len(current.history) == count
        np.testing.assert_allclose(
            query_velocities(current),
            ((positions(current.state) - previous) / partial_dt)[[1, 2, 0]],
            atol=2e-6,
        )
        return current.database.features[0]

    controller.step_with_query(dt=0.16, query_provider=query)
    np.testing.assert_allclose(calls, (0.04, 0.09, 0.14), atol=1e-12)
    assert len(controller.history) == count + 1


def test_transition_retains_history_and_reset_clears_it():
    controller = make_controller()
    for _ in range(12):
        controller.step(dt=0.01, desired_velocity=(1, 0, 0))
    assert len(controller.history) == 10
    saved = [positions(state) for state in controller.history]
    controller._place(1, 0.3, reset=False)
    for pose, expected in zip(controller.history, saved):
        np.testing.assert_array_equal(positions(pose), expected)
    controller.reset(motion_index=1, time=0.2)
    assert len(controller.history) == 1
    np.testing.assert_array_equal(positions(controller.history[0]), positions(controller.state))
    np.testing.assert_array_equal(query_velocities(controller), 0)


@pytest.mark.parametrize("size", [0, 1, -1])
def test_history_requires_two_poses(size):
    with pytest.raises(ValueError, match="history_size"):
        make_controller(history_size=size)
