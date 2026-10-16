"""Streaming contact estimates and offline detector parity."""

from types import SimpleNamespace

import numpy as np
import pytest

from kangengine.animation import FootContactDetector, FootContactModule
from kangengine.motion_module import ContactData


def test_height_and_full_speed_are_both_required():
    detector = FootContactDetector()
    positions = [[0, 0, 0.02], [0, 0, 0.2], [0, 0, 0.02], [0, 0, 0.02]]
    velocities = [[0, 0, 0], [0, 0, 0], [0.4, 0, 0], [0, 0, 0.4]]
    np.testing.assert_array_equal(
        detector.update(positions, velocities), [True, False, False, False]
    )


def test_hysteresis_release_and_reset():
    detector = FootContactDetector()
    assert detector.update([[0, 0, 0.03]], [[0.2, 0, 0]])[0]
    result = detector.update([[0, 0, 0.055]], [[0.42, 0, 0]])
    assert result[0]
    result[:] = False
    assert detector.update([[0, 0, 0.055]], [[0.42, 0, 0]])[0]
    assert not detector.update([[0, 0, 0.07]], [[0, 0, 0]])[0]
    assert not detector.update([[0, 0, 0.055]], [[0, 0, 0]])[0]
    assert detector.update([[0, 0, 0.03]], [[0, 0, 0]])[0]
    assert not detector.update([[0, 0, 0.03]], [[0.6, 0, 0]])[0]
    assert detector.update([[0, 0, 0.03]], [[0, 0, 0]])[0]
    detector.reset()
    assert not detector.update([[0, 0, 0.055]], [[0.42, 0, 0]])[0]


def test_slopes_steps_and_missing_support():
    detector = FootContactDetector()
    normals = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    ground = np.array([[0, 0, 2], [4, 0, 1], [0, 3, 0]], dtype=float)
    positions = ground + normals * np.array([0.02, 0.03, 0.2])[:, None]
    velocities = np.zeros((3, 3))
    np.testing.assert_array_equal(
        detector.update(positions, velocities, ground_points=ground, ground_normals=normals),
        [True, True, False],
    )
    ground[1] = np.nan
    np.testing.assert_array_equal(
        detector.update(
            positions,
            velocities,
            ground_points=ground,
            ground_normals=normals,
            supported=[True, False, True],
        ),
        [True, False, False],
    )


def test_y_up_plane_and_independent_detectors():
    first, second = FootContactDetector(), FootContactDetector()
    assert first.update(
        [[0, 2.02, 0]], [[0, 0, 0]],
        ground_points=(0, 2, 0), ground_normals=(0, 1, 0),
    )[0]
    assert first.update(
        [[0, 2.055, 0]], [[0, 0, 0]],
        ground_points=(0, 2, 0), ground_normals=(0, 1, 0),
    )[0]
    assert not second.update(
        [[0, 2.055, 0]], [[0, 0, 0]],
        ground_points=(0, 2, 0), ground_normals=(0, 1, 0),
    )[0]


def test_offline_results_preserved_and_streaming_matches_without_hysteresis():
    rng = np.random.default_rng(17)
    positions = rng.uniform(-0.1, 0.1, (40, 4, 3)).astype(np.float32)
    velocities = rng.uniform(-0.4, 0.4, positions.shape).astype(np.float32)
    data = ContactData(positions, velocities, list(range(4)), [str(i) for i in range(4)])
    floors = positions[:, :, 1].min(axis=0)
    expected = (positions[:, :, 1] <= floors + 0.045) & (
        np.linalg.norm(velocities, axis=-1) <= 0.35
    )
    np.testing.assert_array_equal(data.contacts(0.045, 0.35), expected)
    ground = np.zeros((4, 3))
    ground[:, 1] = floors
    detector = FootContactDetector(release_height=0.045, release_velocity=0.35)
    streamed = [
        detector.update(p, v, ground_points=ground, ground_normals=(0, 1, 0))
        for p, v in zip(positions, velocities)
    ]
    np.testing.assert_array_equal(streamed, expected)


@pytest.mark.parametrize("kwargs", [{"release_height": 0.01}, {"release_velocity": 0.1}])
def test_release_limits_cannot_be_stricter_than_entry(kwargs):
    with pytest.raises(ValueError, match="release limits"):
        FootContactDetector(**kwargs)


def test_module_baseline_dt_and_teleport_reset():
    transforms = np.eye(4)[None]
    transforms[0, 2, 3] = 0.02
    state = SimpleNamespace(compute_global_matrices=lambda: transforms.copy())
    module = FootContactModule(SimpleNamespace(index=lambda name: 0), ["foot"])

    assert not module.update(state=state, dt=0.1)[0]
    assert module.update(state=state, dt=0.1)[0]
    transforms[0, 0, 3] += 0.06
    assert not module.update(state=state, dt=0.1)[0]
    transforms[0, 0, 3] += 0.06
    assert module.update(state=state, dt=0.2)[0]

    transforms[0, 0, 3] += 10
    module.reset(state=state)
    assert not module.contacts.any()
    assert module.update(state=state, dt=0.1)[0]
    module.reset()
    assert module.positions is None
    assert not module.update(state=state, dt=0.1)[0]


def test_module_probe_rotation_releases_contact():
    transforms = np.eye(4)[None]
    transforms[0, 2, 3] = 0.02
    state = SimpleNamespace(compute_global_matrices=lambda: transforms.copy())
    module = FootContactModule(
        SimpleNamespace(index=lambda name: 0), ["ankle"], offsets=[[0.1, 0, 0]]
    )
    module.reset(state=state)
    assert module.update(state=state, dt=1 / 60)[0]

    # A fixed ankle with a rotating sole probe is moving, not planted.
    transforms[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert not module.update(state=state, dt=1 / 60)[0]
    np.testing.assert_allclose(module.positions, [[0, 0.1, 0.02]])


def test_module_repeated_joint_probes_and_terrain():
    transforms = np.eye(4)[None]
    transforms[0, 2, 3] = 1
    state = SimpleNamespace(compute_global_matrices=lambda: transforms.copy())
    module = FootContactModule(
        SimpleNamespace(index=lambda name: 0),
        ["ankle", "ankle"],
        offsets=[[-0.05, 0, -0.03], [0.12, 0, -0.03]],
    )
    module.reset(state=state)
    np.testing.assert_allclose(module.positions, [[-0.05, 0, 0.97], [0.12, 0, 0.97]])
    np.testing.assert_array_equal(
        module.update(
            state=state, dt=0.1,
            ground_points=[[0, 0, 0.95], [0, 0, 0.5]],
        ),
        [True, False],
    )
    assert not module.update(
        state=state, dt=0.1, ground_points=[0, 0, 0.95], supported=False,
    ).any()
