"""Check partial appearance updates without mutating producer buffers."""

import numpy as np
import pytest

from kangengine.adapters.newton.buffers import instance_rgba


def test_partial_updates_preserve_omitted_channels_and_inputs():
    initial = instance_rgba((0.2, 0.4, 0.6), [0.25, 0.75], 2)
    changed = instance_rgba(None, [1.0, 0.0], 2, previous=initial)
    np.testing.assert_allclose(changed[:, :3], initial[:, :3])
    np.testing.assert_allclose(initial[:, 3], [0.25, 0.75])
    recolored = instance_rgba((1.0, 0.0, 0.0), None, 2, previous=changed)
    np.testing.assert_allclose(recolored, [[1, 0, 0, 1], [1, 0, 0, 0]])


def test_mesh_defaults_broadcast_and_explicit_opacity_overrides():
    defaults = instance_rgba((0.2, 0.4, 0.6), 0.3, 1)
    np.testing.assert_allclose(
        instance_rgba(None, [0.5, 1], 2, previous=defaults),
        [[0.2, 0.4, 0.6, 0.5], [0.2, 0.4, 0.6, 1]],
    )
    with pytest.raises(ValueError, match="opacities has 3 entries"):
        instance_rgba(None, [0.1, 0.2, 0.3], 2)


def test_cuda_opacity_requires_explicit_readback():
    class DeviceArray:
        device = "cuda:0"

        def numpy(self):
            raise AssertionError("unexpected CUDA download")

    with pytest.raises(RuntimeError, match="implicit readback"):
        instance_rgba(None, DeviceArray(), 1)


def test_equal_updates_reuse_rgba_but_in_place_changes_are_detected():
    colors = np.array([[0.2, 0.4, 0.6]] * 3, dtype=np.float32)
    alpha = np.array([0.25, 0.5, 1.0], dtype=np.float32)
    previous = instance_rgba(colors, alpha, 3)
    assert instance_rgba(colors, alpha, 3, previous=previous) is previous
    assert instance_rgba(colors.copy(), alpha.copy(), 3, previous=previous) is previous
    assert instance_rgba(None, None, 3, previous=previous) is previous

    colors[1, 0] = 0.9
    changed = instance_rgba(colors, alpha, 3, previous=previous)
    assert changed is not previous
    np.testing.assert_allclose(previous[1], [0.2, 0.4, 0.6, 0.5])
    np.testing.assert_allclose(changed[1], [0.9, 0.4, 0.6, 0.5])
    alpha[1] = 0.75
    updated = instance_rgba(None, alpha, 3, previous=changed)
    np.testing.assert_allclose(updated[1], [0.9, 0.4, 0.6, 0.75])


def test_broadcast_and_clipped_updates_reuse_effectively_equal_rgba():
    previous = instance_rgba((1.0, 0.0, 0.0), 1.0, 3)
    assert instance_rgba((1.0, 0.0, 0.0), 1.0, 3, previous=previous) is previous
    assert instance_rgba(None, 2.0, 3, previous=previous) is previous
