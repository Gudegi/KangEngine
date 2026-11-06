from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from kangengine.app.application import DebugOverlay


def test_arrow_heads_preserve_tips_colors_and_direction():
    app = SimpleNamespace(log_debug_lines=Mock())
    overlay = DebugOverlay(app)
    starts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    ends = starts + [[2, 0, 0], [0, 0, 0], [0, -3, 0]]
    colors = np.eye(4, dtype=np.float32)[:3]
    assert overlay.arrows(
        "/arrows", starts, ends, colors, width=3, overlay=True,
    ) is overlay
    path, line_starts, line_ends, line_colors, width, hidden, on_top = (
        app.log_debug_lines.call_args.args
    )
    assert (path, width, hidden, on_top) == ("/arrows", 3, False, True)
    np.testing.assert_array_equal(line_starts[:2], starts[[0, 2]])
    np.testing.assert_array_equal(line_ends[:2], ends[[0, 2]])
    np.testing.assert_array_equal(line_colors, np.tile(colors[[0, 2]], (5, 1)))

    # All heads meet at their tips and extend backward along the shaft.
    tips = np.tile(ends[[0, 2]], (4, 1))
    np.testing.assert_array_equal(line_starts[2:], tips)
    vectors = ends[[0, 2]] - starts[[0, 2]]
    lengths = np.linalg.norm(vectors, axis=1)
    directions = np.tile(vectors / lengths[:, None], (4, 1))
    head_vectors = line_ends[2:] - tips
    projections = np.sum(head_vectors * directions, axis=1)
    np.testing.assert_allclose(projections, np.tile(-0.2 * lengths, 4), atol=1e-6)
    lateral = head_vectors - projections[:, None] * directions
    np.testing.assert_allclose(
        np.linalg.norm(lateral, axis=1), np.tile(0.08 * lengths, 4), atol=1e-6,
    )
    assert np.isfinite(line_ends).all()


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("colors", [None, [[1, 1, 1, 1]], [1, 1, 1, 1]])
def test_empty_or_zero_length_arrows_replace_previous_batch(count, colors):
    app = SimpleNamespace(log_debug_lines=Mock())
    overlay = DebugOverlay(app)
    overlay.arrows("/arrows", [[0, 0, 0]], [[1, 0, 0]])
    points = np.zeros((count, 3), dtype=np.float32)
    overlay.arrows("/arrows", points, points, colors)
    path, starts, ends, _, _, hidden, on_top = app.log_debug_lines.call_args.args
    assert path == "/arrows"
    assert starts.shape == ends.shape == (0, 3)
    assert not hidden and not on_top


def test_arrows_reject_endpoint_broadcasting():
    overlay = DebugOverlay(SimpleNamespace(log_debug_lines=Mock()))
    with pytest.raises(ValueError, match="matching shape"):
        overlay.arrows("/arrows", np.zeros((2, 3)), np.ones((1, 3)))


@pytest.mark.parametrize("on_top", [False, True])
def test_axes_preserve_transform_with_overlay(on_top):
    app = SimpleNamespace(log_debug_axes=Mock())
    overlay = DebugOverlay(app)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    overlay.axes("/axes", [1, 2, 3], rotation, length=2, overlay=on_top)
    path, transform, length, _, hidden, actual_overlay = app.log_debug_axes.call_args.args
    assert (path, length, hidden, actual_overlay) == ("/axes", 2, False, on_top)
    np.testing.assert_array_equal(transform[:3, :3], rotation)
    np.testing.assert_array_equal(transform[:3, 3], [1, 2, 3])
