"""Draw world COM positions as fixed-size, depth-independent viewport markers."""

import numpy as np

from .._core import _ke


COM_COLORS = (_ke.ColorType.PASTEL_YELLOW, _ke.ColorType.PASTEL_PURPLE)


def _marker_segments():
    angles = np.arange(16) * (2 * np.pi / 16)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    starts = np.concatenate((ring, [[-0.6, 0], [0, -0.6]]))
    ends = np.concatenate((np.roll(ring, -1, axis=0), [[0.6, 0], [0, 0.6]]))
    return np.stack((starts, ends), axis=1)


_MARKER_SEGMENTS = _marker_segments()


def marker_lines(
    points, radius, view_projection, inverse_view_projection, width, height
):
    """Unproject pixel offsets around each visible center into world-space lines."""
    clip = np.column_stack((points, np.ones(len(points)))) @ view_projection.T
    valid = (clip[:, 3] > 0) & (np.abs(clip[:, 2]) <= clip[:, 3])
    clip = clip[valid]
    if not len(clip):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)

    offsets = _MARKER_SEGMENTS * radius * [2 / width, 2 / height]
    expanded = np.broadcast_to(clip[:, None, None, :], (len(clip), 18, 2, 4)).copy()
    expanded[..., :2] += offsets[None] * clip[:, None, None, 3:4]
    world = expanded @ inverse_view_projection.T
    world = (world[..., :3] / world[..., 3:4]).reshape(-1, 2, 3).astype(np.float32)
    return world[:, 0], world[:, 1]


class COMDisplay:
    def __init__(self, app):
        self.app = app
        self.path = "/__debug__/physics_com"
        self._signature = None

    def sync(self, source, bodies, articulations):
        positions = source.com_positions(bodies=bodies, articulations=articulations)
        width, height = self.app.get_render_width(), self.app.get_render_height()
        if width <= 0 or height <= 0:
            return
        view_projection = np.asarray(self.app.get_projection_matrix()) @ np.asarray(
            self.app.get_view_matrix()
        )
        signature = (
            bodies,
            articulations,
            width,
            height,
            view_projection.tobytes(),
            *(points.tobytes() for points in positions),
        )
        if signature == self._signature:
            return
        self._signature = signature

        overlay = self.app.debug_overlay
        inverse_view_projection = None
        for idx, (enabled, points, radius) in enumerate(
            zip(
                (bodies, articulations),
                positions,
                (6.0, 9.0),
            )
        ):
            path = f"{self.path}/{idx}"
            if not enabled or not len(points):
                overlay.remove(path + "/0_outline")
                overlay.remove(path + "/1_color")
                continue
            if inverse_view_projection is None:
                inverse_view_projection = np.linalg.inv(view_projection)
            starts, ends = marker_lines(
                points, radius, view_projection, inverse_view_projection, width, height
            )
            color = _ke.ColorLibrary.get(COM_COLORS[idx])
            # Sorted paths draw the colored stroke over its wider black outline.
            overlay.lines(
                path + "/0_outline",
                starts,
                ends,
                [[0, 0, 0, 1]],
                width=4.0,
                overlay=True,
            )
            overlay.lines(
                path + "/1_color",
                starts,
                ends,
                [[color.r, color.g, color.b, 1]],
                width=2.0,
                overlay=True,
            )

    def release(self):
        for idx in (0, 1):
            for suffix in ("0_outline", "1_color"):
                self.app.debug_overlay.remove(f"{self.path}/{idx}/{suffix}")
        self._signature = None
