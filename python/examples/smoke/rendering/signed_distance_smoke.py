"""Check sphere containment, surface overlap, query reuse, and their display."""

from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import kangengine as ke
from examples.rendering.signed_distance import (
    PRESETS, SLICE_PLANES, WALL_CENTER, WALL_HALF_EXTENTS,
    SignedDistanceApp, distance_colors, wall_mesh,
)
from kangengine.utils._example_args import render_example_parser


def main():
    args = render_example_parser(description=__doc__).parse_args()
    vertices, faces = wall_mesh()
    sdf = ke.geometry.SignedDistanceMesh(vertices, faces)
    points = np.random.default_rng(27).uniform([-1.5, 0, -1.2], [1.5, 2.8, 1.2], (500, 3))
    q = np.abs(points - WALL_CENTER) - WALL_HALF_EXTENTS
    expected = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    np.testing.assert_allclose(sdf.query(points)[0], expected, atol=2e-7)
    np.testing.assert_allclose(
        distance_colors(np.array([-1, -0.25, 0, 0.25, 1]), 0.5),
        [[1, 0, 0, 1], [1, .5, .5, 1], [1, 1, 1, 1], [.5, .5, 1, 1], [0, 0, 1, 1]],
    )

    app = SignedDistanceApp()
    try:
        app.initialize(width=1280, height=850, up_axis=ke.UpAxis.Y, renderer=args.renderer)
        app.set_frame_capture_active(True)
        app.setup()
        frames = []
        for name, expected_distance in (("Outside", 0.6), ("Surface overlap", 0.15), ("Fully inside", -0.3)):
            app.center = np.array(PRESETS[name])
            for _ in range(3):
                app.draw_frame()
            np.testing.assert_allclose(app.distance, expected_distance, atol=2e-7)
            np.testing.assert_allclose(app.closest, [0.3, 1.4, 0.4], atol=2e-7)
            np.testing.assert_allclose(app.gradient, [0, 0, 1], atol=2e-7)
            assert (app.clearance < 0) == (name != "Outside")
            frames.append(app.read_rgb_pixels().astype(float))

        queries = app.query_count
        assert app._panel_initialized, "Interactive controls must render"
        app.draw_frame()
        assert app.query_count == queries, "Unchanged frames must reuse the query"
        app.radius = 0.3
        app.draw_frame()
        assert app.query_count == queries, "Changing radius does not require another point query"
        np.testing.assert_allclose(app.clearance, -0.6, atol=2e-7)
        for frame in frames[1:]:
            assert np.count_nonzero(np.abs(frame - frames[0]).max(axis=2) > 20) > 200

        # Exercise optional layers and a zero-distance gradient (no arrow).
        app.show_slice = True
        app.show_solid = True
        app.center = np.array([0.3, 1.4, float(np.float32(0.4))])
        app.draw_frame()
        np.testing.assert_allclose(app.distance, 0, atol=1e-10)
        np.testing.assert_allclose(app.gradient, 0, atol=1e-10)
        # All slice orientations use the same world frame as the wall.
        for plane, axes in SLICE_PLANES.items():
            app.slice_plane = plane
            app.slice_offset = float(WALL_CENTER[axes[2]])
            app.draw_frame()
            np.testing.assert_allclose(app.slice_points[:, axes[2]], app.slice_offset)
            q = np.abs(app.slice_points - WALL_CENTER) - WALL_HALF_EXTENTS
            expected = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
            np.testing.assert_allclose(app.slice_distances, expected, atol=2e-7)
            np.testing.assert_allclose(
                app.slice_points - app.slice_closest,
                app.slice_distances[:, None] * app.slice_gradients, atol=2e-7,
            )

        with patch.object(app.sdf, "query", side_effect=AssertionError("Unexpected SDF requery")):
            app.color_range = 0.2
            app.show_sample = True
            app.sample_idx += 1
            app.set_camera_view(position=[3.3, 3.0, 4.9], target=[0, 1.4, 0])
            app.draw_frame()
            app.draw_frame()
            app.show_slice = False
            app.slice_offset += 0.1
            app.draw_frame()
        queries = app.slice_query_count
        app.show_slice = True
        app.draw_frame()
        assert app.slice_query_count == queries + 1
        app.slice_offset += 0.1
        app.draw_frame()
        assert app.slice_query_count == queries + 2

        app.show_slice = False
        app.show_solid = False
        app.center = np.array(PRESETS["Surface overlap"])
        app.radius = 0.25
        for _ in range(3):
            app.draw_frame()
        without_slice = app.read_rgb_pixels().astype(float)
        app.show_slice = True
        app.slice_plane = "XZ"
        app.slice_offset = 1.4
        app.color_range = 0.5
        for _ in range(3):
            app.draw_frame()
        with_slice = app.read_rgb_pixels().astype(float)
        assert np.count_nonzero(np.abs(with_slice - without_slice).max(axis=2) > 20) > 1000
        capture = Path(tempfile.mkdtemp(prefix="kang-signed-distance-")) / "preview.png"
        assert app.write_pixels_png(path=str(capture), flip_y=True)
        print(f"Preview: {capture}")
        print("PASS: analytic distances, closest points, gradients, sphere overlap, query reuse, visual changes")
        print("PASS: continuous colors, three slice planes, selected samples, display-only changes reuse queries")
    finally:
        app.close()


if __name__ == "__main__":
    main()
