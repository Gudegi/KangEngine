"""Display OBJ vertices with a height-based color gradient.

Run:
    python python/examples/asset/bunny_point_cloud.py
"""

from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.utils._example_args import render_example_parser


OBJ_PATH = Path(__file__).resolve().parents[3] / "assets/external/bunny.obj"
POINT_SIZE = 3.0
COLOR_STOPS = np.array(
    [
        [0.03, 0.12, 1.0],
        [0.0, 0.9, 1.0],
        [1.0, 0.9, 0.0],
        [1.0, 0.05, 0.02],
    ],
    dtype=np.float32,
)


class BunnyPointCloud(ke.App):
    def setup(self):
        mesh: ke.scene.MeshData = ke.asset.load_obj(path=str(OBJ_PATH))
        points = np.asarray(mesh.vertices, dtype=np.float32)
        lower, upper = points.min(axis=0), points.max(axis=0)

        # Map Y height to linear RGB; point colors require an alpha channel.
        height = (points[:, 1] - lower[1]) / (upper[1] - lower[1])
        stops = np.linspace(0.0, 1.0, len(COLOR_STOPS))
        colors = np.ones((len(points), 4), dtype=np.float32)
        for channel in range(3):
            colors[:, channel] = np.interp(height, stops, COLOR_STOPS[:, channel])
        self.debug.overlay.points(
            path="/bunny",
            points=points,
            colors=colors,
            size=POINT_SIZE,
        )

        center = (lower + upper) * 0.5
        radius = float(np.linalg.norm(upper - lower)) * 0.5
        camera = self.get_camera()
        camera.set_near_plane(distance=radius * 0.01)
        camera.set_far_plane(distance=radius * 10.0)
        self.set_camera_view(
            position=center + np.array([0.4, 0.3, 3.0]) * radius,
            target=center,
        )
        self.set_camera_move_speed(radius)


def main():
    args = render_example_parser(description=__doc__).parse_args()
    app = BunnyPointCloud()
    app.initialize(width=1280, height=900, up_axis=ke.UpAxis.Y, renderer=args.renderer)
    app.start()


if __name__ == "__main__":
    main()
