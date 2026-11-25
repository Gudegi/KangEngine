"""View PLY meshes or point clouds.

Usage:
    python python/examples/asset/view_ply.py --ply-path model.ply
    python python/examples/asset/view_ply.py --ply-path scan.ply --colors-srgb
"""

from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.utils._example_args import render_example_parser


POINT_SIZE = 6.0


class PlyViewer(ke.App):
    def __init__(self, ply_path: str, *, colors_srgb: bool = False,
                 texture_path: str | None = None):
        super().__init__()
        self.mesh: ke.scene.MeshData = ke.asset.load_ply(
            path=str(Path(ply_path).expanduser()), colors_srgb=colors_srgb,
        )
        self.texture_path = Path(texture_path).expanduser() if texture_path else None
        if self.texture_path is not None and not self.mesh.indices:
            raise ValueError("Image textures require a PLY mesh with UVs, not a point cloud")

    def setup(self):
        points = np.asarray(self.mesh.vertices, dtype=np.float32)
        if not len(points):
            raise ValueError("The PLY file contains no vertices to display")

        if self.mesh.indices:
            if self.texture_path is None:
                material = self.create_color_material()
            else:
                texture: ke.render.Texture = self.load_texture(
                    path=str(self.texture_path), srgb=True,
                )
                material = self.create_phong_material(
                    diffuse=ke.Vec3(1, 1, 1), diffuse_map=texture,
                )
            material.unlit = True
            surface = self.scene.add_mesh(
                path="/ply", mesh_data=self.mesh, material=material,
                color=ke.Vec4(1, 1, 1, 1),
            )
            surface.set_double_sided(True)
        else:
            rgb = self.mesh.colors
            colors = None
            if rgb:
                colors = np.ones((len(points), 4), dtype=np.float32)
                colors[:, :3] = np.asarray(rgb)
            self.debug.overlay.points(
                path="/ply", points=points, colors=colors, size=POINT_SIZE,
            )

        # Frame the data without changing its coordinates.
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = (lower + upper) * 0.5
        radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.1)
        camera = self.get_camera()
        camera.set_near_plane(distance=radius * 0.01)
        camera.set_far_plane(distance=radius * 10)
        direction = [0, 0.6, 3] if self.up_axis == ke.UpAxis.Y else [0, -3, 0.6]
        self.set_camera_view(position=center + np.array(direction) * radius, target=center)


def main():
    parser = render_example_parser(description=__doc__)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--texture-path", help="Color image for a mesh with UV coordinates")
    parser.add_argument("--up-axis", choices=("Y", "Z"), default="Y",
                        help="Viewer up axis; source coordinates are preserved")
    parser.add_argument("--colors-srgb", action="store_true",
                        help="Decode sRGB vertex colors to linear RGB")
    args = parser.parse_args()
    app = PlyViewer(args.ply_path, colors_srgb=args.colors_srgb, texture_path=args.texture_path)
    app.initialize(
        up_axis=ke.UpAxis.Y if args.up_axis == "Y" else ke.UpAxis.Z,
        renderer=args.renderer,
    )
    app.start()


if __name__ == "__main__":
    main()
