"""Check PLY mesh, point, and texture rendering."""

from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from asset.view_ply import PlyViewer


def main():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for has_faces in (True, False):
            path = directory / "sample.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 3\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                + ("element face 1\nproperty list uchar int vertex_indices\n" if has_faces else "")
                + "end_header\n-1 -1 0 255 0 0\n1 -1 0 0 255 0\n0 1 0 0 0 255\n"
                + ("3 0 1 2\n" if has_faces else ""),
                encoding="ascii",
            )
            app = PlyViewer(str(path))
            app.initialize(width=256, height=256, headless=True, hide_ui=True)
            try:
                app.setup()
                app.draw_frame()
                image_path = directory / "frame.png"
                assert app.write_pixels_png(path=str(image_path), flip_y=True)
                rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
                for channel in range(3):
                    other = np.max(np.delete(rgb, channel, axis=2), axis=2)
                    assert np.count_nonzero(rgb[:, :, channel] - other > 50) > 5, (
                        has_faces, channel,
                    )
                if has_faces:
                    prim = app.scene.get_prim_at_path("/ply")
                    assert prim.get_mesh_component().resource_handle != 0
                else:
                    assert not app.mesh.indices
            finally:
                app.close()

        # Check UV orientation with a three-color texture.
        path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_indices\n"
            "element multi_texture_vertex 3\nproperty uchar tx\nproperty float u\nproperty float v\n"
            "element multi_texture_face 1\nproperty uchar tx\nproperty uint tn\n"
            "property list uchar int texture_vertex_indices\nend_header\n"
            "-1 -1 0\n1 -1 0\n0 1 0\n3 0 1 2\n"
            "0 0.05 0.95\n0 0.95 0.95\n0 0.5 0.05\n0 0 3 0 1 2\n",
            encoding="ascii",
        )
        texture_path = directory / "texture.png"
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:32, :, 2] = 255
        pixels[32:, :32, 0] = 255
        pixels[32:, 32:, 1] = 255
        Image.fromarray(pixels).save(texture_path)
        for up_axis in (ke.UpAxis.Y, ke.UpAxis.Z):
            app = PlyViewer(str(path), texture_path=str(texture_path))
            app.initialize(width=256, height=256, headless=True, hide_ui=True, up_axis=up_axis)
            try:
                app.setup()
                np.testing.assert_array_equal(app.mesh.vertices, [[-1, -1, 0], [1, -1, 0], [0, 1, 0]])
                assert len(app.textures) == 1
                assert app.scene.get_prim_at_path("/ply").get_mesh_component().resource_handle != 0
                app.draw_frame()
                assert app.write_pixels_png(path=str(image_path), flip_y=True)
                rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
                centers = []
                for channel in range(3):
                    other = np.max(np.delete(rgb, channel, axis=2), axis=2)
                    locations = np.argwhere(rgb[:, :, channel] - other > 80)
                    assert len(locations) > 20, (up_axis, channel)
                    centers.append(locations.mean(axis=0))
                if up_axis == ke.UpAxis.Y:
                    assert centers[0][1] < centers[1][1], "texture U direction was reversed"
                    assert centers[2][0] < centers[0][0], "texture V direction was reversed"
            finally:
                app.close()
    print("PASS: PLY RGB meshes, points, separate UV indices, and Y/Z-up views")


if __name__ == "__main__":
    main()
