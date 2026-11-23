"""Check articulation mesh merging preserves optional vertex colors and sources."""

import numpy as np

import kangengine as ke


def main():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.eye(3, dtype=np.float32)
    tree = ke.animation.SkeletonTree(["root"], [-1], [[0, 0, 0]], [[1, 0, 0, 0]])

    for first in (None, colors):
        for second in (None, colors[::-1].copy()):
            meshes = [
                ke.scene.MeshData.from_arrays(vertices, faces, colors=rgb)
                for rgb in (first, second)
            ]
            data = ke.asset.ArticulationDesc(
                tree, joints={}, collision_geoms={}, inertials={},
                visual_geoms=[ke.asset.VisualGeomDesc(mesh_data=mesh) for mesh in meshes],
            )
            asset = ke.visual.ArticulationVisualAsset.from_data(data)
            scene = ke.scene.create_backend(type=ke.scene.BackendType.NATIVE)
            asset.define_mesh_assets(scene, "/meshes", split_visual_geoms=False)
            merged = scene.get_prim_at_path("/meshes/body_0").get_mesh_data()
            np.testing.assert_allclose(merged.vertices, np.tile(vertices, (2, 1)))
            np.testing.assert_array_equal(np.asarray(merged.indices).reshape(-1), np.arange(6))
            if first is None and second is None:
                assert len(merged.colors) == 0
            else:
                expected = np.concatenate([
                    np.ones((3, 3)) if rgb is None else rgb for rgb in (first, second)
                ])
                np.testing.assert_allclose(merged.colors, expected)
            for mesh, rgb in zip(meshes, (first, second)):
                np.testing.assert_allclose(mesh.vertices, vertices)
                if rgb is None:
                    assert len(mesh.colors) == 0
                else:
                    np.testing.assert_allclose(mesh.colors, rgb)
    print("PASS: articulation vertex colors survive merging in either order without source mutation")


if __name__ == "__main__":
    main()
