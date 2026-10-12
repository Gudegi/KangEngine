"""Bind-space conversion shared by FBX characters and deformable surfaces."""

import numpy as np
import pytest

from kangengine import animation, asset, visual


def _write_skin(path, *, inconsistent=False):
    def values(matrix):
        return ",".join(str(v) for v in matrix.T.ravel())

    mesh_bind = np.diag([2.0, 3.0, 4.0, 1.0])
    mesh_bind[:3, 3] = (5, 6, 7)
    clusters = []
    for slot in range(2):
        bone_bind = np.eye(4)
        bone_bind[1, 3] = slot + 1
        relative = np.linalg.inv(bone_bind) @ mesh_bind
        if inconsistent and slot == 1:
            relative[0, 3] += 1
        clusters.append(f"""
        Deformer: {20 + slot}, "SubDeformer::Bone{slot}", "Cluster" {{
            Indexes: *3 {{ a: 0,1,2 }}
            Weights: *3 {{ a: 0.5,0.5,0.5 }}
            Transform: *16 {{ a: {values(relative)} }}
            TransformLink: *16 {{ a: {values(bone_bind)} }}
        }}""")
    text = (
        """; FBX 7.4.0 project file
FBXHeaderExtension: { FBXHeaderVersion: 1003
 FBXVersion: 7400
}
Objects: {
 Geometry: 1, "Geometry::Triangle", "Mesh" {
  Vertices: *9 { a: 0,0,0,1,0,0,0,1,0 }
  PolygonVertexIndex: *3 { a: 0,1,-3 }
  LayerElementNormal: 0 {
   Version: 101
   Name: ""
   MappingInformationType: "ByPolygonVertex"
   ReferenceInformationType: "Direct"
   Normals: *9 { a: 1,1,0,1,1,0,1,1,0 }
  }
 }
 Model: 2, "Model::Triangle", "Mesh" { Version: 232 }
 Model: 3, "Model::Bone0", "LimbNode" {
  Version: 232
  Properties70: {
   P: "Lcl Translation", "Lcl Translation", "", "A",0,1,0
  }
 }
 Model: 4, "Model::Bone1", "LimbNode" {
  Version: 232
  Properties70: {
   P: "Lcl Translation", "Lcl Translation", "", "A",0,1,0
  }
 }
 Deformer: 10, "Deformer::Skin", "Skin" { Version: 101 }
 AnimationStack: 30, "AnimStack::Static", "" {
  Properties70: {
   P: "LocalStart", "KTime", "Time", "",0
   P: "LocalStop", "KTime", "Time", "",46186158000
  }
 }
 AnimationLayer: 31, "AnimLayer::Base", "" { }
"""
        + "\n".join(clusters)
        + """
}
Connections: {
 C: "OO",1,2
 C: "OO",2,0
 C: "OO",3,0
 C: "OO",4,3
 C: "OO",10,1
 C: "OO",20,10
 C: "OO",21,10
 C: "OO",3,20
 C: "OO",4,21
 C: "OO",31,30
}
"""
    )
    # OpenFBX's ASCII reader expects each element body on separate lines.
    path.write_text(text.replace("{", "{\n").replace("}", "\n}\n"))


def test_bind_baked_once_across_loader_paths_and_surface(tmp_path):
    path = tmp_path / "skin.fbx"
    _write_skin(path)
    mesh = asset.FBXLoader.load_skinned_meshes(str(path), scale=1)[0]
    expected = np.array(((5, 6, 7), (7, 6, 7), (5, 9, 7)))
    np.testing.assert_allclose(mesh.vertices, expected, atol=1e-6)
    normal = np.array((0.5, 1 / 3, 0))
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(mesh.normals, np.tile(normal, (3, 1)), atol=1e-6)
    tree = asset.FBXLoader.load_skeleton(str(path), scale=1)
    surface = visual.SkinnedSurfaceAsset.from_fbx(tree, mesh)
    np.testing.assert_allclose(surface.positions, expected, atol=1e-6)
    single = asset.FBXLoader.load_character(str(path), scale=1, fps=30)
    bind_path = tmp_path / "separate_bind.fbx"
    bind_path.write_bytes(path.read_bytes())
    split = asset.FBXLoader.load_character_with_bind(
        str(path), str(bind_path), scale=1, fps=30
    )
    for character in (single, split):
        np.testing.assert_allclose(
            character.skinned_meshes[0].vertices, expected, atol=1e-6
        )


def test_external_skeleton_maps_names_and_rejects_missing_bones(tmp_path):
    path = tmp_path / "skin.fbx"
    _write_skin(path)
    tree = animation.SkeletonTree(
        names=["Model::Bone1", "Model::Bone0"],
        parents=[1, -1],
        local_translations=[[0, 1, 0], [0, 1, 0]],
        local_rotations_wxyz=[[1, 0, 0, 0]] * 2,
    )
    mesh = asset.FBXLoader.load_skinned_meshes(str(path), scale=1, skeleton_tree=tree)[
        0
    ]
    assert mesh.bone_node_indices == [
        tree.node_names().index(name) for name in mesh.bone_names
    ]
    missing = animation.SkeletonTree(
        names=["Missing"],
        parents=[-1],
        local_translations=[[0, 0, 0]],
        local_rotations_wxyz=[[1, 0, 0, 0]],
    )
    with pytest.raises(RuntimeError, match="missing from target skeleton"):
        asset.FBXLoader.load_skinned_meshes(str(path), scale=1, skeleton_tree=missing)


def test_inconsistent_cluster_bind_transforms_are_rejected(tmp_path):
    path = tmp_path / "bad_skin.fbx"
    _write_skin(path, inconsistent=True)
    with pytest.raises(RuntimeError, match="inconsistent bind transforms"):
        asset.FBXLoader.load_skinned_meshes(str(path), scale=1)
