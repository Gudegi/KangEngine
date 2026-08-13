"""Newton mesh conversion for KangEngine render resources."""

from __future__ import annotations

import numpy as np

from ..._core import _ke


def compute_vertex_normals(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    normals = np.zeros_like(points)
    if len(triangles) == 0:
        return normals
    edges_a = points[triangles[:, 1]] - points[triangles[:, 0]]
    edges_b = points[triangles[:, 2]] - points[triangles[:, 0]]
    face_normals = np.cross(edges_a, edges_b)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    return normals


def mesh_data_from_arrays(points, indices, normals=None, uvs=None):
    """Build a KangEngine ``MeshData`` from host arrays."""

    positions = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    index_values = np.asarray(indices, dtype=np.uint32).reshape(-1)
    normal_values = (
        compute_vertex_normals(positions, index_values)
        if normals is None
        else np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    )
    if len(normal_values) != len(positions):
        raise ValueError("mesh normals must match vertex count")

    mesh = _ke.scene.MeshData()
    mesh.vertices = [_ke.Vec3(*map(float, value)) for value in positions]
    mesh.normals = [_ke.Vec3(*map(float, value)) for value in normal_values]
    mesh.indices = [int(value) for value in index_values]
    if uvs is not None:
        uv_values = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
        if len(uv_values) != len(positions):
            raise ValueError("mesh UVs must match vertex count")
        mesh.uvs = [_ke.Vec2(*map(float, value)) for value in uv_values]
    return mesh
