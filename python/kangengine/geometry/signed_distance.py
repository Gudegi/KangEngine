"""Native BVH signed distances, independent of physics and rendering."""

from __future__ import annotations

import hashlib

import numpy as np

from .._core import _ke


class SignedDistanceMesh:
    """Query a closed, consistently oriented mesh in the supplied coordinate frame.

    Distances are positive outside and negative inside, in vertex units. Queries
    use native engine BVH triangle proximity and ray parity, not voxels.
    Vertices use the engine float32 mesh representation. Input must describe
    a non-self-intersecting solid; topology checks cannot prove that property.
    """

    def __init__(self, vertices, faces):
        mesh = _ke.scene.MeshData.from_arrays(vertices, faces)
        self._native = _ke.geometry.SignedDistanceMesh(mesh)
        self.content_hash = hashlib.sha256(
            np.asarray(mesh.vertices, dtype=np.float32).tobytes() +
            np.asarray(mesh.indices, dtype=np.int32).tobytes()).hexdigest()

    def query(self, points, *, chunk_size=2048):
        """Return signed distance [...], closest point [...,3], gradient [...,3].

        There is no bounded grid or out-of-grid free-space assumption. Gradient
        is zero at the surface, where a unique derivative need not exist.
        Ambiguous ray signs raise rather than assert free space. The immutable
        mesh snapshot and its BVH are reused across calls.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("Query points must have shape (..., 3)")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        flat = points.reshape(-1, 3)
        # Most interactive and per-frame queries fit in one batch. Keep the
        # native output buffers instead of allocating and copying a second set.
        if len(flat) <= chunk_size:
            distances, closest, gradient = self._native.query(flat)
            return (distances.reshape(points.shape[:-1]), closest.reshape(points.shape),
                    gradient.reshape(points.shape))

        distances = np.empty(len(flat))
        closest = np.empty_like(flat)
        gradient = np.empty_like(flat)
        for start in range(0, len(flat), chunk_size):
            selected = slice(start, start + chunk_size)
            distances[selected], closest[selected], gradient[selected] = self._native.query(flat[selected])
        return (distances.reshape(points.shape[:-1]), closest.reshape(points.shape),
                gradient.reshape(points.shape))
