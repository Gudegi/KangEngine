"""Static terrain meshes shared by collision, rendering, and height queries."""

from __future__ import annotations

import numpy as np

from .._core import _ke


class TerrainMesh:
    """Immutable Z-up terrain surface. Cooking is cached per world/source mesh.

    Mesh queries optionally require ``kangengine[terrain]`` (Warp). Terrain
    construction and PhysX collision do not require Warp or a graphics context.
    """

    def __init__(self, vertices, indices, *, max_triangles_per_chunk=250_000):
        vertices = np.array(vertices, dtype=np.float32, order="C", copy=True)
        raw_indices = np.asarray(indices)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("vertices must be finite [N, 3]")
        if raw_indices.dtype.kind not in "iu" or raw_indices.size % 3:
            raise ValueError("indices must contain integer triangles")
        if (
            raw_indices.size == 0
            or raw_indices.min() < 0
            or raw_indices.max() >= len(vertices)
        ):
            raise ValueError("triangle index out of range or empty mesh")
        if max_triangles_per_chunk < 1:
            raise ValueError("max_triangles_per_chunk must be positive")
        indices = np.array(raw_indices, dtype=np.int32, order="C", copy=True).reshape(
            -1, 3
        )
        vertices.flags.writeable = indices.flags.writeable = False
        self.vertices, self.indices = vertices, indices
        self._meshes = []
        # Bound BVH cooking size. Compact each chunk so it does not duplicate
        # the full terrain's vertex allocation on the CPU and GPU.
        for first in range(0, len(indices), max_triangles_per_chunk):
            faces = indices[first : first + max_triangles_per_chunk]
            used, inverse = np.unique(faces, return_inverse=True)
            self._meshes.append(
                _ke.scene.MeshData.from_arrays(
                    vertices[used], inverse.astype(np.int32).reshape(-1, 3)
                )
            )
        self._queries = {}

    @classmethod
    def from_heightfield(
        cls, terrain, *, center=True, slope_threshold=None, diagonal="anti"
    ):
        """Build from an existing SubTerrain or TerrainGrid without changing it."""
        heights = terrain.height_meters()
        rows, cols = heights.shape
        y, x = np.mgrid[:rows, :cols]
        x, y = x.astype(np.float32), y.astype(np.float32)
        if slope_threshold is not None:
            if slope_threshold <= 0:
                raise ValueError("slope_threshold must be positive")
            threshold = slope_threshold * terrain.horizontal_scale
            dx, dy, corner = (np.zeros_like(heights) for _ in range(3))
            dx[:, :-1] += heights[:, 1:] - heights[:, :-1] > threshold
            dx[:, 1:] -= heights[:, :-1] - heights[:, 1:] > threshold
            dy[:-1] += heights[1:] - heights[:-1] > threshold
            dy[1:] -= heights[:-1] - heights[1:] > threshold
            corner[:-1, :-1] += heights[1:, 1:] - heights[:-1, :-1] > threshold
            corner[1:, 1:] -= heights[:-1, :-1] - heights[1:, 1:] > threshold
            x += dx + corner * (dx == 0)
            y += dy + corner * (dy == 0)
        if center:
            x, y = x - (cols - 1) / 2, y - (rows - 1) / 2
        vertices = np.stack(
            (x * terrain.horizontal_scale, y * terrain.horizontal_scale, heights),
            axis=-1,
        ).reshape(-1, 3)
        base = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)).ravel()
        indices = np.stack(
            (base, base + 1, base + cols, base + 1, base + cols + 1, base + cols),
            axis=-1,
        ).reshape(-1, 3)
        if diagonal == "main":
            indices = np.stack(
                (base, base + 1, base + cols + 1, base, base + cols + 1, base + cols),
                axis=-1,
            ).reshape(-1, 3)
        elif diagonal != "anti":
            raise ValueError("diagonal must be 'main' or 'anti'")
        return cls(vertices, indices)

    def create(
        self,
        world,
        *,
        position=(0.0, 0.0, 0.0),
        material=None,
        num_prims_per_leaf=4,
        weld_tolerance=0.0,
    ):
        """Create static GPU-compatible collision instances in a physics world."""
        return TerrainInstance(
            self,
            world,
            position=position,
            material=material,
            num_prims_per_leaf=num_prims_per_leaf,
            weld_tolerance=weld_tolerance,
        )

    def sample_height(self, positions, *, ray_start_height=None):
        """Return downward mesh hits at world XY on the input tensor's device.

        No hit returns -inf. A supplied ray_start_height matches a height
        scanner's finite origin; otherwise rays start above the entire mesh.
        """
        from ._raycast import TerrainRaycaster

        key = str(positions.device)
        if key not in self._queries:
            self._queries[key] = TerrainRaycaster(
                self.vertices, self.indices, positions.device
            )
        if ray_start_height is None:
            ray_start_height = float(self.vertices[:, 2].max()) + 1.0
        return self._queries[key].sample_height(positions, ray_start_height)


class TerrainInstance:
    """World-owned static actors and optional registered render prims."""

    def __init__(
        self,
        asset,
        world,
        *,
        position,
        material,
        num_prims_per_leaf=4,
        weld_tolerance=0.0,
    ):
        self.asset, self.world = asset, world
        self.position = tuple(float(v) for v in position)
        if len(self.position) != 3 or not np.isfinite(self.position).all():
            raise ValueError("position must contain three finite values")
        self._actors = []
        self._prims = []
        try:
            for mesh in asset._meshes:
                options = {
                    "position": self.position,
                    "num_prims_per_leaf": num_prims_per_leaf,
                    "weld_tolerance": weld_tolerance,
                }
                if material is not None:
                    options["material"] = material
                self._actors.append(world.add_triangle_mesh(mesh, **options))
        except Exception:
            self.remove()
            raise

    def add_visual(self, scene, path, *, material):
        """Register source meshes/materials through the normal scene API."""
        for i, mesh in enumerate(self.asset._meshes):
            chunk_path = f"{path}/chunk_{i}"
            view = scene.add_mesh(chunk_path, mesh, material)
            view.set_world_translation(_ke.Vec3(*self.position))
            self._prims.append((scene, chunk_path))

    def sample_height(self, positions, *, ray_start_height=None):
        offset = positions.new_tensor(self.position)
        local = positions - offset
        if ray_start_height is not None:
            ray_start_height = ray_start_height - self.position[2]
        return (
            self.asset.sample_height(local, ray_start_height=ray_start_height)
            + self.position[2]
        )

    def remove(self):
        """Remove instances before closing their world; cooked data stays cached."""
        self.remove_visual()
        for actor in self._actors:
            self.world.remove_static_actor(actor)
        self._actors.clear()

    def remove_visual(self):
        """Remove render prims before closing their scene/viewer."""
        for scene, path in self._prims:
            scene.remove_prim(path)
        self._prims.clear()
