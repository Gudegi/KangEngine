# Reference : https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/main/omniisaacgymenvs/utils/terrain_utils/terrain_utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .._core import _ke


@dataclass
class SubTerrain:
    """Discrete height-field terrain buffer.

    ``height_field_raw`` stores integer height units.  Convert to meters with
    height_meters() before building render/physics meshes.
    """

    width: int
    length: int
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005

    def __post_init__(self):
        self.width = int(self.width)
        self.length = int(self.length)
        self.horizontal_scale = float(self.horizontal_scale)
        self.vertical_scale = float(self.vertical_scale)
        if self.width < 2 or self.length < 2:
            raise ValueError("SubTerrain width/length must be at least 2")
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)

    def height_meters(self) -> np.ndarray:
        return self.height_field_raw.astype(np.float32) * self.vertical_scale

    def to_mesh(
        self, *, up_axis=_ke.UpAxis.Y, center: bool = True, backend: str = "cpp"
    ):
        return height_field_to_mesh(
            self.height_meters(),
            up_axis=up_axis,
            horizontal_scale=self.horizontal_scale,
            center=center,
            backend=backend,
        )


class TerrainGrid:
    """Large height-field assembled from multiple overlapping sub-terrains.

    Adjacent tiles share one vertex row/column, so the final mesh is continuous
    instead of being a set of separate islands.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tile_width: int,
        tile_length: int,
        *,
        horizontal_scale: float = 0.05,
        vertical_scale: float = 0.005,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.tile_width = int(tile_width)
        self.tile_length = int(tile_length)
        self.horizontal_scale = float(horizontal_scale)
        self.vertical_scale = float(vertical_scale)
        if self.rows < 1 or self.cols < 1:
            raise ValueError("TerrainGrid rows/cols must be at least 1")
        if self.tile_width < 2 or self.tile_length < 2:
            raise ValueError("TerrainGrid tile width/length must be at least 2")

        width = self.rows * (self.tile_width - 1) + 1
        length = self.cols * (self.tile_length - 1) + 1
        self.height_field_raw = np.zeros((width, length), dtype=np.int16)

    @property
    def width(self) -> int:
        return int(self.height_field_raw.shape[0])

    @property
    def length(self) -> int:
        return int(self.height_field_raw.shape[1])

    def tile_origin(self, row: int, col: int) -> tuple[int, int]:
        return row * (self.tile_width - 1), col * (self.tile_length - 1)

    def set_tile(self, row: int, col: int, tile: SubTerrain):
        """Copy a sub-terrain into the grid at tile coordinates."""

        row = int(row)
        col = int(col)
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("tile row/col out of range")
        if tile.height_field_raw.shape != (self.tile_width, self.tile_length):
            raise ValueError("tile shape must match TerrainGrid tile_width/tile_length")
        r0, c0 = self.tile_origin(row, col)
        self.height_field_raw[r0 : r0 + self.tile_width, c0 : c0 + self.tile_length] = (
            tile.height_field_raw
        )
        return self

    def fill(self, generator: Callable[[SubTerrain, int, int], SubTerrain]):
        """Generate every tile with ``generator(tile, row, col)``."""

        for row in range(self.rows):
            for col in range(self.cols):
                tile = SubTerrain(
                    self.tile_width,
                    self.tile_length,
                    horizontal_scale=self.horizontal_scale,
                    vertical_scale=self.vertical_scale,
                )
                result = generator(tile, row, col)
                self.set_tile(row, col, tile if result is None else result)
        return self

    def height_meters(self) -> np.ndarray:
        return self.height_field_raw.astype(np.float32) * self.vertical_scale

    def to_mesh(
        self, *, up_axis=_ke.UpAxis.Y, center: bool = True, backend: str = "cpp"
    ):
        return height_field_to_mesh(
            self.height_meters(),
            up_axis=up_axis,
            horizontal_scale=self.horizontal_scale,
            center=center,
            backend=backend,
        )


def _to_height_units(value: float, vertical_scale: float) -> int:
    return int(round(float(value) / float(vertical_scale)))


def random_uniform_terrain(
    terrain: SubTerrain,
    min_height: float,
    max_height: float,
    step: float = 0.02,
    rng=None,
) -> SubTerrain:
    """Add discrete uniform random heights to the terrain."""

    rng = np.random.default_rng() if rng is None else rng
    min_h = _to_height_units(min_height, terrain.vertical_scale)
    max_h = _to_height_units(max_height, terrain.vertical_scale)
    step_h = max(1, abs(_to_height_units(step, terrain.vertical_scale)))
    choices = np.arange(min_h, max_h + step_h, step_h, dtype=np.int16)
    terrain.height_field_raw += rng.choice(
        choices, size=terrain.height_field_raw.shape
    ).astype(np.int16)
    return terrain


def sloped_terrain(terrain: SubTerrain, slope: float = 1.0) -> SubTerrain:
    """Add a linear slope along the terrain width axis."""

    max_height = slope * terrain.horizontal_scale * terrain.width
    heights = np.linspace(0.0, max_height, terrain.width, dtype=np.float32)
    units = np.rint(heights / terrain.vertical_scale).astype(np.int16)
    terrain.height_field_raw += units[:, None]
    return terrain


def pyramid_sloped_terrain(
    terrain: SubTerrain, slope: float = 1.0, platform_size: float = 1.0
) -> SubTerrain:
    """Add a pyramid slope clipped to a flat center platform."""

    x = np.arange(terrain.width, dtype=np.float32)
    y = np.arange(terrain.length, dtype=np.float32)
    cx = max((terrain.width - 1) * 0.5, 1.0)
    cy = max((terrain.length - 1) * 0.5, 1.0)
    wx = 1.0 - np.abs(x - cx) / cx
    wy = 1.0 - np.abs(y - cy) / cy
    max_height = (
        slope * terrain.horizontal_scale * min(terrain.width, terrain.length) * 0.5
    )
    heights = max_height * np.outer(wx, wy)
    terrain.height_field_raw += np.rint(heights / terrain.vertical_scale).astype(
        np.int16
    )

    half = int(max(0.0, platform_size) / terrain.horizontal_scale * 0.5)
    if half > 0:
        mx = terrain.width // 2
        my = terrain.length // 2
        terrain.height_field_raw[
            max(0, mx - half) : min(terrain.width, mx + half),
            max(0, my - half) : min(terrain.length, my + half),
        ] = terrain.height_field_raw[mx, my]
    return terrain


def stairs_terrain(
    terrain: SubTerrain, step_width: float, step_height: float
) -> SubTerrain:
    """Add stairs along the terrain width axis."""

    step_width_px = max(1, int(round(step_width / terrain.horizontal_scale)))
    step_height_units = _to_height_units(step_height, terrain.vertical_scale)
    for start in range(0, terrain.width, step_width_px):
        level = start // step_width_px
        terrain.height_field_raw[start : start + step_width_px, :] += (
            level * step_height_units
        )
    return terrain


def wave_terrain(
    terrain: SubTerrain, num_waves: float = 2.0, amplitude: float = 0.2
) -> SubTerrain:
    """Add sinusoidal waves along both terrain axes."""

    x = np.linspace(0.0, np.pi * 2.0 * num_waves, terrain.width)
    y = np.linspace(0.0, np.pi * 2.0 * num_waves, terrain.length)
    heights = 0.5 * amplitude * (np.sin(x)[:, None] + np.cos(y)[None, :])
    terrain.height_field_raw += np.rint(heights / terrain.vertical_scale).astype(
        np.int16
    )
    return terrain


def discrete_obstacles_terrain(
    terrain: SubTerrain,
    max_height: float,
    min_size: float,
    max_size: float,
    num_rects: int,
    platform_size: float = 1.0,
    rng=None,
) -> SubTerrain:
    """Scatter axis-aligned rectangular height obstacles."""

    rng = np.random.default_rng() if rng is None else rng
    max_h = abs(_to_height_units(max_height, terrain.vertical_scale))
    min_px = max(1, int(round(min_size / terrain.horizontal_scale)))
    max_px = max(min_px, int(round(max_size / terrain.horizontal_scale)))
    heights = np.array([-max_h, -max_h // 2, max_h // 2, max_h], dtype=np.int16)
    for _ in range(int(num_rects)):
        w = int(rng.integers(min_px, max_px + 1))
        length = int(rng.integers(min_px, max_px + 1))
        if w >= terrain.width or length >= terrain.length:
            continue
        x0 = int(rng.integers(0, terrain.width - w))
        y0 = int(rng.integers(0, terrain.length - length))
        terrain.height_field_raw[x0 : x0 + w, y0 : y0 + length] = rng.choice(heights)

    half = int(max(0.0, platform_size) / terrain.horizontal_scale * 0.5)
    if half > 0:
        mx = terrain.width // 2
        my = terrain.length // 2
        terrain.height_field_raw[
            max(0, mx - half) : min(terrain.width, mx + half),
            max(0, my - half) : min(terrain.length, my + half),
        ] = 0
    return terrain


def height_field_to_mesh(
    heights,
    *,
    horizontal_scale: float = 1.0,
    up_axis=_ke.UpAxis.Y,
    center: bool = True,
    backend: str = "cpp",
):
    """Convert a 2D height array in meters to ``scene.MeshData``."""

    if backend == "python":
        return height_field_to_mesh_python(
            heights,
            horizontal_scale=horizontal_scale,
            up_axis=up_axis,
            center=center,
        )
    if backend != "cpp":
        raise ValueError("backend must be 'cpp' or 'python'")

    return _ke.asset.height_field_to_mesh(
        np.asarray(heights, dtype=np.float32),
        up_axis,
        horizontal_scale=float(horizontal_scale),
        center=bool(center),
    )


def height_field_to_mesh_python(
    heights,
    *,
    horizontal_scale: float = 1.0,
    up_axis=_ke.UpAxis.Y,
    center: bool = True,
):
    """Pure Python height field to ``scene.MeshData`` conversion.

    This is useful for prototyping terrain generators without adding new C++
    bindings. Use height_field_to_mesh() with the default ``backend="cpp"``
    when mesh generation itself becomes a hot path.
    """

    heights = np.asarray(heights, dtype=np.float32)
    if heights.ndim != 2:
        raise ValueError("height_field_to_mesh_python expected shape [rows, cols]")
    rows, cols = heights.shape
    if rows < 2 or cols < 2:
        raise ValueError(
            "height_field_to_mesh_python requires at least a 2x2 height field"
        )

    horizontal_scale = float(horizontal_scale)
    origin_x = (cols - 1) * 0.5 if center else 0.0
    origin_z = (rows - 1) * 0.5 if center else 0.0

    positions = np.empty((rows * cols, 3), dtype=np.float32)
    uvs = []
    for row in range(rows):
        z = (row - origin_z) * horizontal_scale
        v = row / (rows - 1)
        for col in range(cols):
            x = (col - origin_x) * horizontal_scale
            h = float(heights[row, col])
            index = row * cols + col
            if up_axis == _ke.UpAxis.Z:
                positions[index] = (x, z, h)
            else:
                positions[index] = (x, h, z)
            uvs.append(_ke.Vec2(col / (cols - 1), v))

    indices = []
    for row in range(rows - 1):
        row_base = row * cols
        next_row_base = (row + 1) * cols
        for col in range(cols - 1):
            i0 = row_base + col
            i1 = i0 + 1
            i2 = next_row_base + col
            i3 = i2 + 1
            indices.extend((i0, i2, i1, i1, i2, i3))

    normals = _height_field_normals(positions, indices)

    mesh = _ke.scene.MeshData()
    mesh.vertices = [_ke.Vec3(float(x), float(y), float(z)) for x, y, z in positions]
    mesh.normals = [_ke.Vec3(float(x), float(y), float(z)) for x, y, z in normals]
    mesh.uvs = uvs
    mesh.indices = indices
    return mesh


def _height_field_normals(positions: np.ndarray, indices: list[int]) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    for i in range(0, len(indices), 3):
        ia, ib, ic = indices[i], indices[i + 1], indices[i + 2]
        a = positions[ia]
        b = positions[ib]
        c = positions[ic]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length > 1.0e-8:
            normal /= length
            normals[ia] += normal
            normals[ib] += normal
            normals[ic] += normal

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-8
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0.0, 1.0, 0.0)
    return normals
