"""Mesh terrain generators with vertical faces (Z-up, centered at XY zero)."""

import numpy as np

from .terrain_mesh import TerrainMesh
from .heightfield import SubTerrain


_CORNERS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)
_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def pyramid_sloped_mesh(
    *,
    size=(8.0, 8.0),
    slope=0.2,
    platform_width=2.0,
    border_width=0.25,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
):
    """Create a pyramid clipped at the central platform's corner height.

    Unlike the legacy raised-platform heightfield generator, clipping keeps
    the platform continuous with the surrounding slopes. Negative slopes
    produce an inverted pyramid. The border remains at height zero.
    """
    if (
        min(size) <= 0
        or horizontal_scale <= 0
        or vertical_scale <= 0
        or border_width < 0
    ):
        raise ValueError("invalid slope terrain dimensions")
    nx, ny = (int(v / horizontal_scale) + 1 for v in size)
    border = int(border_width / horizontal_scale) + 1
    ix, iy = nx - 2 * border, ny - 2 * border
    if min(ix, iy) < 4 or not 0 < platform_width < min(ix, iy) * horizontal_scale:
        raise ValueError("platform does not fit inside the heightfield")
    cx, cy = ix // 2, iy // 2
    ramp_x = (cx - np.abs(np.arange(ix) - cx)) / cx
    ramp_y = (cy - np.abs(np.arange(iy) - cy)) / cy
    peak = int(slope * ix * horizontal_scale / 2 / vertical_scale)
    raw = peak * np.outer(ramp_y, ramp_x)
    half = int(platform_width / horizontal_scale / 2)
    height = raw[cy - half, cx - half]
    clipped = np.rint(np.clip(raw, min(0, height), max(0, height)))
    if not np.isfinite(clipped).all() or np.abs(clipped).max() > np.iinfo(np.int16).max:
        raise ValueError("slope heights exceed the heightfield's int16 range")
    field = SubTerrain(ny, nx, horizontal_scale, vertical_scale)
    field.height_field_raw[border:-border, border:-border] = clipped
    return TerrainMesh.from_heightfield(
        field, slope_threshold=slope_threshold, diagonal="main"
    )


def _box_arrays(boxes):
    vertices, indices = [], []
    for size, position in boxes:
        if min(size) <= 0:
            raise ValueError("terrain box dimensions must be positive")
        vertices.append(_CORNERS * np.asarray(size) / 2 + position)
        indices.append(_FACES + (len(vertices) - 1) * 8)
    return np.concatenate(vertices), np.concatenate(indices)


def _border(size, inner_size, top, bottom):
    x, y = size
    ix, iy = inner_size
    z, height = (top + bottom) / 2, top - bottom
    result = []
    if y > iy:
        for sign in (-1, 1):
            result.append(((x, (y - iy) / 2, height), (0, sign * (y + iy) / 4, z)))
    if x > ix:
        for sign in (-1, 1):
            result.append((((x - ix) / 2, iy, height), (sign * (x + ix) / 4, 0, z)))
    return result


def pyramid_stairs_mesh(
    *,
    size=(8.0, 8.0),
    step_height=0.1,
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
    inverted=False,
):
    """Create closed stair rings and a central platform, including risers.

    Dimensions follow IsaacLab's pyramid stair convention: the first ring is
    one step above/below the border and the platform is one further step.
    """
    if step_height <= 0 or step_width <= 0 or platform_width <= 0 or border_width < 0:
        raise ValueError("invalid stair dimensions")
    inner = tuple(float(v) - 2 * border_width for v in size)
    if min(inner) < platform_width:
        raise ValueError("platform does not fit inside the terrain")
    steps = int(min((v - platform_width) // (2 * step_width) + 1 for v in inner))
    total = (steps + 1) * step_height
    boxes = _border(size, inner, 0, -step_height)
    for k in range(steps + 1):
        widths = tuple(v - 2 * k * step_width for v in inner)
        top = (k + 1) * step_height * (-1 if inverted else 1)
        bottom = (
            -total
            if inverted and k < steps
            else (-total - step_height if inverted else -step_height)
        )
        if k == steps:
            boxes.append(
                ((widths[0], widths[1], top - bottom), (0, 0, (top + bottom) / 2))
            )
        else:
            next_size = tuple(v - 2 * step_width for v in widths)
            boxes.extend(_border(widths, next_size, top, bottom))
    return TerrainMesh(*_box_arrays(boxes))


def random_grid_mesh(
    *, size=(8.0, 8.0), grid_width=0.45, grid_height=0.1, platform_width=2.0, rng=None
):
    """Create random box tops and a flat raised central platform."""
    if (
        min(size) <= 0
        or grid_width <= 0
        or not 0 <= grid_height < 1
        or platform_width <= 0
    ):
        raise ValueError("invalid random grid dimensions")
    rng = np.random.default_rng() if rng is None else rng
    nx, ny = (int(v / grid_width) for v in size)
    if min(nx, ny) < 1:
        raise ValueError("grid_width exceeds terrain size")
    inner = (nx * grid_width, ny * grid_width)
    boxes = _border(size, inner, 0, -1)
    heights = rng.uniform(-grid_height, grid_height, (nx, ny))
    for x in range(nx):
        for y in range(ny):
            top = heights[x, y]
            boxes.append(
                (
                    (grid_width, grid_width, top + 1),
                    (
                        (x + 0.5) * grid_width - inner[0] / 2,
                        (y + 0.5) * grid_width - inner[1] / 2,
                        (top - 1) / 2,
                    ),
                )
            )
    boxes.append(
        (
            (platform_width, platform_width, 1 + grid_height),
            (0, 0, (grid_height - 1) / 2),
        )
    )
    return TerrainMesh(*_box_arrays(boxes))
