"""Height-field terrain construction and mesh conversion."""

from .._public import set_public_module
from .heightfield import (
    SubTerrain,
    TerrainGrid,
    discrete_obstacles_terrain,
    height_field_to_mesh,
    height_field_to_mesh_python,
    pyramid_sloped_terrain,
    random_uniform_terrain,
    sloped_terrain,
    stairs_terrain,
    wave_terrain,
)

__all__ = [
    "SubTerrain",
    "TerrainGrid",
    "discrete_obstacles_terrain",
    "height_field_to_mesh",
    "height_field_to_mesh_python",
    "pyramid_sloped_terrain",
    "random_uniform_terrain",
    "sloped_terrain",
    "stairs_terrain",
    "wave_terrain",
]

for _value in (
    SubTerrain,
    TerrainGrid,
    discrete_obstacles_terrain,
    height_field_to_mesh,
    height_field_to_mesh_python,
    pyramid_sloped_terrain,
    random_uniform_terrain,
    sloped_terrain,
    stairs_terrain,
    wave_terrain,
):
    set_public_module(_value, __name__)

del _value
