"""Height-field terrain construction and mesh conversion."""

from .._public import set_public_module
from .terrain_mesh import TerrainMesh, TerrainInstance
from .mesh_generators import pyramid_sloped_mesh, pyramid_stairs_mesh, random_grid_mesh
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
    "TerrainMesh",
    "TerrainInstance",
    "pyramid_stairs_mesh",
    "pyramid_sloped_mesh",
    "random_grid_mesh",
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
    TerrainMesh,
    TerrainInstance,
    pyramid_stairs_mesh,
    pyramid_sloped_mesh,
    random_grid_mesh,
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
