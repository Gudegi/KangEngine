"""Primitive geometry data factories.

These functions create mesh payloads only. They do not define scene prims,
register resources, or create renderables. Use ``app.scene.add_mesh(...)`` or
``scene.Prim.set_mesh_data(...)`` to place the returned data in a scene.
"""

from __future__ import annotations

from .._core import _ke


create_cube_data = _ke.geometry.create_cube_data
create_plane_data = _ke.geometry.create_plane_data
create_sphere_data = _ke.geometry.create_sphere_data
create_box_data = _ke.geometry.create_box_data
create_cylinder_data = _ke.geometry.create_cylinder_data
create_arrow_data = _ke.geometry.create_arrow_data
create_capsule_data = _ke.geometry.create_capsule_data
create_cone_data = _ke.geometry.create_cone_data


__all__ = [
    "create_arrow_data",
    "create_box_data",
    "create_capsule_data",
    "create_cone_data",
    "create_cube_data",
    "create_cylinder_data",
    "create_plane_data",
    "create_sphere_data",
]
