"""Construct shared rigid asset descriptions without a graphics context."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence
from weakref import WeakKeyDictionary

import numpy as np

from .._core import _ke
from .._public import unwrap_native
from ..material import PhongMaterial

if TYPE_CHECKING:
    from ..terrain import TerrainMesh
    from . import CollisionGeomDesc, InertialDesc, RigidDesc

_rigid_descriptions = WeakKeyDictionary()
_meshes = WeakKeyDictionary()
_surfaces = WeakKeyDictionary()


def load_obj_rigid(
    obj_path: str | Path,
    *,
    collision_geoms: Sequence[CollisionGeomDesc] | None = None,
    inertial: InertialDesc | None = None,
) -> RigidDesc:
    """Load a metre-scale Z-up OBJ with its material subsets and textures.

    Coordinates are preserved. Visual meshes retain normals and UVs; ray queries
    use these surfaces independently of physics collision geometry.

    Args:
        obj_path: OBJ file whose texture paths are resolved by the loader.
        collision_geoms: Colliders in the same local frame. None uses the OBJ
            triangle mesh, supported by static and kinematic instances. Dynamic
            bodies need primitive or convex colliders.
        inertial: Authored mass and inertia; None uses the instance density.
    """
    obj_path = Path(obj_path).expanduser().resolve()
    info = _ke.asset.load_obj_with_materials(str(obj_path))
    source_materials = info.materials
    materials = [_obj_material(source) for source in source_materials]

    # Keep material subsets separate while sharing each source mesh.
    parts = [(part.mesh_data, part.material_index) for part in info.subsets]
    if not parts:
        parts = [(info.mesh_data, info.primary_material_index)]
    visuals = [
        _ke.asset.VisualGeomDesc(
            mesh_data=mesh,
            material=materials[index] if index >= 0 else None,
            rgba=(
                1.0, 1.0, 1.0,
                source_materials[index].diffuse_color[3] if index >= 0 else 1.0,
            ),
        )
        for mesh, index in parts
    ]

    return _ke.asset.RigidDesc(
        visual_geoms=visuals,
        collision_geoms=(
            [_triangle_collider(info.mesh_data)]
            if collision_geoms is None else list(collision_geoms)
        ),
        inertial=inertial,
        asset_dir=str(obj_path.parent),
    )


def _obj_material(source):
    material = PhongMaterial()
    material.ambient = _ke.Vec3(*source.ambient_color[:3])
    material.diffuse = _ke.Vec3(*source.diffuse_color[:3])
    material.specular = _ke.Vec3(*source.specular_color[:3])
    material.shininess = source.shininess
    material.texture_paths = {
        f"{role}_map": path
        for role in ("diffuse", "specular", "alpha", "normal")
        if (path := getattr(source, f"{role}_texture_path"))
    }
    return material


def rigid_from_terrain_mesh(
    mesh: TerrainMesh,
    *,
    collision_geoms: Sequence[CollisionGeomDesc] | None = None,
    inertial: InertialDesc | None = None,
) -> RigidDesc:
    """Share a procedural surface's meshes and ray-query cache as one rigid.

    Geometry uses metres and Z-up. Without collision_geoms, the surface supplies
    triangle collision for static or kinematic instances. Dynamic bodies need
    primitive or convex colliders. Source geometry is immutable after creation.
    """
    if collision_geoms is None:
        collision_mesh = (
            mesh._meshes[0] if len(mesh._meshes) == 1
            else _ke.scene.MeshData.from_arrays(mesh.vertices, mesh.indices)
        )
        collision_geoms = [_triangle_collider(collision_mesh)]

    data = _ke.asset.RigidDesc(
        visual_geoms=[
            _ke.asset.VisualGeomDesc(mesh_data=part, rgba=(1.0, 1.0, 1.0, 1.0))
            for part in mesh._meshes
        ],
        collision_geoms=list(collision_geoms),
        inertial=inertial,
    )
    _surfaces[data] = mesh
    return data


def _triangle_collider(mesh):
    return _ke.asset.CollisionGeomDesc(
        type=_ke.asset.CollisionGeomDescType.TRIANGLE_MESH,
        mesh=unwrap_native(mesh),
    )


def _as_rigid_desc(data):
    # Existing single-body MJCF/URDF imports still use ArticulationDesc.
    if isinstance(data, _ke.asset.ArticulationDesc):
        if data not in _rigid_descriptions:
            _rigid_descriptions[data] = _ke.asset.RigidDesc.from_articulation(data)
        return _rigid_descriptions[data]
    return data


def _resolve_visual_mesh(data, geom):
    if geom.mesh_data is not None:
        return geom.mesh_data

    path = Path(data.asset_dir) / geom.mesh_file
    cache = _meshes.setdefault(data, {})
    if path not in cache:
        loaders = {".obj": _ke.asset.load_obj, ".stl": _ke.asset.load_stl}
        cache[path] = loaders[path.suffix.lower()](str(path))
    return cache[path]


def _visual_surface(data):
    if data not in _surfaces:
        if not data.visual_geoms:
            raise ValueError("mesh queries require authored visual_geoms")

        vertices, indices = [], []
        offset = 0
        for geom in data.visual_geoms:
            mesh = _resolve_visual_mesh(data, geom)
            points = np.array([(v.x, v.y, v.z) for v in mesh.vertices], np.float32)
            points *= (geom.scale.x, geom.scale.y, geom.scale.z)

            # Bake the visual-local pose once; instance poses remain query inputs.
            quat = np.asarray(geom.quat_xyzw, dtype=np.float32)
            twice_cross = 2.0 * np.cross(quat[:3], points)
            points += quat[3] * twice_cross + np.cross(quat[:3], twice_cross)
            points += (geom.pos.x, geom.pos.y, geom.pos.z)
            vertices.append(points)
            indices.append(np.asarray(mesh.indices, np.int32).reshape(-1, 3) + offset)
            offset += len(points)

        _surfaces[data] = _Surface(np.concatenate(vertices), np.concatenate(indices))
    return _surfaces[data]


class _Surface:
    def __init__(self, vertices, indices):
        vertices.flags.writeable = indices.flags.writeable = False
        self.vertices, self.indices = vertices, indices
        self._queries = {}

    def _get_query(self, device):
        from ..geometry._raycast import MeshRaycaster

        key = str(device)
        if key not in self._queries:
            self._queries[key] = MeshRaycaster(self.vertices, self.indices, device)
        return self._queries[key]
