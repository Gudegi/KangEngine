"""Rigid visual parts authored independently of simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .._core import _ke
from ..asset.rigid_asset import _resolve_visual_mesh
from ..rigid import expand_rigid_body_state, rigid_shape_specs
from ..utils.batched_rotations import quat_xyzw_to_matrix
from .. import render as render_api
from .utils import (
    apply_visual_alpha,
    normalize_color_array,
    visual_color,
    visual_material,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from .. import Vec4
    from ..app import App, RenderablePrimView
    from ..asset import RigidDesc
    from ..material import Material
    from ..render import TransformSource
    from ..scene import Prim


__all__ = ["RigidVisual"]


def _shape_mesh(spec):
    size = spec.size
    if spec.geom_type == "Sphere":
        return _ke.geometry.create_sphere_data(float(size[0]), 24, 12)
    if spec.geom_type == "Box":
        return _ke.geometry.create_box_data(*(float(v * 2.0) for v in size))
    if spec.geom_type == "Cylinder":
        return _ke.geometry.create_cylinder_data(
            float(size[0]), float(size[1] * 2.0), _ke.UpAxis.X, 24
        )
    if spec.geom_type == "Capsule":
        return _ke.geometry.create_capsule_data(
            float(size[0]), float(size[1] * 2.0), _ke.UpAxis.X, 24
        )
    if spec.geom_type in ("ConvexMesh", "TriangleMesh"):
        return spec.mesh_data
    raise ValueError(f"unsupported rigid visual geometry type: {spec.geom_type}")


class RigidVisual:
    """Create rigid visual parts from a descriptor without a physics actor.

    Authored visual geometry takes priority over collision geometry. SceneGraph
    instances start at the identity world pose; ExternalBuffer instances receive
    their transforms from the owning visual batch.
    """

    local_pos: npt.NDArray[np.float32]
    local_quat_xyzw: npt.NDArray[np.float32]
    local_scale: npt.NDArray[np.float32]
    base_colors: npt.NDArray[np.float32]

    def __init__(
        self,
        app: App,
        path: str,
        data: RigidDesc,
        *,
        material: Material | None = None,
        add_shapes: bool = True,
        color: Vec4 | npt.ArrayLike | None = None,
        transform_source: TransformSource | None = None,
    ) -> None:
        self.app = app
        self.path = path
        self._removed = False
        self._transform_source = (
            render_api.TransformSource.SCENE_GRAPH
            if transform_source is None
            else transform_source
        )
        self.views: list[RenderablePrimView] = []
        self.body_prims: list[Prim] = []
        self.body_handles: list[int] = []
        geoms = list(data.visual_geoms)
        self.specs = () if geoms else rigid_shape_specs(data)

        # Authored appearance is independent of the actor's collision shapes.
        if geoms:
            meshes = [_resolve_visual_mesh(data, geom) for geom in geoms]
            self.local_pos = np.array([geom.pos for geom in geoms], dtype=np.float32)
            self.local_quat_xyzw = np.array(
                [geom.quat_xyzw for geom in geoms], dtype=np.float32
            )
            self.local_scale = np.array(
                [geom.scale for geom in geoms], dtype=np.float32
            )
            self.base_colors = np.stack([visual_color(geom) for geom in geoms])
        else:
            meshes = [_shape_mesh(spec) for spec in self.specs]
            self.local_pos = np.stack([spec.local_pos for spec in self.specs])
            self.local_quat_xyzw = np.stack(
                [spec.local_quat_xyzw for spec in self.specs]
            )
            self.local_scale = np.ones((len(meshes), 3), dtype=np.float32)
            self.base_colors = np.ones((len(meshes), 4), dtype=np.float32)

        # Register each part with its own material and render handle.
        app.scene.define_prim(path, _ke.scene.PrimType.XFORM)
        rgba = normalize_color_array(color)
        for part_id, mesh in enumerate(meshes):
            geom = geoms[part_id] if geoms else None
            part_material = visual_material(
                app,
                geom,
                asset_dir=data.asset_dir,
                override=material,
            )
            display_color = (
                self.base_colors[part_id]
                if rgba is None
                else rgba * self.base_colors[part_id]
            )
            part_color = _ke.Vec4(*display_color)

            part_path = f"{path}/part_{part_id}"
            if add_shapes:
                view = app.scene.add_mesh(
                    part_path,
                    mesh,
                    material=part_material,
                    color=part_color,
                    transform_source=self._transform_source,
                )
                apply_visual_alpha(view, part_material, display_color)
                self.views.append(view)
                self.body_prims.append(view.prim)
                self.body_handles.append(
                    int(app.get_scene_render_system()._handle(view.component))
                )
            else:
                prim = app.scene.define_prim(part_path, _ke.scene.PrimType.MESH)
                component = prim.add_mesh_component()
                component.mesh_data = mesh
                component.resource_handle = app._register_mesh_resource(mesh)
                prim.set_display_color_alpha(part_color)
                self.body_prims.append(prim)

        if self._transform_source == render_api.TransformSource.SCENE_GRAPH:
            self.apply_pose(position=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0))

    def apply_pose(self, *, position: npt.ArrayLike, quat_xyzw: npt.ArrayLike) -> None:
        """Apply a world pose to the SceneGraph visual parts.

        Args:
            position: World translation in metres, shape (3,), as CPU values.
            quat_xyzw: World rotation in XYZW order, shape (4,), as CPU values.

        Raises:
            RuntimeError: The visual was removed or uses an external transform buffer.
        """
        if self._removed:
            raise RuntimeError("RigidVisual has been removed from the scene")
        if self._transform_source != render_api.TransformSource.SCENE_GRAPH:
            raise RuntimeError(
                "ExternalBuffer rigid poses are owned by the visual batch"
            )

        positions, rotations = expand_rigid_body_state(
            position,
            quat_xyzw,
            self.local_pos,
            self.local_quat_xyzw,
        )
        matrices = np.repeat(np.eye(4, dtype=np.float32)[None], len(positions), axis=0)
        matrices[:, :3, :3] = (
            quat_xyzw_to_matrix(torch.from_numpy(rotations)).numpy()
            * self.local_scale[:, None]
        )
        matrices[:, :3, 3] = positions
        for prim, matrix in zip(self.body_prims, matrices):
            prim.set_world_matrix(_ke.Mat4(np.ascontiguousarray(matrix.T)))

    def remove(self) -> None:
        """Remove this instance once, preserving shared resources and other instances."""
        if self._removed:
            return

        for view in self.views:
            view.remove()
        self.views.clear()
        self.body_handles.clear()
        self.body_prims.clear()
        self.app.scene.remove_prim(self.path)
        self._removed = True
