"""Visual surfaces whose vertex data deforms after scene registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

from .._core import _ke

if TYPE_CHECKING:
    from ..animation import SkeletonState, SkeletonTree


def _mesh_data(
    positions: npt.ArrayLike,
    normals: npt.ArrayLike,
    indices: npt.ArrayLike,
    uvs: npt.ArrayLike | None = None,
) -> Any:
    mesh = _ke.scene.MeshData()
    mesh.vertices = np.asarray(positions, np.float32).reshape(-1, 3).tolist()
    mesh.normals = np.asarray(normals, np.float32).reshape(-1, 3).tolist()
    mesh.indices = np.asarray(indices, np.int32).reshape(-1).tolist()
    if uvs is not None:
        mesh.uvs = np.asarray(uvs, np.float32).reshape(-1, 2).tolist()
    return mesh


def _vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    faces = indices.reshape(-1, 3)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-8)


@dataclass(frozen=True)
class SkinnedSurfaceAsset:
    """Static topology and four-weight LBS data uploaded once."""

    skeleton_tree: SkeletonTree
    positions: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    indices: npt.NDArray[np.int32]
    bone_indices: npt.NDArray[np.int32]
    bone_weights: npt.NDArray[np.float32]
    bone_node_indices: npt.NDArray[np.int32]
    inverse_bind_matrices: npt.NDArray[np.float32]
    uvs: npt.NDArray[np.float32] | None = None

    @classmethod
    def from_fbx(
        cls, skeleton_tree: SkeletonTree, mesh_info: Any
    ) -> SkinnedSurfaceAsset:
        """Convert one imported FBX skinned mesh without reading the file again."""
        positions = np.asarray(mesh_info.vertices, np.float32)
        indices = np.asarray(mesh_info.mesh_data.indices, np.int32)
        normals = np.asarray(mesh_info.normals, np.float32)
        if normals.shape != positions.shape:
            normals = _vertex_normals(positions, indices)
        imported_uvs = mesh_info.mesh_data.uvs
        uvs = (
            np.asarray(imported_uvs, np.float32)
            if len(imported_uvs) == len(positions)
            else None
        )
        return cls(
            skeleton_tree,
            positions,
            normals,
            indices,
            np.asarray(mesh_info.bone_indices, np.int32),
            np.asarray(mesh_info.bone_weights, np.float32),
            np.asarray(mesh_info.bone_node_indices, np.int32),
            np.asarray(mesh_info.inverse_bind_matrices, np.float32),
            uvs,
        )

    def __post_init__(self):
        positions = np.asarray(self.positions, np.float32)
        normals = np.asarray(self.normals, np.float32)
        indices = np.asarray(self.indices, np.int32)
        bone_indices = np.asarray(self.bone_indices, np.int32)
        bone_weights = np.asarray(self.bone_weights, np.float32)
        bone_nodes = np.asarray(self.bone_node_indices, np.int32)
        inverse_binds = np.asarray(self.inverse_bind_matrices, np.float32)
        uvs = None if self.uvs is None else np.asarray(self.uvs, np.float32)
        vertices = len(positions)
        if positions.shape != (vertices, 3) or normals.shape != positions.shape:
            raise ValueError("positions and normals must have shape [vertices, 3]")
        if bone_indices.shape != (vertices, 4) or bone_weights.shape != (vertices, 4):
            raise ValueError("bone indices and weights must have shape [vertices, 4]")
        if inverse_binds.shape != (len(bone_nodes), 4, 4):
            raise ValueError("inverse bind matrices must have shape [bones, 4, 4]")
        if np.any(bone_nodes < 0) or np.any(
            bone_nodes >= self.skeleton_tree.num_joints()
        ):
            raise ValueError("bone node indices must reference the skeleton tree")
        if np.any(bone_indices < -1) or np.any(bone_indices >= len(bone_nodes)):
            raise ValueError("bone indices must reference the surface bone palette")
        if uvs is not None and uvs.shape != (vertices, 2):
            raise ValueError("uvs must have shape [vertices, 2]")
        if np.any(bone_weights < 0.0):
            raise ValueError("bone weights must be non-negative")
        sums = bone_weights.sum(axis=1)
        if np.any(sums <= 1e-8):
            raise ValueError("every vertex must have at least one skinning weight")
        bone_weights = bone_weights / sums[:, None]
        for name, value in (
            ("positions", positions),
            ("normals", normals),
            ("indices", indices),
            ("bone_indices", bone_indices),
            ("bone_weights", bone_weights),
            ("bone_node_indices", bone_nodes),
            ("inverse_bind_matrices", inverse_binds),
        ):
            object.__setattr__(self, name, np.ascontiguousarray(value))
        if uvs is not None:
            object.__setattr__(self, "uvs", np.ascontiguousarray(uvs))

    def create_native(self) -> tuple[Any, Any]:
        mesh = _mesh_data(self.positions, self.normals, self.indices, self.uvs)
        skin = _ke.scene.SkinnedMeshData(
            mesh,
            self.bone_indices,
            self.bone_weights,
            self.bone_node_indices.tolist(),
            self.inverse_bind_matrices,
        )
        return mesh, skin


@dataclass
class _SkinnedSurfacePart:
    view: Any
    asset: SkinnedSurfaceAsset
    app: Any
    material: Any
    native_mesh: Any
    native_skin: Any
    bind_geometry_modified: bool = False
    texture_overrides: dict[Any, Any] | None = None
    alpha_mode: Any = None
    alpha_cutoff: float = 0.5
    opaque_alpha_mode: Any = None
    casts_shadow: bool = True


class SkinnedSurface:
    """One or more skinned render parts driven by a shared skeleton."""

    def __init__(
        self,
        view: Any,
        asset: SkinnedSurfaceAsset,
        app: Any,
        material: Any,
        native_mesh: Any,
        native_skin: Any,
    ):
        self._parts = [
            _SkinnedSurfacePart(
                view, asset, app, material, native_mesh, native_skin
            )
        ]
        self._removed = False

    @classmethod
    def _from_parts(cls, parts: list[_SkinnedSurfacePart]) -> SkinnedSurface:
        if not parts:
            raise ValueError("SkinnedSurface requires at least one render part")
        joint_count = parts[0].asset.skeleton_tree.num_joints()
        if any(part.asset.skeleton_tree.num_joints() != joint_count for part in parts):
            raise ValueError("all skinned surface parts must share one skeleton")
        surface = cls.__new__(cls)
        surface._parts = parts
        surface._removed = False
        return surface

    def _require_alive(self) -> None:
        if self._removed:
            raise RuntimeError("SkinnedSurface has been removed from the scene")

    @classmethod
    def create(
        cls,
        app: Any,
        path: str,
        asset: SkinnedSurfaceAsset,
        material: Any,
        color: Any = None,
    ) -> SkinnedSurface:
        mesh, skin = asset.create_native()
        prim = app.scene.define_prim(path, _ke.scene.PrimType.MESH)
        prim.set_mesh_data(mesh)
        if color is not None:
            prim.set_display_color_alpha(color)
        return cls(
            app.add_skinned_mesh(prim, material, skin),
            asset,
            app,
            material,
            mesh,
            skin,
        )

    @classmethod
    def create_from_fbx(
        cls,
        app: Any,
        path: str,
        skeleton_tree: SkeletonTree,
        mesh_info: Any,
        material: Any,
        color: Any = None,
    ) -> SkinnedSurface:
        """Create one surface from one already imported FBX skinned mesh."""
        asset = SkinnedSurfaceAsset.from_fbx(skeleton_tree, mesh_info)
        return cls.create(app, path, asset, material, color=color)

    @classmethod
    def create_from_fbx_result(
        cls,
        app: Any,
        path: str,
        result: Any,
        material: Any = None,
        color: Any = None,
        use_imported_materials: bool = True,
    ) -> SkinnedSurface:
        """Create one logical surface from every skinned mesh in an FBX result.

        By default, distinct imported FBX materials become retained PBR
        materials. Passing ``material`` applies one caller-owned override to
        every part instead.
        """
        motion = result.motion
        meshes = result.skinned_meshes
        if not meshes:
            raise ValueError("FBX result contains no skinned meshes")
        base_path = path.rstrip("/")
        parts: list[_SkinnedSurfacePart] = []
        imported_materials: list[Any | None] = []
        imported_pbr_materials: dict[tuple[Any, ...], Any] = {}

        def load_texture(texture_path: str) -> Any | None:
            if not texture_path or not Path(texture_path).is_file():
                return None
            return app.load_texture(Path(texture_path), flip=True)

        for index, mesh_info in enumerate(meshes):
            materials = list(getattr(mesh_info, "materials", ()))
            material_index = int(getattr(mesh_info, "primary_material_index", -1))
            imported_material = (
                materials[material_index]
                if 0 <= material_index < len(materials)
                else None
            )
            part_color = color
            part_material = material
            if part_material is None:
                diffuse = None
                normal = None
                base_color = _ke.Vec4(1.0, 1.0, 1.0, 1.0)
                material_key: tuple[Any, ...] = (None,)
                if use_imported_materials and imported_material is not None:
                    diffuse_path = imported_material.diffuse_texture_path
                    normal_path = imported_material.normal_texture_path
                    diffuse = load_texture(diffuse_path)
                    normal = load_texture(normal_path)
                    rgba = tuple(float(v) for v in imported_material.diffuse_color)
                    base_color = _ke.Vec4(*rgba)
                    material_key = (
                        imported_material.name,
                        rgba,
                        str(Path(diffuse_path).resolve()) if diffuse is not None else "",
                        str(Path(normal_path).resolve()) if normal is not None else "",
                    )
                part_material = imported_pbr_materials.get(material_key)
                if part_material is None:
                    part_material = app.create_pbr_material(
                        base_color=base_color,
                        metallic=0.0,
                        roughness=0.7,
                        base_color_texture=diffuse,
                        normal_texture=normal,
                    )
                    imported_pbr_materials[material_key] = part_material
            elif (
                part_color is None
                and use_imported_materials
                and imported_material is not None
            ):
                part_color = _ke.Vec4(*imported_material.diffuse_color)
            name = (
                "".join(
                    character if character.isalnum() or character == "_" else "_"
                    for character in mesh_info.name
                ).strip("_")
                or f"mesh_{index}"
            )
            child = cls.create_from_fbx(
                app,
                f"{base_path}/{index}_{name}",
                motion.skeleton_tree,
                mesh_info,
                part_material,
                color=part_color,
            )
            parts.extend(child._parts)
            imported_materials.append(imported_material)

        surface = cls._from_parts(parts)
        if use_imported_materials:
            for part, imported_material in zip(parts, imported_materials):
                if imported_material is None:
                    continue
                diffuse = load_texture(imported_material.diffuse_texture_path)
                if diffuse is not None and material is not None:
                    part.view.set_texture(diffuse, 0)
                    if part.texture_overrides is None:
                        part.texture_overrides = {}
                    part.texture_overrides[0] = diffuse
                if diffuse is not None:
                    if Path(
                        imported_material.diffuse_texture_path
                    ).suffix.lower() in {".png", ".tga"}:
                        # FBX frequently omits alpha semantics for face and
                        # foliage atlases. Match the generic FBX mesh importer:
                        # keep depth writes and discard transparent texels.
                        part.view.set_alpha_mode(_ke.AlphaMode.MASK, 0.5)
                        part.alpha_mode = _ke.AlphaMode.MASK
                        part.alpha_cutoff = 0.5
                        part.opaque_alpha_mode = _ke.AlphaMode.MASK
                normal = load_texture(imported_material.normal_texture_path)
                if normal is not None and material is not None:
                    part.view.set_texture(normal, 5)
                    if part.texture_overrides is None:
                        part.texture_overrides = {}
                    part.texture_overrides[5] = normal
        return surface

    def create_instance(
        self,
        path: str,
        *,
        material: Any = None,
        color: Any = None,
    ) -> SkinnedSurface:
        """Create an independently posed instance sharing this surface's assets.

        Bind geometry, skin weights, textures, and materials are shared unless
        a material override is supplied. The new prims and runtime skinning
        buffers are independent from the source surface.
        """
        self._require_alive()
        base_path = path.rstrip("/")
        if not base_path:
            raise ValueError("path must not be empty")
        parts: list[_SkinnedSurfacePart] = []
        multi_part = len(self._parts) > 1
        for index, source in enumerate(self._parts):
            app = source.app
            source_name = source.view.path.rsplit("/", 1)[-1]
            part_path = (
                f"{base_path}/{index}_{source_name}" if multi_part else base_path
            )
            prim = app.scene.define_prim(part_path, _ke.scene.PrimType.MESH)
            prim.set_mesh_data(source.native_mesh)
            if color is None:
                prim.set_display_color_alpha(source.view.get_base_color())
            else:
                prim.set_display_color_alpha(color)
            part_material = source.material if material is None else material
            view = app.add_skinned_mesh(prim, part_material, source.native_skin)
            part = _SkinnedSurfacePart(
                view,
                source.asset,
                app,
                part_material,
                source.native_mesh,
                source.native_skin,
                texture_overrides=dict(source.texture_overrides or {}),
                alpha_mode=source.alpha_mode,
                alpha_cutoff=source.alpha_cutoff,
                opaque_alpha_mode=source.opaque_alpha_mode,
                casts_shadow=source.casts_shadow,
            )
            for role, texture in part.texture_overrides.items():
                view.set_texture(texture, role)
            if part.alpha_mode is not None:
                view.set_alpha_mode(part.alpha_mode, part.alpha_cutoff)
            view.set_casts_shadow(part.casts_shadow)
            parts.append(part)
        instance = type(self)._from_parts(parts)
        if color is not None:
            values = tuple(float(value) for value in color)
            if len(values) == 4 and values[3] < 1.0:
                instance.set_alpha_mode(_ke.AlphaMode.BLEND)
        return instance

    @property
    def prim(self) -> Any:
        self._require_alive()
        if len(self._parts) != 1:
            raise ValueError("prim is available only for a single-part surface")
        return self._parts[0].view.prim

    @property
    def prims(self) -> tuple[Any, ...]:
        self._require_alive()
        return tuple(part.view.prim for part in self._parts)

    @property
    def view(self) -> Any:
        self._require_alive()
        if len(self._parts) != 1:
            raise ValueError("view is available only for a single-part surface")
        return self._parts[0].view

    @property
    def views(self) -> tuple[Any, ...]:
        self._require_alive()
        return tuple(part.view for part in self._parts)

    @property
    def asset(self) -> SkinnedSurfaceAsset:
        self._require_alive()
        if len(self._parts) != 1:
            raise ValueError("asset is available only for a single-part surface")
        return self._parts[0].asset

    @property
    def assets(self) -> tuple[SkinnedSurfaceAsset, ...]:
        self._require_alive()
        return tuple(part.asset for part in self._parts)

    @property
    def skeleton_tree(self) -> SkeletonTree:
        self._require_alive()
        return self._parts[0].asset.skeleton_tree

    def remove(self) -> bool:
        """Remove every render part from the scene.

        Shared assets and materials remain alive for other surface instances.
        Calling ``remove`` more than once is safe and returns ``False`` after
        the first call.
        """
        if self._removed:
            return False
        removed = False
        for part in self._parts:
            removed = bool(part.view.remove()) or removed
        self._removed = True
        return removed

    def update_joint_matrices(self, joint_matrices: npt.ArrayLike) -> SkinnedSurface:
        self._require_alive()
        if len(self._parts) != 1:
            raise ValueError(
                "update_joint_matrices is available only for a single-part surface"
            )
        self._parts[0].view.update_skinning(joint_matrices)
        return self

    def update_bind_geometry(
        self,
        positions: npt.ArrayLike,
        normals: npt.ArrayLike | None = None,
    ) -> SkinnedSurface:
        """Update pre-skinning vertices on a single-part surface."""
        self._require_alive()
        if len(self._parts) != 1:
            raise ValueError(
                "update_bind_geometry is available only for a single-part surface"
            )
        part = self._parts[0]
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        if len(positions) != len(part.asset.positions):
            raise ValueError("positions must match the skinned surface vertices")
        if normals is not None:
            normals = np.asarray(normals, np.float32).reshape(-1, 3)
        part.view.update_geometry(positions, normals)
        part.bind_geometry_modified = True
        return self

    def reset_bind_geometry(self) -> SkinnedSurface:
        """Restore original bind vertices on a single-part surface."""
        self._require_alive()
        if len(self._parts) != 1:
            raise ValueError(
                "reset_bind_geometry is available only for a single-part surface"
            )
        part = self._parts[0]
        if part.bind_geometry_modified:
            part.view.update_geometry(part.asset.positions, part.asset.normals)
            part.bind_geometry_modified = False
        return self

    def set_visible(self, visible: bool) -> SkinnedSurface:
        self._require_alive()
        for part in self._parts:
            part.view.set_visible(visible)
        return self

    def set_casts_shadow(self, casts_shadow: bool) -> SkinnedSurface:
        self._require_alive()
        for part in self._parts:
            part.view.set_casts_shadow(casts_shadow)
            part.casts_shadow = bool(casts_shadow)
        return self

    def set_alpha_mode(self, mode: Any, cutoff: float = 0.5) -> SkinnedSurface:
        self._require_alive()
        for part in self._parts:
            part.view.set_alpha_mode(mode, cutoff)
            part.alpha_mode = mode
            part.alpha_cutoff = float(cutoff)
            if mode != _ke.AlphaMode.BLEND:
                part.opaque_alpha_mode = mode
        return self

    def set_color(self, color: Any) -> SkinnedSurface:
        self._require_alive()
        for part in self._parts:
            part.view.set_base_color(color)
        return self

    def set_alpha(self, alpha: float) -> SkinnedSurface:
        """Set per-instance opacity without modifying the shared material."""
        self._require_alive()
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        for part in self._parts:
            color = part.view.get_base_color()
            part.view.set_base_color((color.x, color.y, color.z, alpha))
            mode = (
                _ke.AlphaMode.BLEND
                if alpha < 1.0
                else part.opaque_alpha_mode or _ke.AlphaMode.OPAQUE
            )
            part.view.set_alpha_mode(mode, part.alpha_cutoff)
            part.alpha_mode = mode
        return self

    @overload
    def apply_state(self, state: SkeletonState) -> SkinnedSurface: ...

    @overload
    def apply_state(
        self,
        root_translation: npt.ArrayLike,
        local_rotations_wxyz: npt.ArrayLike,
    ) -> SkinnedSurface: ...

    def apply_state(
        self,
        state_or_root_translation: SkeletonState | npt.ArrayLike,
        local_rotations_wxyz: npt.ArrayLike | None = None,
    ) -> SkinnedSurface:
        """Apply a SkeletonState or a root translation and local rotations."""
        self._require_alive()
        if local_rotations_wxyz is None:
            state = state_or_root_translation
            if not hasattr(state, "compute_global_matrices"):
                raise TypeError(
                    "apply_state expects a SkeletonState or "
                    "(root_translation, local_rotations_wxyz)"
                )
        else:
            state = _ke.animation.SkeletonState.from_rotation_and_root_translation(
                self.skeleton_tree,
                local_rotations_wxyz,
                state_or_root_translation,
                True,
            )
        if state.num_joints() != self.skeleton_tree.num_joints():
            raise ValueError("state skeleton does not match the skinned surface")
        globals_ = state.compute_global_matrices()
        for part in self._parts:
            matrices = _ke.animation.compute_skinning_matrices(
                part.asset.bone_node_indices,
                part.asset.inverse_bind_matrices,
                globals_,
            )
            part.view.update_skinning(matrices)
        return self


class DeformableSurface:
    """TODO scaffold for directly streamed deformable geometry.

    This path does not yet guarantee per-surface geometry isolation or complete
    dynamic bounds handling. It is not a supported cloth/soft-body visual API.
    """

    def __init__(self, view: Any):
        self.view = view

    @classmethod
    def create(
        cls,
        app: Any,
        path: str,
        positions: npt.ArrayLike,
        normals: npt.ArrayLike,
        indices: npt.ArrayLike,
        material: Any,
        color: Any = None,
    ) -> DeformableSurface:
        mesh = _mesh_data(positions, normals, indices)
        return cls(app.scene.add_mesh(path, mesh, material, color=color))

    @property
    def prim(self) -> Any:
        return self.view.prim

    def update(
        self,
        positions: npt.ArrayLike,
        normals: npt.ArrayLike | None = None,
    ) -> DeformableSurface:
        """Upload geometry through the current experimental batch path."""
        self.view.update_geometry(positions, normals)
        return self

    def set_visible(self, visible: bool) -> DeformableSurface:
        self.view.set_visible(visible)
        return self

    def set_casts_shadow(self, casts_shadow: bool) -> DeformableSurface:
        self.view.set_casts_shadow(casts_shadow)
        return self

    def set_alpha_mode(self, mode: Any, cutoff: float = 0.5) -> DeformableSurface:
        self.view.set_alpha_mode(mode, cutoff)
        return self


__all__ = [
    "DeformableSurface",
    "SkinnedSurface",
    "SkinnedSurfaceAsset",
]
