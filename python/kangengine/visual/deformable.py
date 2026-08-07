"""Visual surfaces whose vertex data deforms after scene registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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


@dataclass(frozen=True)
class SkinnedSurfaceAsset:
    """Static topology and four-weight LBS data uploaded once."""

    positions: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    indices: npt.NDArray[np.int32]
    bone_indices: npt.NDArray[np.int32]
    bone_weights: npt.NDArray[np.float32]
    bone_node_indices: npt.NDArray[np.int32]
    inverse_bind_matrices: npt.NDArray[np.float32]
    uvs: npt.NDArray[np.float32] | None = None

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
        if uvs is not None and uvs.shape != (vertices, 2):
            raise ValueError("uvs must have shape [vertices, 2]")
        if np.any(bone_weights < 0.0):
            raise ValueError("bone weights must be non-negative")
        sums = bone_weights.sum(axis=1)
        if np.any(sums <= 1e-8):
            raise ValueError("every vertex must have at least one skinning weight")
        bone_weights = bone_weights / sums[:, None]
        for name, value in (
            ("positions", positions), ("normals", normals), ("indices", indices),
            ("bone_indices", bone_indices), ("bone_weights", bone_weights),
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


class SkinnedSurface:
    """Scene object updated by a small joint-matrix palette."""

    def __init__(self, view: Any, asset: SkinnedSurfaceAsset):
        self.view = view
        self.asset = asset
        self._bind_geometry_modified = False

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
        return cls(app.add_skinned_mesh(prim, material, skin), asset)

    @property
    def prim(self) -> Any:
        return self.view.prim

    def update_joint_matrices(
        self, joint_matrices: npt.ArrayLike
    ) -> SkinnedSurface:
        self.view.update_skinning(joint_matrices)
        return self

    def update_bind_geometry(
        self,
        positions: npt.ArrayLike,
        normals: npt.ArrayLike | None = None,
    ) -> SkinnedSurface:
        """Update pre-skinning vertices while preserving skin weights."""
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        if len(positions) != len(self.asset.positions):
            raise ValueError("positions must match the skinned surface vertices")
        if normals is not None:
            normals = np.asarray(normals, np.float32).reshape(-1, 3)
        self.view.update_geometry(positions, normals)
        self._bind_geometry_modified = True
        return self

    def reset_bind_geometry(self) -> SkinnedSurface:
        """Restore the asset's original pre-skinning vertices and normals."""
        if self._bind_geometry_modified:
            self.view.update_geometry(self.asset.positions, self.asset.normals)
            self._bind_geometry_modified = False
        return self

    def set_visible(self, visible: bool) -> SkinnedSurface:
        self.view.set_visible(visible)
        return self

    def set_casts_shadow(self, casts_shadow: bool) -> SkinnedSurface:
        self.view.set_casts_shadow(casts_shadow)
        return self

    def set_alpha_mode(self, mode: Any, cutoff: float = 0.5) -> SkinnedSurface:
        self.view.set_alpha_mode(mode, cutoff)
        return self

    def apply_state(self, state: SkeletonState) -> SkinnedSurface:
        globals_ = state.compute_global_matrices()
        matrices = _ke.animation.compute_skinning_matrices(
            self.asset.bone_node_indices,
            self.asset.inverse_bind_matrices,
            globals_,
        )
        return self.update_joint_matrices(matrices)

    def set_pose(
        self,
        skeleton_tree: SkeletonTree,
        rotations_wxyz: npt.ArrayLike,
        root_translation: npt.ArrayLike,
        is_local: bool = True,
    ) -> SkeletonState:
        state = _ke.animation.SkeletonState.from_rotation_and_root_translation(
            skeleton_tree, rotations_wxyz, root_translation, is_local
        )
        self.apply_state(state)
        return state


class DeformableSurface:
    """Dynamic vertex stream suitable for cloth, soft bodies, and correctives."""

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
        self.view.update_geometry(positions, normals)
        return self

    def set_visible(self, visible: bool) -> DeformableSurface:
        self.view.set_visible(visible)
        return self

    def set_casts_shadow(self, casts_shadow: bool) -> DeformableSurface:
        self.view.set_casts_shadow(casts_shadow)
        return self

    def set_alpha_mode(
        self, mode: Any, cutoff: float = 0.5
    ) -> DeformableSurface:
        self.view.set_alpha_mode(mode, cutoff)
        return self


__all__ = [
    "DeformableSurface",
    "SkinnedSurface",
    "SkinnedSurfaceAsset",
]
