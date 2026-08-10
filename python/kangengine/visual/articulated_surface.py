"""Kinematics-owned rigid-link surfaces."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy.typing as npt
import numpy as np

from .._core import _ke

if TYPE_CHECKING:
    from ..animation import SkeletonState, SkeletonTree
    from ..app import App, RenderablePrimView
    from ..material import Material
    from ..scene import Prim


def _mesh_asset_path(mjcf_path: Path, scale: float, order: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", mjcf_path.name).strip("_") or "robot"
    key = f"{mjcf_path}|{float(scale):.9g}|{order}".encode()
    digest = hashlib.sha1(key).hexdigest()[:10]
    return f"/.Resources/Meshes/ArticulatedSurfaces/{stem}_{digest}"


def _copy_skeleton_tree(tree: Any) -> Any:
    translations = np.empty((tree.num_joints(), 3), dtype=np.float32)
    rotations = np.empty((tree.num_joints(), 4), dtype=np.float32)
    for index in range(tree.num_joints()):
        translation = tree.local_translation(index)
        rotation = tree.local_rotation(index)
        translations[index] = (translation.x, translation.y, translation.z)
        rotations[index] = (rotation.w, rotation.x, rotation.y, rotation.z)
    return _ke.animation.SkeletonTree(
        tree.node_names(), tree.parent_indices(), translations, rotations
    )


def _alpha_from_color(color: npt.ArrayLike) -> float | None:
    if hasattr(color, "w"):
        return float(color.w)
    values = np.asarray(color, dtype=np.float32).reshape(-1)
    return float(values[3]) if len(values) >= 4 else None


class ArticulatedSurfaceAsset:
    """Reusable MJCF skeleton and rigid-link mesh asset."""

    def __init__(self, native: Any, mesh_asset_path: str):
        self._native = native
        self._mesh_asset_path = mesh_asset_path
        self._defined_apps: set[int] = set()
        self._skeleton_tree = None

    @classmethod
    def from_mjcf(
        cls,
        mjcf_path: str | Path,
        *,
        scale: float = 1.0,
        order: str = "DFS",
    ) -> ArticulatedSurfaceAsset:
        """Load a reusable kinematic surface asset from MJCF."""
        path = Path(mjcf_path).expanduser().resolve()
        native = _ke.animation.ArticulationVisualAsset.from_mjcf(
            str(path), float(scale), order
        )
        return cls(native, _mesh_asset_path(path, scale, order))

    @property
    def num_bodies(self) -> int:
        return self._native.num_bodies()

    def create(
        self,
        app: App,
        path: str,
        material: Material,
        *,
        color: npt.ArrayLike | None = None,
    ) -> ArticulatedSurface:
        """Create an independently posed surface instance."""
        scene = app.scene.native
        app_key = id(app)
        if app_key not in self._defined_apps:
            self._native.define_mesh_assets(scene, self._mesh_asset_path, True)
            self._defined_apps.add(app_key)
        native = self._native.instantiate(
            scene, path.rstrip("/"), self._mesh_asset_path, True
        )
        views = [app.scene.add_renderable(prim, material) for prim in native.render_prims()]
        surface = ArticulatedSurface(app, path.rstrip("/"), self, native, views, material)
        if color is not None:
            surface.set_color(color)
        return surface


class ArticulatedSurface:
    """Application-owned FK surface composed of rigid link meshes."""

    def __init__(
        self,
        app: App,
        path: str,
        asset: ArticulatedSurfaceAsset,
        native: Any,
        views: list[RenderablePrimView],
        material: Material,
    ):
        self._app = app
        self._path = path
        self._asset = asset
        self._native = native
        self._views = views
        self._material = material
        if asset._skeleton_tree is None:
            asset._skeleton_tree = _copy_skeleton_tree(native.skeleton())
        self._skeleton_tree = asset._skeleton_tree
        self._removed = False
        self._opaque_alpha_modes = [_ke.AlphaMode.OPAQUE] * len(views)

    @classmethod
    def create_from_mjcf(
        cls,
        app: App,
        path: str,
        mjcf_path: str | Path,
        material: Material,
        *,
        color: npt.ArrayLike | None = None,
        scale: float = 1.0,
        order: str = "DFS",
    ) -> ArticulatedSurface:
        """Load an MJCF asset and create its first kinematic instance."""
        asset = ArticulatedSurfaceAsset.from_mjcf(
            mjcf_path, scale=scale, order=order
        )
        return asset.create(app, path, material, color=color)

    def _require_alive(self) -> None:
        if self._removed:
            raise RuntimeError("ArticulatedSurface has been removed from the scene")

    @property
    def asset(self) -> ArticulatedSurfaceAsset:
        return self._asset

    @property
    def skeleton_tree(self) -> SkeletonTree:
        self._require_alive()
        return self._skeleton_tree

    @property
    def prims(self) -> tuple[Prim, ...]:
        self._require_alive()
        return tuple(view.prim for view in self._views)

    @property
    def views(self) -> tuple[RenderablePrimView, ...]:
        self._require_alive()
        return tuple(self._views)

    def create_instance(
        self,
        path: str,
        *,
        material: Material | None = None,
        color: npt.ArrayLike | None = None,
    ) -> ArticulatedSurface:
        """Create a visual-only instance initialized from the current pose."""
        self._require_alive()
        instance = self._asset.create(
            self._app,
            path,
            self._material if material is None else material,
            color=color,
        )
        instance.apply_state(self._native.state())
        if color is None:
            for source, target in zip(self._views, instance._views):
                source_color = source.get_base_color()
                target.set_base_color(source_color)
                if source_color.w < 1.0:
                    target.set_alpha_mode(_ke.AlphaMode.BLEND)
        return instance

    @overload
    def apply_state(self, state: SkeletonState) -> ArticulatedSurface: ...

    @overload
    def apply_state(
        self,
        root_translation: npt.ArrayLike,
        local_rotations_wxyz: npt.ArrayLike,
    ) -> ArticulatedSurface: ...

    def apply_state(
        self,
        state_or_root_translation: SkeletonState | npt.ArrayLike,
        local_rotations_wxyz: npt.ArrayLike | None = None,
    ) -> ArticulatedSurface:
        """Apply a SkeletonState or root translation and WXYZ local rotations."""
        self._require_alive()
        if local_rotations_wxyz is None:
            state = state_or_root_translation
            if not hasattr(state, "num_joints"):
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
            raise ValueError("state skeleton does not match the articulated surface")
        self._native.set_root_translation(state.root_translation())
        for index in range(state.num_joints()):
            self._native.set_joint_rotation(index, state.rotation(index))
        self._native.apply_pose()
        return self

    def set_color(self, color: npt.ArrayLike) -> ArticulatedSurface:
        self._require_alive()
        for view in self._views:
            view.set_base_color(color)
        alpha = _alpha_from_color(color)
        if alpha is not None and alpha < 1.0:
            self.set_alpha(alpha)
        return self

    def set_alpha_mode(
        self, mode: Any, cutoff: float = 0.5
    ) -> ArticulatedSurface:
        self._require_alive()
        for index, view in enumerate(self._views):
            view.set_alpha_mode(mode, cutoff)
            if mode != _ke.AlphaMode.BLEND:
                self._opaque_alpha_modes[index] = mode
        return self

    def set_alpha(self, alpha: float) -> ArticulatedSurface:
        self._require_alive()
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        for index, view in enumerate(self._views):
            color = view.get_base_color()
            view.set_base_color((color.x, color.y, color.z, alpha))
            mode = (
                _ke.AlphaMode.BLEND
                if alpha < 1.0
                else self._opaque_alpha_modes[index]
            )
            view.set_alpha_mode(mode)
        return self

    def set_visible(self, visible: bool) -> ArticulatedSurface:
        self._require_alive()
        for view in self._views:
            view.set_visible(visible)
        return self

    def set_casts_shadow(self, casts_shadow: bool) -> ArticulatedSurface:
        self._require_alive()
        for view in self._views:
            view.set_casts_shadow(casts_shadow)
        return self

    def remove(self) -> bool:
        """Remove the instance while preserving its shared asset resources."""
        if self._removed:
            return False
        removed = bool(self._app.remove_prim(self._path))
        self._removed = True
        return removed


__all__ = ["ArticulatedSurface", "ArticulatedSurfaceAsset"]
