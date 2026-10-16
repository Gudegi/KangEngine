"""Native articulated visual assets and their scene-prim instances."""

from __future__ import annotations

from typing import Never, overload

import numpy.typing as npt
import torch

from .. import Quat, Vec3
from ..animation import SkeletonState, SkeletonTree
from ..asset import ArticulationDesc
from ..scene import Prim, SceneBackend

__all__ = ["ArticulationVisual", "ArticulationVisualAsset"]

class ArticulationVisual:
    """Articulated scene prims driven by an internal skeleton state.

    Create through from_mjcf/from_urdf or ArticulationVisualAsset.instantiate;
    direct construction is unavailable. Scene arguments are native SceneBackend
    objects, such as app.get_native_scene(). The scene owns all returned prims
    and must keep them alive while this visual is used. Pose arrays are CPU values.
    """

    def __new__(cls) -> Never:
        """Direct construction raises TypeError; use a factory method."""
        ...

    @staticmethod
    def from_mjcf(
        mjcf_path: str,
        scene: SceneBackend,
        path: str = "/robot",
        scale: float = 1.0,
        order: str = "DFS",
        mesh_asset_base_path: str = "",
        target_coordinate_system: str = "z_up_x_forward",
    ) -> ArticulationVisual:
        """Load MJCF geometry and author its body prims in the supplied scene."""
        ...

    @staticmethod
    def from_urdf(
        urdf_path: str,
        scene: SceneBackend,
        path: str = "/robot",
        scale: float = 1.0,
        order: str = "DFS",
        mesh_asset_base_path: str = "",
        target_coordinate_system: str = "z_up_x_forward",
    ) -> ArticulationVisual:
        """Load URDF geometry and author its body prims in the supplied scene."""
        ...

    def apply_pose(self) -> None:
        """Apply the stored skeleton pose to the body prims."""
        ...
    @overload
    def set_joint_quat_wxyz(
        self, index: int, quat_wxyz: npt.ArrayLike | torch.Tensor
    ) -> None:
        """Set a node rotation from WXYZ values (4,) in the state's frame.

        Call apply_pose after modifying the stored pose.
        """
        ...
    @overload
    def set_joint_quat_wxyz(self, index: int, quat_wxyz: Quat) -> None: ...
    @overload
    def set_root_translation(self, translation: npt.ArrayLike | torch.Tensor) -> None:
        """Set root translation (3,) in skeleton coordinates before asset scaling.

        Call apply_pose after modifying the stored pose.
        """
        ...
    @overload
    def set_root_translation(self, translation: Vec3) -> None: ...
    def reset_to_zero_pose(self) -> None:
        """Reset the stored pose and immediately apply it to the scene prims."""
        ...
    def set_pickable(self, pickable: bool) -> None:
        """Set whether the rendered geometry participates in picking."""
        ...
    def skeleton(self) -> SkeletonTree:
        """Return the borrowed hierarchy; this reference keeps the visual alive."""
        ...
    def state(self) -> SkeletonState:
        """Return borrowed mutable state; this reference keeps the visual alive.

        Call apply_pose after editing it to update the scene prims.
        """
        ...
    def body_prim(self, index: int) -> Prim:
        """Return a scene-owned body prim at a valid skeleton-node index."""
        ...
    def body_prims(self) -> list[Prim]:
        """Return scene-owned body prims, including nodes without visual meshes."""
        ...
    def render_prims(self) -> list[Prim]:
        """Return scene-owned mesh prims, possibly several per body."""
        ...
    def render_prim_body_indices(self) -> list[int]:
        """Return the owning skeleton-node index for each render prim."""
        ...
    def render_prim_visual_indices(self) -> list[int]:
        """Return descriptor visual indices, or -1 for body/collision meshes."""
        ...
    def num_bodies(self) -> int:
        """Return the number of skeleton body nodes."""
        ...

class ArticulationVisualAsset:
    """Reusable articulated geometry prepared for scene instantiation.

    Create through a from_* factory; direct construction is unavailable.
    Instantiation authors scene prims without registering their renderer materials.
    """

    def __new__(cls) -> Never:
        """Direct construction raises TypeError; use a factory method."""
        ...

    @staticmethod
    def from_data(
        data: ArticulationDesc, scale: float = 1.0, asset_path: str = ""
    ) -> ArticulationVisualAsset:
        """Prepare reusable visual geometry from a parsed articulation descriptor."""
        ...

    @staticmethod
    def from_mjcf(
        mjcf_path: str,
        scale: float = 1.0,
        order: str = "DFS",
        target_coordinate_system: str = "z_up_x_forward",
    ) -> ArticulationVisualAsset:
        """Load reusable articulation geometry from MJCF."""
        ...

    @staticmethod
    def from_urdf(
        urdf_path: str,
        scale: float = 1.0,
        order: str = "DFS",
        target_coordinate_system: str = "z_up_x_forward",
    ) -> ArticulationVisualAsset:
        """Load reusable articulation geometry from URDF."""
        ...
    def define_mesh_assets(
        self,
        scene: SceneBackend,
        mesh_asset_base_path: str,
        split_visual_geoms: bool = False,
    ) -> None:
        """Define shared mesh prims, optionally preserving individual visual geoms."""
        ...
    def instantiate(
        self,
        scene: SceneBackend,
        path: str = "/robot",
        mesh_asset_base_path: str = "",
        split_visual_geoms: bool = False,
        hierarchical: bool = False,
    ) -> ArticulationVisual:
        """Author an instance in a native scene that owns its prims.

        A nonempty mesh_asset_base_path shares mesh prims at that scene location.
        split_visual_geoms preserves separate visual parts; hierarchical preserves
        the source skeleton's parent-child prim hierarchy.
        """
        ...
    def num_bodies(self) -> int:
        """Return the number of skeleton body nodes in the asset."""
        ...
