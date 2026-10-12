"""Integrated FBX skinned-character visualization."""

from __future__ import annotations

import numpy.typing as npt

from .. import Vec4
from ..animation import SkeletonMotion, SkeletonState, SkeletonTree
from ..app import App
from ..material import Material

class SkinVisual:
    """FBX skinning with an imported clip or an external compatible skeleton.

    With skeleton_tree, fbx_path supplies only the skin. Bone names must match;
    this does not retarget different proportions or joint axes. Apply external
    poses with apply_state/apply_pose; motion/apply_time require an imported clip.
    Do not combine skeleton_tree with bind_fbx_path, clip_index or fps.
    """

    @staticmethod
    def from_fbx(
        app: App,
        material: Material,
        fbx_path: str,
        bind_fbx_path: str | None = None,
        path: str = "/fbx_character",
        clip_index: int = -1,
        fps: float = -1.0,
        scale: float = 0.01,
        use_materials: bool = True,
        *,
        skeleton_tree: SkeletonTree | None = None,
    ) -> SkinVisual: ...

    def apply_time(self, time: float, loop: bool = True) -> SkeletonState: ...
    def apply_state(self, state: SkeletonState) -> SkeletonState: ...
    def apply_pose(
        self,
        root_translation: npt.ArrayLike,
        local_rotations_wxyz: npt.ArrayLike,
    ) -> SkeletonState: ...
    def set_visible(self, visible: bool) -> None: ...
    def set_pickable(self, pickable: bool) -> None: ...
    def set_color(self, color: Vec4 | npt.ArrayLike) -> None: ...
    def set_casts_shadow(self, enabled: bool = True) -> None: ...
    def remove(self) -> bool: ...
    def motion(self) -> SkeletonMotion: ...
    def num_meshes(self) -> int: ...
