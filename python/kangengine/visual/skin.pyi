"""Integrated FBX skinned-character visualization."""

from __future__ import annotations

import numpy.typing as npt

from .. import Vec4
from ..animation import SkeletonMotion, SkeletonState
from ..app import App
from ..material import Material

class SkinVisual:
    """Integrated FBX motion-loading and skinning bridge."""

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
    ) -> SkinVisual: ...

    def apply_time(self, time: float, loop: bool = True) -> SkeletonState: ...
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
