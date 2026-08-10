"""Skeleton pose and motion visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from .. import Vec4
from ..animation import SkeletonMotion, SkeletonState
from ..app import App
from ..material import Material

_Color4 = Vec4 | Sequence[float]

class SkeletalVisualConfig:
    """Style settings for bone lines and joint markers."""

    bone_color: Vec4
    joint_color: Vec4
    bone_radius: float
    joint_radius: float
    segments: int
    visible: bool
    show_joints: bool

    def __init__(
        self,
        *,
        bone_color: _Color4 | None = None,
        joint_color: _Color4 | None = None,
        bone_radius: float = 0.006,
        joint_radius: float = 0.025,
        segments: int = 8,
        visible: bool = True,
        show_joints: bool = True,
    ) -> None:
        """Create skeleton visual settings from keyword fields."""
        ...

class SkeletalVisual:
    """Instanced line/point renderer for skeleton poses and motion clips."""

    def __init__(self) -> None: ...

    @overload
    @staticmethod
    def define(
        app: App,
        material: Material,
        base_path: str,
        state: SkeletonState,
        config: SkeletalVisualConfig = ...,
    ) -> SkeletalVisual: ...

    @overload
    @staticmethod
    def define(
        app: App,
        material: Material,
        base_path: str,
        motion: SkeletonMotion,
        time: float = 0.0,
        loop: bool = True,
        config: SkeletalVisualConfig = ...,
    ) -> SkeletalVisual: ...

    def apply_state(self, state: SkeletonState) -> None: ...
    def apply_motion(
        self, motion: SkeletonMotion, time: float, loop: bool = True
    ) -> None: ...
    def set_visible(self, visible: bool) -> None: ...
    def set_show_joints(self, show_joints: bool) -> None: ...
    def bone_handle(self) -> int: ...
    def joint_handle(self) -> int: ...
