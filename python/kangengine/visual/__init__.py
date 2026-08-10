"""Visual API surface.

Static visual asset/instance wrappers live directly under ``ke.visual``.
Simulation-state visual sync lives under ``ke.visual.sim``.
"""

from __future__ import annotations

from .articulation import ArticulationVisual, ArticulationVisualAsset
from .articulated_surface import ArticulatedSurface, ArticulatedSurfaceAsset
from .deformable import DeformableSurface, SkinnedSurface, SkinnedSurfaceAsset
from .skin import SkinVisual
from .skeletal import SkeletalVisual, SkeletalVisualConfig
from . import sim

__all__ = [
    "ArticulationVisual",
    "ArticulationVisualAsset",
    "ArticulatedSurface",
    "ArticulatedSurfaceAsset",
    "DeformableSurface",
    "SkinVisual",
    "SkinnedSurface",
    "SkinnedSurfaceAsset",
    "SkeletalVisual",
    "SkeletalVisualConfig",
    "sim",
]
