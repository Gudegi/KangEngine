"""Visual API surface.

Visual asset/instance wrappers live directly under ``ke.visual``.
Simulation-state visual sync lives under ``ke.visual.sim``.
"""

from __future__ import annotations

from .articulation_visual import ArticulationVisual, ArticulationVisualAsset
from .articulated_surface import ArticulatedSurface, ArticulatedSurfaceAsset
from .deformable import DeformableSurface, SkinnedSurface, SkinnedSurfaceAsset
from .rigid_visual import RigidVisual
from .skin import SkinVisual
from .skeletal import SkeletalVisual, SkeletalVisualConfig
from . import sim

__all__ = [
    "ArticulationVisual",
    "ArticulationVisualAsset",
    "ArticulatedSurface",
    "ArticulatedSurfaceAsset",
    "DeformableSurface",
    "RigidVisual",
    "SkinVisual",
    "SkinnedSurface",
    "SkinnedSurfaceAsset",
    "SkeletalVisual",
    "SkeletalVisualConfig",
    "sim",
]
