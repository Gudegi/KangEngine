"""Articulated rigid-link visual asset wrappers."""

from __future__ import annotations

from .._core import _ke
from .._public import set_public_module

ArticulationVisual = set_public_module(
    _ke.animation.ArticulationVisual,
    __name__,
)
ArticulationVisualAsset = set_public_module(
    _ke.animation.ArticulationVisualAsset,
    __name__,
)

__all__ = [
    "ArticulationVisual",
    "ArticulationVisualAsset",
]
