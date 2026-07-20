"""Skeleton line/joint visual wrappers."""

from __future__ import annotations

from .._core import _ke
from .._public import set_public_module

SkeletalVisualConfig = set_public_module(
    _ke.animation.SkeletalVisualConfig,
    __name__,
)
SkeletalVisual = set_public_module(_ke.animation.SkeletalVisual, __name__)

__all__ = [
    "SkeletalVisual",
    "SkeletalVisualConfig",
]
