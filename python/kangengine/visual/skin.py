"""Skinned mesh visual wrappers."""

from __future__ import annotations

from .._core import _ke
from .._public import set_public_module

SkinVisual = set_public_module(_ke.SkinVisual, __name__)

__all__ = ["SkinVisual"]
