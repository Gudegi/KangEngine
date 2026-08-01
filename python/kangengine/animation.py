"""Animation, skeleton, skinning, and character bridge APIs."""

from __future__ import annotations

from ._core import _ke
from ._public import export_public_module

__all__ = export_public_module(_ke.animation, globals())

# Visual objects are implemented in the native animation binding for now, but
# the public Python surface owns them under ``kangengine.visual``.
for _name in (
    "ArticulationVisual",
    "ArticulationVisualAsset",
    "SkeletalVisual",
    "SkeletalVisualConfig",
):
    globals().pop(_name, None)
    if _name in __all__:
        __all__.remove(_name)

del _name
