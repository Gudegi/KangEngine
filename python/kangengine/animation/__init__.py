"""Animation, skeleton, skinning, and character bridge APIs."""

from __future__ import annotations

from .._core import _ke
from .._public import export_public_module
from .retarget_batch import RetargetBatchProcessor, RetargetBatchResult
from .coordinates import (
    CoordinateSystem,
    convert_motion_coordinates,
    convert_skeleton_coordinates,
    convert_state_coordinates,
)
from . import IK as IK, filter as filter
from .retarget import RetargetConfig, Retargeter, retarget_motion
from .transform import transform_motion

__all__ = export_public_module(_ke.animation, globals())
__all__ += [
    "CoordinateSystem",
    "RetargetConfig",
    "RetargetBatchProcessor",
    "RetargetBatchResult",
    "Retargeter",
    "convert_motion_coordinates",
    "convert_skeleton_coordinates",
    "convert_state_coordinates",
    "filter",
    "IK",
    "retarget_motion",
    "transform_motion",
]

# Visual objects are implemented in the native animation binding for now, but
# the public Python surface owns them under ``kangengine.visual``.
for _name in (
    "ArticulationVisual",
    "ArticulationVisualAsset",
    "SkeletalVisual",
    "SkeletalVisualConfig",
    "solve_full_body_ik",
    "solve_full_body_ik_batch",
):
    globals().pop(_name, None)
    if _name in __all__:
        __all__.remove(_name)

del _name
