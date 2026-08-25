"""Animation, skeleton, skinning, and character bridge APIs."""

from __future__ import annotations

from importlib import import_module

from .._core import _ke
from .._public import export_public_module, set_public_module
from .articulation_motion import (
    ArticulationMotion,
    ArticulationMappingResult,
    ArticulationMotionMappingResult,
    ArticulationMotionMapper,
    ArticulationCoordinateBlock,
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
)
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
    "ArticulationMotion",
    "ArticulationMappingResult",
    "ArticulationMotionMappingResult",
    "ArticulationMotionMapper",
    "ArticulationCoordinateBlock",
    "ArticulationCoordinateLayout",
    "ArticulationCoordinateType",
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
    "MotionKinematics",
    "MotionLibrary",
    "MotionSample",
    "retarget_motion",
    "transform_motion",
]

for _type in (
    ArticulationMotion,
    ArticulationMappingResult,
    ArticulationMotionMappingResult,
    ArticulationMotionMapper,
    ArticulationCoordinateBlock,
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
):
    set_public_module(_type, __name__)

_LAZY_IMPORTS = {
    "MotionKinematics": (".motion_library", "MotionKinematics"),
    "MotionLibrary": (".motion_library", "MotionLibrary"),
    "MotionSample": (".motion_library", "MotionSample"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attr_name)
    set_public_module(value, __name__)
    globals()[name] = value
    return value

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

del _name, _type
