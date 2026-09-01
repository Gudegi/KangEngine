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
from .articulation_io import (
    load_articulation_motion_npz,
    save_articulation_motion_npz,
)
from .coordinates import (
    CoordinateSystem,
    convert_motion_coordinates,
    convert_skeleton_coordinates,
    convert_state_coordinates,
)
from . import IK as IK, datasets as datasets, filter as filter
from .retarget import (
    AngleRetargetConfig,
    AngleRetargetProcessor,
    AngleRetargetResult,
    AngleRetargeter,
    AngleTargetProfile,
    AuxiliaryRetargetPoint,
    IKEffectorMapping,
    IKRetargetConfig,
    IKRetargetProcessor,
    IKRetargetResult,
    IKTargetProfile,
    MotionSourceProfile,
    retarget_angle_motion,
    scale_skeleton_motion,
)
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
    "AuxiliaryRetargetPoint",
    "CoordinateSystem",
    "AngleRetargetConfig",
    "AngleRetargetProcessor",
    "AngleRetargetResult",
    "AngleRetargeter",
    "convert_motion_coordinates",
    "convert_skeleton_coordinates",
    "convert_state_coordinates",
    "datasets",
    "filter",
    "IK",
    "IKEffectorMapping",
    "MotionSourceProfile",
    "AngleTargetProfile",
    "IKRetargetConfig",
    "IKRetargetProcessor",
    "IKRetargetResult",
    "IKTargetProfile",
    "MotionKinematics",
    "MotionLibrary",
    "MotionSample",
    "load_articulation_motion_npz",
    "retarget_angle_motion",
    "save_articulation_motion_npz",
    "scale_skeleton_motion",
    "transform_motion",
]

for _type in (
    AuxiliaryRetargetPoint,
    ArticulationMotion,
    ArticulationMappingResult,
    ArticulationMotionMappingResult,
    ArticulationMotionMapper,
    ArticulationCoordinateBlock,
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
    IKEffectorMapping,
    MotionSourceProfile,
    AngleTargetProfile,
    IKRetargetConfig,
    IKTargetProfile,
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
