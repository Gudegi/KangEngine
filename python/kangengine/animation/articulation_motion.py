"""Native articulation motion coordinate metadata."""

from .._core import _ke
from .._public import set_public_module

ArticulationMotion = _ke.animation.ArticulationMotion
ArticulationMappingResult = _ke.animation.ArticulationMappingResult
ArticulationMotionMappingResult = _ke.animation.ArticulationMotionMappingResult
ArticulationMotionMapper = _ke.animation.ArticulationMotionMapper
ArticulationCoordinateBlock = _ke.animation.ArticulationCoordinateBlock
ArticulationCoordinateLayout = _ke.animation.ArticulationCoordinateLayout
ArticulationCoordinateType = _ke.animation.ArticulationCoordinateType

__all__ = [
    "ArticulationMotion",
    "ArticulationMappingResult",
    "ArticulationMotionMappingResult",
    "ArticulationMotionMapper",
    "ArticulationCoordinateBlock",
    "ArticulationCoordinateLayout",
    "ArticulationCoordinateType",
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
    set_public_module(_type, "kangengine.animation")

del _type
