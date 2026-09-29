"""Low-level physics configuration, worlds, objects, and GPU synchronization."""

from .._public import set_public_module
from . import wrappers as _wrappers
from .wrappers import *
from .d6_batch import D6Batch
from .articulation_builder import ArticulationBuilder

__all__ = [*_wrappers.__all__, "D6Batch", "ArticulationBuilder"]

for _name in __all__:
    set_public_module(globals()[_name], __name__)

for _type in (
    ArticulationBuilder.FixedJoint,
    ArticulationBuilder.RevoluteJoint,
    ArticulationBuilder.PrismaticJoint,
    ArticulationBuilder.SphericalJoint,
    ArticulationBuilder.BoxShape,
    ArticulationBuilder.SphereShape,
):
    set_public_module(_type, __name__)

del _name, _type
