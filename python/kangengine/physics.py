"""Low-level physics world and articulation APIs."""
from __future__ import annotations

from ._core import _ke
from ._public import set_public_module

_NAMES = [
    "PhysicsGpuDynamicsConfig",
    "PhysicsConfig",
    "ContactPoint",
    "RigidDynamic",
    "PhysicsWorld",
    "ArticulationConfig",
    "Articulation",
    "PhysicsBridge",
    "SimMemoryType",
    "SimDType",
    "SimLifetimePolicy",
    "GpuArrayView",
    "GpuPhysicsConfig",
    "PhysicsGpuStateViews",
    "PhysicsGpuSystem",
]

__all__ = []
_value = None

for _name in _NAMES:
    if hasattr(_ke, _name):
        _value = set_public_module(getattr(_ke, _name), __name__)
        globals()[_name] = _value
        __all__.append(_name)

del _name, _value, _NAMES
