"""Low-level physics world and articulation APIs."""
from __future__ import annotations

from ._core import _ke
from ._public import set_public_module

_PHYSICS_NAMES = [
    "PhysicsGpuDynamicsConfig",
    "PhysicsConfig",
    "PhysicsMaterialDesc",
    "CollisionMaterialOverride",
    "mjcf_friction_to_physx",
    "ContactPoint",
    "RigidDynamic",
    "PhysicsWorld",
    "ArticulationConfig",
    "Articulation",
    "PhysicsBridge",
]

_TOPLEVEL_SIM_NAMES = [
    "SimMemoryType",
    "SimDType",
    "SimLifetimePolicy",
    "GpuArrayView",
    "SimModel",
    "SimState",
    "SimVisualBatch",
]

_PHYSICS_GPU_NAMES = [
    "GpuPhysicsConfig",
    "PhysicsGpuStateViews",
    "PhysicsGpuSystem",
    "aggregate_contact_sensors_cuda",
]

__all__ = []
_value = None
_physics = getattr(_ke, "physics", None)

for _name in _PHYSICS_NAMES:
    if _physics is not None and hasattr(_physics, _name):
        _value = set_public_module(getattr(_physics, _name), __name__)
        globals()[_name] = _value
        __all__.append(_name)

for _name in _TOPLEVEL_SIM_NAMES:
    if hasattr(_ke, _name) and _name not in globals():
        _value = set_public_module(getattr(_ke, _name), __name__)
        globals()[_name] = _value
        __all__.append(_name)

for _name in _PHYSICS_GPU_NAMES:
    if _physics is not None and hasattr(_physics, _name):
        _value = set_public_module(getattr(_physics, _name), __name__)
        globals()[_name] = _value
        __all__.append(_name)

del _name, _value, _PHYSICS_NAMES, _TOPLEVEL_SIM_NAMES, _PHYSICS_GPU_NAMES, _physics
