"""Python API surface for KangEngine."""

import os as _os
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING

_assets_dir = _Path(__file__).resolve().parent / "assets"
if _assets_dir.exists():
    _os.environ.setdefault("KANGENGINE_ASSETS_ROOT", str(_assets_dir))

from ._core import _ke
from ._public import set_public_module as _set_public_module
from . import (
    adapters,
    animation,
    asset,
    geometry,
    input,
    material,
    physics,
    recording,
    render,
    scene,
    terrain,
    visual,
)
from .app import (
    App,
    DebugGeometry,
    DebugOverlay,
    ObjImportView,
    RenderablePrimView,
    ScreenText,
    SceneContext,
    WorldText,
)
from .utils import (
    COMMON,
    DEFAULT_PROFILE_ORDER,
    GENO,
    JOINT_PROFILES,
    JointMapper,
    JointSemantic,
    KW,
    KW5,
    MIXAMO,
    preset_rgba,
)

# TODO: Keep Torch-heavy modules lazy until CUDA context interop is explicit.
# This avoids accidental Torch CUDA initialization before PhysX GPU setup.
_LAZY_IMPORTS = {
    "SimulationPacer": (".sim.run_mode", "SimulationPacer"),
    "SimulationRunConfig": (".sim.run_mode", "SimulationRunConfig"),
    "SimulationRunMode": (".sim.run_mode", "SimulationRunMode"),
    "SimulationTimingConfig": (".sim.timing", "SimulationTimingConfig"),
    "KangEngineEngine": (".adapters.mimickit", "KangEngineEngine"),
    "build_mimickit_engine": (".adapters.mimickit", "build_engine"),
    "install_mimickit_engine_builder": (
        ".adapters.mimickit",
        "install_mimickit_engine_builder",
    ),
}
_LAZY_MODULES = {
    "exports": ".exports",
    "motion_module": ".motion_module",
    "sim": ".sim",
}

if _TYPE_CHECKING:
    from . import exports as exports
    from . import motion_module as motion_module
    from . import sim as sim
    from .sim.run_mode import SimulationPacer, SimulationRunConfig, SimulationRunMode
    from .sim.timing import SimulationTimingConfig
    from .adapters.mimickit import (
        KangEngineEngine,
        build_engine as build_mimickit_engine,
        install_mimickit_engine_builder,
    )


def __getattr__(name):
    from importlib import import_module

    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        value = import_module(module_name, __name__)
        globals()[name] = value
        return value

    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


# Core engine API. Keep top-level exports focused on common viewer/app usage;
# heavier simulation and MimicKit APIs stay lazy via _LAZY_IMPORTS.
UpAxis = _set_public_module(_ke.UpAxis, __name__)
InteractionMode = _set_public_module(_ke.InteractionMode, __name__)
RayPickResult = _set_public_module(_ke.RayPickResult, __name__)
ColorLibrary = _set_public_module(_ke.ColorLibrary, __name__)
ColorType = _set_public_module(_ke.ColorType, __name__)
Color = _set_public_module(_ke.Color, __name__)
MotionSequencerPanel = _set_public_module(_ke.MotionSequencerPanel, __name__)
FixedStepClock = _set_public_module(_ke.FixedStepClock, __name__)
# GLM-style math types and helpers exposed by the C++ extension.
Vec3 = _set_public_module(_ke.Vec3, __name__)
Vec2 = _set_public_module(_ke.Vec2, __name__)
Vec4 = _set_public_module(_ke.Vec4, __name__)
Quat = _set_public_module(_ke.Quat, __name__)
Mat3 = _set_public_module(_ke.Mat3, __name__)
Mat4 = _set_public_module(_ke.Mat4, __name__)
translate = _set_public_module(_ke.translate, __name__)
scale = _set_public_module(_ke.scale, __name__)

# Bound C++ submodules exposed through Python wrapper modules.
imgui = _ke.imgui
keys = _ke.keys

__all__ = [
    "App",
    "adapters",
    "DebugGeometry",
    "DebugOverlay",
    "ObjImportView",
    "RenderablePrimView",
    "ScreenText",
    "SceneContext",
    "WorldText",
    "JointMapper",
    "JointSemantic",
    "COMMON",
    "DEFAULT_PROFILE_ORDER",
    "GENO",
    "JOINT_PROFILES",
    "KW",
    "KW5",
    "MIXAMO",
    "recording",
    "visual",
    "terrain",
    "KangEngineEngine",
    "build_mimickit_engine",
    "install_mimickit_engine_builder",
    "UpAxis",
    "InteractionMode",
    "RayPickResult",
    "Vec3",
    "Vec2",
    "Vec4",
    "Quat",
    "Mat3",
    "Mat4",
    "translate",
    "scale",
    "scene",
    "asset",
    "animation",
    "exports",
    "geometry",
    "input",
    "material",
    "motion_module",
    "physics",
    "render",
    "sim",
    "imgui",
    "keys",
    "ColorLibrary",
    "ColorType",
    "Color",
    "MotionSequencerPanel",
    "FixedStepClock",
    "SimulationPacer",
    "SimulationRunConfig",
    "SimulationRunMode",
    "SimulationTimingConfig",
    "preset_rgba",
]

del _assets_dir, _Path, _os, _set_public_module, _ke
