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
    animation,
    asset,
    character,
    geometry,
    input,
    material,
    physics,
    render,
    scene,
    terrain,
    visual,
)
from .app import (
    App,
    DebugGeometry,
    DebugOverlay,
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
    "KangEngineEngine": (".mimickit_engine", "KangEngineEngine"),
    "build_mimickit_engine": (".mimickit_engine", "build_engine"),
    "install_mimickit_engine_builder": (
        ".mimickit_engine",
        "install_mimickit_engine_builder",
    ),
}
_LAZY_MODULES = {
    "motion_module": ".motion_module",
    "sim": ".sim",
}

if _TYPE_CHECKING:
    from . import motion_module as motion_module
    from . import sim as sim
    from .mimickit_engine import (
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
Camera = _set_public_module(_ke.Camera, __name__)
UpAxis = _set_public_module(_ke.UpAxis, __name__)
InteractionMode = _set_public_module(_ke.InteractionMode, __name__)
RayPickResult = _set_public_module(_ke.RayPickResult, __name__)
DirectionalLight = _set_public_module(_ke.DirectionalLight, __name__)
PointLight = _set_public_module(_ke.PointLight, __name__)
SpotLight = _set_public_module(_ke.SpotLight, __name__)
ColorLibrary = _set_public_module(_ke.ColorLibrary, __name__)
ColorType = _set_public_module(_ke.ColorType, __name__)
Color = _set_public_module(_ke.Color, __name__)
MotionSequencerPanel = _set_public_module(_ke.MotionSequencerPanel, __name__)
# GLM-style math types and helpers exposed by the C++ extension.
vec3 = _set_public_module(_ke.vec3, __name__)
vec2 = _set_public_module(_ke.vec2, __name__)
vec4 = _set_public_module(_ke.vec4, __name__)
quat = _set_public_module(_ke.quat, __name__)
mat3 = _set_public_module(_ke.mat3, __name__)
mat4 = _set_public_module(_ke.mat4, __name__)
translate = _set_public_module(_ke.translate, __name__)
scale = _set_public_module(_ke.scale, __name__)

# Bound C++ submodules exposed through Python wrapper modules.
imgui = _ke.imgui
keys = _ke.keys

# Enum values exported by pybind11's export_values().
X = _ke.X
Y = _ke.Y
Z = _ke.Z
OpenGL = _ke.OpenGL
Vulkan = _ke.Vulkan
WebGPU = _ke.WebGPU

__all__ = [
    "App",
    "DebugGeometry",
    "DebugOverlay",
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
    "visual",
    "terrain",
    "KangEngineEngine",
    "build_mimickit_engine",
    "install_mimickit_engine_builder",
    "Camera",
    "UpAxis",
    "InteractionMode",
    "RayPickResult",
    "DirectionalLight",
    "PointLight",
    "SpotLight",
    "vec3",
    "vec2",
    "vec4",
    "quat",
    "mat3",
    "mat4",
    "translate",
    "scale",
    "scene",
    "asset",
    "animation",
    "geometry",
    "input",
    "material",
    "motion_module",
    "physics",
    "render",
    "sim",
    "imgui",
    "keys",
    "X",
    "Y",
    "Z",
    "OpenGL",
    "Vulkan",
    "WebGPU",
    "ColorLibrary",
    "ColorType",
    "Color",
    "MotionSequencerPanel",
    "preset_rgba",
]

del _assets_dir, _Path, _os, _set_public_module, _ke
