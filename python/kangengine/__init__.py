"""Python API surface for KangEngine."""

import os as _os
from pathlib import Path as _Path

_assets_dir = _Path(__file__).resolve().parent / "assets"
if _assets_dir.exists():
    _os.environ.setdefault("KANGENGINE_ASSETS_ROOT", str(_assets_dir))

from ._core import _ke
from ._public import set_public_module as _set_public_module
from . import animation, asset, physics, scene
from .app import App, NativeApp
from .motion_editor import (
    MotionEditor,
    MotionPlayer,
    MotionSampleData,
    ContactData,
    MotionCameraFollower,
    RootTrajectoryData,
    TrackingData,
)
from .motion_modules import (
    MotionModule,
    ContactModule,
    RootTrajectoryModule,
    TargetModule,
    TrackingModule,
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
from .visual import KangWorldVisualBridge
# TODO: Keep Torch-heavy modules lazy until CUDA context interop is explicit.
# This avoids accidental Torch CUDA initialization before PhysX GPU setup.
_LAZY_IMPORTS = {
    "ControlMode": (".sim", "ControlMode"),
    "SimDevice": (".sim", "SimDevice"),
    "KangSimWorld": (".sim", "KangSimWorld"),
    "KangEngineEngine": (".mimickit_engine", "KangEngineEngine"),
    "build_mimickit_engine": (".mimickit_engine", "build_engine"),
    "install_mimickit_engine_builder": (
        ".mimickit_engine",
        "install_mimickit_engine_builder",
    ),
}


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

# Core engine API. Keep top-level exports focused on common viewer/app usage;
# heavier simulation and MimicKit APIs stay lazy via _LAZY_IMPORTS.
BackendType = _set_public_module(_ke.BackendType, __name__)
GraphicsDevice = _set_public_module(_ke.GraphicsDevice, __name__)
Shader = _set_public_module(_ke.Shader, __name__)
Texture = _set_public_module(_ke.Texture, __name__)
Camera = _set_public_module(_ke.Camera, __name__)
UpAxis = _set_public_module(_ke.UpAxis, __name__)
TransformSource = _set_public_module(_ke.TransformSource, __name__)
InteractionMode = _set_public_module(_ke.InteractionMode, __name__)
ToneMapMode = _set_public_module(_ke.ToneMapMode, __name__)
TextureRole = _set_public_module(_ke.TextureRole, __name__)
RayPickResult = _set_public_module(_ke.RayPickResult, __name__)
ColorLibrary = _set_public_module(_ke.ColorLibrary, __name__)
ColorType = _set_public_module(_ke.ColorType, __name__)
Color = _set_public_module(_ke.Color, __name__)
Material = _set_public_module(_ke.Material, __name__)
PhongMaterial = _set_public_module(_ke.PhongMaterial, __name__)
PBRMaterial = _set_public_module(_ke.PBRMaterial, __name__)
PBRMaterialType = _set_public_module(_ke.PBRMaterialType, __name__)
Renderer = _set_public_module(_ke.Renderer, __name__)
SkinnedCharacterBridge = _set_public_module(_ke.SkinnedCharacterBridge, __name__)
SkeletonVisualBridge = animation.SkeletonVisualBridge
SkeletonVisualConfig = animation.SkeletonVisualConfig
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
    "NativeApp",
    "MotionEditor",
    "MotionModule",
    "MotionPlayer",
    "MotionSampleData",
    "ContactModule",
    "ContactData",
    "MotionCameraFollower",
    "RootTrajectoryModule",
    "RootTrajectoryData",
    "TrackingData",
    "TrackingModule",
    "TargetModule",
    "JointMapper",
    "JointSemantic",
    "COMMON",
    "DEFAULT_PROFILE_ORDER",
    "GENO",
    "JOINT_PROFILES",
    "KW",
    "KW5",
    "MIXAMO",
    "ControlMode",
    "SimDevice",
    "KangSimWorld",
    "KangWorldVisualBridge",
    "KangEngineEngine",
    "build_mimickit_engine",
    "install_mimickit_engine_builder",
    "BackendType",
    "GraphicsDevice",
    "Shader",
    "Texture",
    "Camera",
    "UpAxis",
    "TransformSource",
    "InteractionMode",
    "ToneMapMode",
    "TextureRole",
    "RayPickResult",
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
    "physics",
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
    "Material",
    "PhongMaterial",
    "PBRMaterial",
    "PBRMaterialType",
    "Renderer",
    "SkinnedCharacterBridge",
    "SkeletonVisualBridge",
    "SkeletonVisualConfig",
    "MotionSequencerPanel",
    "preset_rgba",
]

_OPTIONAL_EXPORTS = [
    "PhysicsConfig",
    "PhysicsWorld",
    "ArticulationConfig",
    "Articulation",
    "PhysicsBridge",
]

for _name in _OPTIONAL_EXPORTS:
    if hasattr(physics, _name):
        globals()[_name] = getattr(physics, _name)
        __all__.append(_name)

del _assets_dir, _name, _OPTIONAL_EXPORTS, _Path, _os, _set_public_module, _ke
