"""Python API surface for KangEngine."""

import os as _os
from pathlib import Path as _Path

_assets_dir = _Path(__file__).resolve().parent / "assets"
if _assets_dir.exists():
    _os.environ.setdefault("KANGENGINE_ASSETS_ROOT", str(_assets_dir))

from ._core import _ke
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

# Core engine API
BackendType = _ke.BackendType
GraphicsDevice = _ke.GraphicsDevice
Shader = _ke.Shader
Texture = _ke.Texture
Camera = _ke.Camera
UpAxis = _ke.UpAxis
TransformSource = _ke.TransformSource
InteractionMode = _ke.InteractionMode
RayPickResult = _ke.RayPickResult
ColorLibrary = _ke.ColorLibrary
ColorType = _ke.ColorType
Color = _ke.Color
PhongMaterial = _ke.PhongMaterial
SkinnedCharacterBridge = _ke.SkinnedCharacterBridge
SkeletonVisualBridge = _ke.animation.SkeletonVisualBridge
SkeletonVisualConfig = _ke.animation.SkeletonVisualConfig
MotionSequencerPanel = _ke.MotionSequencerPanel

# GLM-style math types and helpers exposed by the C++ extension.
vec3 = _ke.vec3
vec2 = _ke.vec2
vec4 = _ke.vec4
quat = _ke.quat
mat3 = _ke.mat3
mat4 = _ke.mat4
translate = _ke.translate
scale = _ke.scale

# Bound C++ submodules.
scene = _ke.scene
asset = _ke.asset
animation = _ke.animation
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
    "PhongMaterial",
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
    if hasattr(_ke, _name):
        globals()[_name] = getattr(_ke, _name)
        __all__.append(_name)

del _assets_dir, _name, _OPTIONAL_EXPORTS, _Path, _os, _ke
