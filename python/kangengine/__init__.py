"""Python API surface for KangEngine."""

import os as _os
from pathlib import Path as _Path

_assets_dir = _Path(__file__).resolve().parent / "assets"
if _assets_dir.exists():
    _os.environ.setdefault("KANGENGINE_ASSETS_ROOT", str(_assets_dir))

from ._core import _ke
from .app import App, NativeApp
from .sim import ControlMode, KangSimWorld
from .visual import KangWorldVisualBridge
from .mimickit_engine import (
    KangEngineEngine,
    build_engine as build_mimickit_engine,
    install_mimickit_engine_builder,
)

# Core engine API
BackendType = _ke.BackendType
GraphicsDevice = _ke.GraphicsDevice
Shader = _ke.Shader
Texture = _ke.Texture
Camera = _ke.Camera
UpAxis = _ke.UpAxis
RenderTrack = _ke.RenderTrack
ColorLibrary = _ke.ColorLibrary
ColorType = _ke.ColorType
Color = _ke.Color
PhongMaterial = _ke.PhongMaterial
SkinnedCharacterBridge = _ke.SkinnedCharacterBridge

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
    "ControlMode",
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
    "RenderTrack",
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
