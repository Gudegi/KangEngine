"""Python API surface for KangEngine."""

import os as _os
from pathlib import Path as _Path

_assets_dir = _Path(__file__).resolve().parent / "assets"
if _assets_dir.exists():
    _os.environ.setdefault("KANGENGINE_ASSETS_ROOT", str(_assets_dir))

from . import _kangengine as _ke
from .app import App, NativeApp

# Core engine API
BackendType = _ke.BackendType
GraphicsDevice = _ke.GraphicsDevice
Shader = _ke.Shader
Texture = _ke.Texture
UpAxis = _ke.UpAxis

# GLM-style math types and helpers exposed by the C++ extension.
vec3 = _ke.vec3
vec4 = _ke.vec4
quat = _ke.quat
mat3 = _ke.mat3
mat4 = _ke.mat4
translate = _ke.translate
scale = _ke.scale

# Bound C++ submodules.
scene = _ke.scene
animation = _ke.animation
imgui = _ke.imgui
keys = _ke.keys

# Enum values exported by pybind11's export_values().
Y = _ke.Y
Z = _ke.Z
OpenGL = _ke.OpenGL
Vulkan = _ke.Vulkan
WebGPU = _ke.WebGPU

__all__ = [
    "App",
    "NativeApp",
    "BackendType",
    "GraphicsDevice",
    "Shader",
    "Texture",
    "UpAxis",
    "vec3",
    "vec4",
    "quat",
    "mat3",
    "mat4",
    "translate",
    "scale",
    "scene",
    "animation",
    "imgui",
    "keys",
    "Y",
    "Z",
    "OpenGL",
    "Vulkan",
    "WebGPU",
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
