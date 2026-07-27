"""Low-level rendering API exposed directly from the native bindings.

General scene workflows should prefer kangengine.App,
``app.scene``, and renderable views. This package owns the stable public paths
for lower-level renderer, graphics-device, shader, texture, and external-buffer
APIs; the runtime objects themselves remain pybind types.
"""

from __future__ import annotations

from .._core import _ke
from .._public import set_public_module


BackendType = set_public_module(_ke.BackendType, __name__)
TextureRole = set_public_module(_ke.TextureRole, __name__)
AlphaMode = set_public_module(_ke.AlphaMode, __name__)
ToneMapMode = set_public_module(_ke.ToneMapMode, __name__)
TransformSource = set_public_module(_ke.TransformSource, __name__)
TextAlignment = set_public_module(_ke.TextAlignment, __name__)
TextDepthMode = set_public_module(_ke.TextDepthMode, __name__)
ScreenAnchor = set_public_module(_ke.ScreenAnchor, __name__)
ExternalBufferFormat = set_public_module(_ke.ExternalBufferFormat, __name__)
ExternalSyncPolicy = set_public_module(_ke.ExternalSyncPolicy, __name__)
ExternalBufferDesc = set_public_module(_ke.ExternalBufferDesc, __name__)

TextureWrap = set_public_module(_ke.TextureWrap, __name__)
TextureFilter = set_public_module(_ke.TextureFilter, __name__)
SamplerDesc = set_public_module(_ke.SamplerDesc, __name__)

Renderer = set_public_module(_ke.Renderer, __name__)
GraphicsDevice = set_public_module(_ke.GraphicsDevice, __name__)
Shader = set_public_module(_ke.Shader, __name__)
Texture = set_public_module(_ke.Texture, __name__)


__all__ = [
    "AlphaMode",
    "BackendType",
    "ExternalBufferDesc",
    "ExternalBufferFormat",
    "ExternalSyncPolicy",
    "GraphicsDevice",
    "Renderer",
    "SamplerDesc",
    "ScreenAnchor",
    "Shader",
    "Texture",
    "TextureFilter",
    "TextureRole",
    "TextureWrap",
    "TextAlignment",
    "TextDepthMode",
    "ToneMapMode",
    "TransformSource",
]
