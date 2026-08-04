"""Low-level rendering API exposed directly from the native bindings.

General scene workflows should prefer kangengine.App,
``app.scene``, and renderable views. This package owns the stable public paths
for lower-level renderer, graphics-device, texture, and external-buffer
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
ShaderType = set_public_module(_ke.ShaderType, __name__)
ShaderStage = set_public_module(_ke.ShaderStage, __name__)
ShaderDesc = set_public_module(_ke.ShaderDesc, __name__)
BufferUsage = set_public_module(_ke.BufferUsage, __name__)
VertexFormat = set_public_module(_ke.VertexFormat, __name__)
VertexStepMode = set_public_module(_ke.VertexStepMode, __name__)
VertexAttributeDesc = set_public_module(_ke.VertexAttributeDesc, __name__)
VertexBufferLayout = set_public_module(_ke.VertexBufferLayout, __name__)
PrimitiveTopology = set_public_module(_ke.PrimitiveTopology, __name__)
CullMode = set_public_module(_ke.CullMode, __name__)
CompareFunction = set_public_module(_ke.CompareFunction, __name__)
IndexFormat = set_public_module(_ke.IndexFormat, __name__)
SceneHookBlendMode = set_public_module(_ke.SceneHookBlendMode, __name__)
SceneHookPipelineDesc = set_public_module(_ke.SceneHookPipelineDesc, __name__)
RenderHookPhase = set_public_module(_ke.RenderHookPhase, __name__)
RenderHookContext = set_public_module(_ke.RenderHookContext, __name__)

Renderer = set_public_module(_ke.Renderer, __name__)
GraphicsDevice = set_public_module(_ke.GraphicsDevice, __name__)
Texture = set_public_module(_ke.Texture, __name__)
Buffer = set_public_module(_ke.Buffer, __name__)
GraphicsPipeline = set_public_module(_ke.GraphicsPipeline, __name__)


__all__ = [
    "AlphaMode",
    "BackendType",
    "Buffer",
    "BufferUsage",
    "CompareFunction",
    "CullMode",
    "ExternalBufferDesc",
    "ExternalBufferFormat",
    "ExternalSyncPolicy",
    "GraphicsDevice",
    "GraphicsPipeline",
    "IndexFormat",
    "PrimitiveTopology",
    "RenderHookContext",
    "RenderHookPhase",
    "Renderer",
    "SamplerDesc",
    "SceneHookBlendMode",
    "SceneHookPipelineDesc",
    "ScreenAnchor",
    "Texture",
    "TextureFilter",
    "TextureRole",
    "TextureWrap",
    "ShaderDesc",
    "ShaderStage",
    "ShaderType",
    "TextAlignment",
    "TextDepthMode",
    "ToneMapMode",
    "TransformSource",
    "VertexAttributeDesc",
    "VertexBufferLayout",
    "VertexFormat",
    "VertexStepMode",
]
