"""Python-friendly material authoring facades."""

from __future__ import annotations

from .._public import set_public_module
from .materials import (
    Material,
    NativeMaterial,
    NativePBRMaterial,
    NativePhongMaterial,
    NativeVertexColorMaterial,
    PBRMaterial,
    PBRMaterialType,
    PhongMaterial,
    PhongMaterialType,
    VertexColorMaterial,
    VertexColorStyle,
    unwrap_native,
)

for _type in (Material, VertexColorMaterial, PhongMaterial, PBRMaterial):
    set_public_module(_type, __name__)

del _type

__all__ = [
    "Material",
    "NativeMaterial",
    "NativePBRMaterial",
    "NativePhongMaterial",
    "NativeVertexColorMaterial",
    "PBRMaterial",
    "PBRMaterialType",
    "PhongMaterial",
    "PhongMaterialType",
    "VertexColorMaterial",
    "VertexColorStyle",
    "unwrap_native",
]
