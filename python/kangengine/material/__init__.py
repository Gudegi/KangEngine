"""Python-friendly material authoring facades."""

from __future__ import annotations

from .._public import set_public_module
from .materials import (
    ColorMaterial,
    ColorStyle,
    Material,
    NativeColorMaterial,
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

for _type in (Material, ColorMaterial, PhongMaterial, PBRMaterial):
    set_public_module(_type, __name__)

del _type

__all__ = [
    "ColorMaterial",
    "ColorStyle",
    "Material",
    "NativeColorMaterial",
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
