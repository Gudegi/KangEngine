"""Renderer material API.

Materials describe surface appearance and shader parameter sets.  They are
Python facades over native material objects and can be passed anywhere the
native object is accepted.
"""

from __future__ import annotations

from typing import Any

from .._core import _ke
from .._public import NativeWrapper, set_public_module, unwrap_native


PhongMaterialType = set_public_module(_ke.PhongMaterialType, __package__)
PBRMaterialType = set_public_module(_ke.PBRMaterialType, __package__)

NativeMaterial = set_public_module(_ke.Material, __package__)
NativeVertexColorMaterial = set_public_module(_ke.VertexColorMaterial, __package__)
NativePhongMaterial = set_public_module(_ke.PhongMaterial, __package__)
NativePBRMaterial = set_public_module(_ke.PBRMaterial, __package__)


class Material(NativeWrapper):
    """Base wrapper for renderer materials."""

    def __init__(self, native: Any | None = None, *, shader=None):
        if native is None:
            raise TypeError(
                "ke.material.Material wraps an existing native material; "
                "construct VertexColorMaterial, PhongMaterial, or PBRMaterial instead."
            )
        super().__init__(native)
        if shader is not None:
            self.set_shader(shader)

    def set_shader(self, shader):
        self._native.set_shader(unwrap_native(shader))
        return self

    def get_shader(self):
        return self._native.get_shader()

    @property
    def shader(self):
        """Shader used by this material."""
        return self.get_shader()

    @shader.setter
    def shader(self, value) -> None:
        self.set_shader(value)


class VertexColorMaterial(Material):
    """Material for vertex/display-color shaders."""

    def __init__(self, shader=None, native: Any | None = None):
        if native is None:
            native = (
                _ke.VertexColorMaterial()
                if shader is None
                else _ke.VertexColorMaterial(unwrap_native(shader))
            )
        super().__init__(native)


class PhongMaterial(Material):
    """Phong material wrapper with optional preset initialization."""

    def __init__(self, shader=None, preset=None, native: Any | None = None):
        if native is None:
            native = (
                _ke.PhongMaterial()
                if shader is None
                else _ke.PhongMaterial(unwrap_native(shader))
            )
        super().__init__(native)
        if preset is not None:
            self.load_from_preset(preset)

    def load_from_preset(self, preset):
        self._native.load_from_preset(preset)
        return self

    @property
    def ambient(self):
        """Ambient RGB factor."""
        return self._native.ambient

    @ambient.setter
    def ambient(self, value) -> None:
        self._native.ambient = value

    @property
    def diffuse(self):
        """Diffuse RGB factor."""
        return self._native.diffuse

    @diffuse.setter
    def diffuse(self, value) -> None:
        self._native.diffuse = value

    @property
    def specular(self):
        """Specular RGB factor."""
        return self._native.specular

    @specular.setter
    def specular(self, value) -> None:
        self._native.specular = value

    @property
    def shininess(self) -> float:
        """Specular highlight exponent."""
        return float(self._native.shininess)

    @shininess.setter
    def shininess(self, value: float) -> None:
        self._native.shininess = float(value)

    @property
    def diffuse_map(self):
        """Optional diffuse texture."""
        return self._native.diffuse_map

    @diffuse_map.setter
    def diffuse_map(self, texture) -> None:
        self._native.diffuse_map = unwrap_native(texture)

    @property
    def specular_map(self):
        """Optional specular texture."""
        return self._native.specular_map

    @specular_map.setter
    def specular_map(self, texture) -> None:
        self._native.specular_map = unwrap_native(texture)

    @property
    def alpha_map(self):
        """Optional alpha mask texture, such as OBJ ``map_d``."""
        return self._native.alpha_map

    @alpha_map.setter
    def alpha_map(self, texture) -> None:
        self._native.alpha_map = unwrap_native(texture)

    @property
    def normal_map(self):
        """Optional tangent-space normal map."""
        return self._native.normal_map

    @normal_map.setter
    def normal_map(self, texture) -> None:
        self._native.normal_map = unwrap_native(texture)

    def set_ambient(self, ambient):
        self._native.set_ambient(ambient)
        return self

    def set_diffuse(self, diffuse):
        self._native.set_diffuse(diffuse)
        return self

    def set_specular(self, specular):
        self._native.set_specular(specular)
        return self

    def set_shininess(self, shininess: float):
        self._native.set_shininess(float(shininess))
        return self

    def set_diffuse_map(self, texture):
        self._native.set_diffuse_map(unwrap_native(texture))
        return self

    def set_specular_map(self, texture):
        self._native.set_specular_map(unwrap_native(texture))
        return self

    def set_alpha_map(self, texture):
        self._native.set_alpha_map(unwrap_native(texture))
        return self

    def set_normal_map(self, texture):
        self._native.set_normal_map(unwrap_native(texture))
        return self


class PBRMaterial(Material):
    """PBR material wrapper with optional preset initialization."""

    def __init__(self, shader=None, preset=None, native: Any | None = None):
        if native is None:
            native = (
                _ke.PBRMaterial()
                if shader is None
                else _ke.PBRMaterial(unwrap_native(shader))
            )
        super().__init__(native)
        if preset is not None:
            self.load_from_preset(preset)

    def load_from_preset(self, preset):
        self._native.load_from_preset(preset)
        return self

    @property
    def base_color(self):
        """Base color factor as RGBA."""
        return self._native.base_color

    @base_color.setter
    def base_color(self, value) -> None:
        self._native.base_color = value

    @property
    def metallic(self) -> float:
        """Metallic factor."""
        return float(self._native.metallic)

    @metallic.setter
    def metallic(self, value: float) -> None:
        self._native.metallic = float(value)

    @property
    def roughness(self) -> float:
        """Roughness factor."""
        return float(self._native.roughness)

    @roughness.setter
    def roughness(self, value: float) -> None:
        self._native.roughness = float(value)

    @property
    def emissive_color(self):
        """Emissive RGB color factor."""
        return self._native.emissive_color

    @emissive_color.setter
    def emissive_color(self, value) -> None:
        self._native.emissive_color = value

    @property
    def emissive_strength(self) -> float:
        """Multiplier for emissive_color."""
        return float(self._native.emissive_strength)

    @emissive_strength.setter
    def emissive_strength(self, value: float) -> None:
        self._native.emissive_strength = float(value)

    @property
    def base_color_texture(self):
        return self._native.base_color_texture

    @base_color_texture.setter
    def base_color_texture(self, texture) -> None:
        self._native.base_color_texture = unwrap_native(texture)

    @property
    def normal_texture(self):
        return self._native.normal_texture

    @normal_texture.setter
    def normal_texture(self, texture) -> None:
        self._native.normal_texture = unwrap_native(texture)

    @property
    def metallic_roughness_texture(self):
        return self._native.metallic_roughness_texture

    @metallic_roughness_texture.setter
    def metallic_roughness_texture(self, texture) -> None:
        self._native.metallic_roughness_texture = unwrap_native(texture)

    @property
    def metallic_texture(self):
        return self._native.metallic_texture

    @metallic_texture.setter
    def metallic_texture(self, texture) -> None:
        self._native.metallic_texture = unwrap_native(texture)

    @property
    def roughness_texture(self):
        return self._native.roughness_texture

    @roughness_texture.setter
    def roughness_texture(self, texture) -> None:
        self._native.roughness_texture = unwrap_native(texture)

    @property
    def ao_texture(self):
        return self._native.ao_texture

    @ao_texture.setter
    def ao_texture(self, texture) -> None:
        self._native.ao_texture = unwrap_native(texture)

    @property
    def orm_texture(self):
        return self._native.orm_texture

    @orm_texture.setter
    def orm_texture(self, texture) -> None:
        self._native.orm_texture = unwrap_native(texture)

    @property
    def emissive_texture(self):
        return self._native.emissive_texture

    @emissive_texture.setter
    def emissive_texture(self, texture) -> None:
        self._native.emissive_texture = unwrap_native(texture)

    def set_base_color(self, base_color):
        self._native.set_base_color(base_color)
        return self

    def set_metallic(self, metallic: float):
        self._native.set_metallic(float(metallic))
        return self

    def set_roughness(self, roughness: float):
        self._native.set_roughness(float(roughness))
        return self

    def set_emissive_color(self, emissive_color):
        self._native.set_emissive_color(emissive_color)
        return self

    def set_emissive_strength(self, emissive_strength: float):
        self._native.set_emissive_strength(float(emissive_strength))
        return self

    def set_base_color_texture(self, texture):
        self._native.set_base_color_texture(unwrap_native(texture))
        return self

    def set_normal_texture(self, texture):
        self._native.set_normal_texture(unwrap_native(texture))
        return self

    def set_metallic_roughness_texture(self, texture):
        self._native.set_metallic_roughness_texture(unwrap_native(texture))
        return self

    def set_metallic_texture(self, texture):
        self._native.set_metallic_texture(unwrap_native(texture))
        return self

    def set_roughness_texture(self, texture):
        self._native.set_roughness_texture(unwrap_native(texture))
        return self

    def set_ao_texture(self, texture):
        self._native.set_ao_texture(unwrap_native(texture))
        return self

    def set_orm_texture(self, texture):
        self._native.set_orm_texture(unwrap_native(texture))
        return self

    def set_emissive_texture(self, texture):
        self._native.set_emissive_texture(unwrap_native(texture))
        return self


__all__ = [
    "PhongMaterialType",
    "PBRMaterialType",
    "NativeMaterial",
    "NativeVertexColorMaterial",
    "NativePhongMaterial",
    "NativePBRMaterial",
    "Material",
    "VertexColorMaterial",
    "PhongMaterial",
    "PBRMaterial",
    "unwrap_native",
]
