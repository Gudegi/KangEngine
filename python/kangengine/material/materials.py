"""Renderer material API.

Materials describe surface appearance and shader parameter sets.  They are
Python facades over native material objects and can be passed anywhere the
native object is accepted.
"""

from __future__ import annotations

from typing import Any

from .._core import _ke
from .._public import NativeWrapper, set_public_module, unwrap_native
from ..render.texture_source import TextureSource


PhongMaterialType = set_public_module(_ke.PhongMaterialType, __package__)
PBRMaterialType = set_public_module(_ke.PBRMaterialType, __package__)
ColorStyle = set_public_module(_ke.ColorStyle, __package__)

NativeMaterial = set_public_module(_ke.Material, __package__)
NativeColorMaterial = set_public_module(_ke.ColorMaterial, __package__)
NativePhongMaterial = set_public_module(_ke.PhongMaterial, __package__)
NativePBRMaterial = set_public_module(_ke.PBRMaterial, __package__)


class Material(NativeWrapper):
    """Material parameters and texture references."""

    def __init__(self, native: Any | None = None):
        if native is None:
            raise TypeError(
                "ke.material.Material wraps an existing native material; "
                "construct ColorMaterial, PhongMaterial, or PBRMaterial instead."
            )
        super().__init__(native)
        object.__setattr__(self, "_texture_sources", {})
        object.__setattr__(self, "_resources", None)

    def _bind_resources(self, resources):
        for source in self._texture_sources.values():
            if source._resources is not resources:
                raise ValueError("Texture source belongs to a different App")
        object.__setattr__(self, "_resources", resources)

    def _get_texture(self, slot):
        source = self._texture_sources.get(slot)
        if source is not None and self._native.texture_resources.get(slot) == source.resource_handle:
            return source
        return getattr(self._native, slot)

    def _set_texture(self, slot, texture):
        resources = self._native.texture_resources
        if isinstance(texture, TextureSource):
            if self._resources is not None and texture._resources is not self._resources:
                raise ValueError("Texture source belongs to a different App")
            setattr(self._native, slot, None)
            resources[slot] = texture.resource_handle
            self._texture_sources[slot] = texture
        else:
            setattr(self._native, slot, unwrap_native(texture))
            resources.pop(slot, None)
            self._texture_sources.pop(slot, None)
        self._native.texture_resources = resources

    @property
    def unlit(self) -> bool:
        """Display base color and textures without lighting; defaults to False."""
        return self._native.unlit

    @unlit.setter
    def unlit(self, enabled: bool) -> None:
        self._native.unlit = enabled

    def set_unlit(self, enabled: bool = True):
        """Set unlit rendering and return this material."""
        self._native.set_unlit(enabled)
        return self

    @property
    def texture_resources(self) -> dict[str, int]:
        """SceneResourceManager texture source handles by material slot."""
        return self._native.texture_resources

    @texture_resources.setter
    def texture_resources(self, value: dict[str, int]) -> None:
        self._native.texture_resources = value

    @property
    def texture_paths(self) -> dict[str, str]:
        """Texture file paths loaded when attaching a visual.

        Assign a complete dictionary, for example
        ``{"base_color_texture": "textures/wood.png"}``.
        Relative paths use the descriptor's ``asset_dir``.
        """
        return self._native.texture_paths

    @texture_paths.setter
    def texture_paths(self, value: dict[str, str]) -> None:
        self._native.texture_paths = value

    @property
    def texture_srgb(self) -> dict[str, bool]:
        """sRGB decoding by texture-path slot. Defaults to False (linear).

        Assign a complete dictionary, as with ``texture_paths``.
        """
        return self._native.texture_srgb

    @texture_srgb.setter
    def texture_srgb(self, value: dict[str, bool]) -> None:
        self._native.texture_srgb = value


class ColorMaterial(Material):
    """Use instance color and optional vertex RGB."""

    def __init__(self, style=ColorStyle.UNTEXTURED, native: Any | None = None):
        if native is None:
            native = _ke.ColorMaterial(style)
        super().__init__(native)

    @property
    def style(self):
        return self._native.style

    def set_style(self, style):
        self._native.set_style(style)
        return self


# Compatibility aliases.
VertexColorStyle = ColorStyle
NativeVertexColorMaterial = NativeColorMaterial
VertexColorMaterial = ColorMaterial


class PhongMaterial(Material):
    """Phong material wrapper with optional preset initialization."""

    def __init__(self, preset=None, native: Any | None = None):
        if native is None:
            native = _ke.PhongMaterial()
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
        return self._get_texture("diffuse_map")

    @diffuse_map.setter
    def diffuse_map(self, texture) -> None:
        self._set_texture("diffuse_map", texture)

    @property
    def specular_map(self):
        """Optional specular texture."""
        return self._get_texture("specular_map")

    @specular_map.setter
    def specular_map(self, texture) -> None:
        self._set_texture("specular_map", texture)

    @property
    def alpha_map(self):
        """Optional alpha mask texture, such as OBJ ``map_d``."""
        return self._get_texture("alpha_map")

    @alpha_map.setter
    def alpha_map(self, texture) -> None:
        self._set_texture("alpha_map", texture)

    @property
    def normal_map(self):
        """Optional tangent-space normal map."""
        return self._get_texture("normal_map")

    @normal_map.setter
    def normal_map(self, texture) -> None:
        self._set_texture("normal_map", texture)

    def set_ambient(self, ambient):
        self._native.set_ambient(ambient)
        return self

    def set_diffuse(self, diffuse):
        self._native.set_diffuse(diffuse)
        return self

    def set_specular(self, specular):
        self._native.set_specular(specular)
        return self

    def set_shininess(self, shininess: float) -> PhongMaterial:
        self._native.set_shininess(float(shininess))
        return self

    def set_diffuse_map(self, texture):
        self.diffuse_map = texture
        return self

    def set_specular_map(self, texture):
        self.specular_map = texture
        return self

    def set_alpha_map(self, texture):
        self.alpha_map = texture
        return self

    def set_normal_map(self, texture):
        self.normal_map = texture
        return self


class PBRMaterial(Material):
    """PBR material wrapper with optional preset initialization."""

    def __init__(self, preset=None, native: Any | None = None):
        if native is None:
            native = _ke.PBRMaterial()
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
        return self._get_texture("base_color_texture")

    @base_color_texture.setter
    def base_color_texture(self, texture) -> None:
        self._set_texture("base_color_texture", texture)

    @property
    def normal_texture(self):
        return self._get_texture("normal_texture")

    @normal_texture.setter
    def normal_texture(self, texture) -> None:
        self._set_texture("normal_texture", texture)

    @property
    def metallic_roughness_texture(self):
        return self._get_texture("metallic_roughness_texture")

    @metallic_roughness_texture.setter
    def metallic_roughness_texture(self, texture) -> None:
        self._set_texture("metallic_roughness_texture", texture)

    @property
    def metallic_texture(self):
        return self._get_texture("metallic_texture")

    @metallic_texture.setter
    def metallic_texture(self, texture) -> None:
        self._set_texture("metallic_texture", texture)

    @property
    def roughness_texture(self):
        return self._get_texture("roughness_texture")

    @roughness_texture.setter
    def roughness_texture(self, texture) -> None:
        self._set_texture("roughness_texture", texture)

    @property
    def ao_texture(self):
        return self._get_texture("ao_texture")

    @ao_texture.setter
    def ao_texture(self, texture) -> None:
        self._set_texture("ao_texture", texture)

    @property
    def orm_texture(self):
        return self._get_texture("orm_texture")

    @orm_texture.setter
    def orm_texture(self, texture) -> None:
        self._set_texture("orm_texture", texture)

    @property
    def emissive_texture(self):
        return self._get_texture("emissive_texture")

    @emissive_texture.setter
    def emissive_texture(self, texture) -> None:
        self._set_texture("emissive_texture", texture)

    @property
    def opacity_texture(self):
        """Linear red-channel opacity mask overriding the base-color texture alpha."""
        return self._get_texture("opacity_texture")

    @opacity_texture.setter
    def opacity_texture(self, texture) -> None:
        self._set_texture("opacity_texture", texture)

    def set_base_color(self, base_color):
        self._native.set_base_color(base_color)
        return self

    def set_metallic(self, metallic: float) -> PBRMaterial:
        self._native.set_metallic(float(metallic))
        return self

    def set_roughness(self, roughness: float) -> PBRMaterial:
        self._native.set_roughness(float(roughness))
        return self

    def set_emissive_color(self, emissive_color):
        self._native.set_emissive_color(emissive_color)
        return self

    def set_emissive_strength(self, emissive_strength: float) -> PBRMaterial:
        self._native.set_emissive_strength(float(emissive_strength))
        return self

    def set_base_color_texture(self, texture):
        self.base_color_texture = texture
        return self

    def set_normal_texture(self, texture):
        self.normal_texture = texture
        return self

    def set_metallic_roughness_texture(self, texture):
        self.metallic_roughness_texture = texture
        return self

    def set_metallic_texture(self, texture):
        self.metallic_texture = texture
        return self

    def set_roughness_texture(self, texture):
        self.roughness_texture = texture
        return self

    def set_ao_texture(self, texture):
        self.ao_texture = texture
        return self

    def set_orm_texture(self, texture):
        self.orm_texture = texture
        return self

    def set_emissive_texture(self, texture):
        self.emissive_texture = texture
        return self

    def set_opacity_texture(self, texture):
        self.opacity_texture = texture
        return self


__all__ = [
    "ColorStyle",
    "VertexColorStyle",
    "VertexColorMaterial",
    "NativeVertexColorMaterial",
    "PhongMaterialType",
    "PBRMaterialType",
    "NativeMaterial",
    "NativeColorMaterial",
    "NativePhongMaterial",
    "NativePBRMaterial",
    "Material",
    "ColorMaterial",
    "PhongMaterial",
    "PBRMaterial",
    "unwrap_native",
]
