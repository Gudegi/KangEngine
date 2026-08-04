"""Typed material authoring API for IDE completion and generated API reference."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar

from kangengine.render import Texture

class PhongMaterialType(Enum):
    """Built-in Blinn-Phong material preset."""
    EMERALD: PhongMaterialType
    JADE: PhongMaterialType
    OBSIDIAN: PhongMaterialType
    PEARL: PhongMaterialType
    RUBY: PhongMaterialType
    TURQUOISE: PhongMaterialType
    BRASS: PhongMaterialType
    BRONZE: PhongMaterialType
    CHROME: PhongMaterialType
    COPPER: PhongMaterialType
    GOLD: PhongMaterialType
    SILVER: PhongMaterialType
    BLACK_PLASTIC: PhongMaterialType
    CYAN_PLASTIC: PhongMaterialType
    GREEN_PLASTIC: PhongMaterialType
    RED_PLASTIC: PhongMaterialType
    WHITE_PLASTIC: PhongMaterialType
    YELLOW_PLASTIC: PhongMaterialType
    BLACK_RUBBER: PhongMaterialType
    CYAN_RUBBER: PhongMaterialType
    GREEN_RUBBER: PhongMaterialType
    RED_RUBBER: PhongMaterialType
    WHITE_RUBBER: PhongMaterialType
    YELLOW_RUBBER: PhongMaterialType

class PBRMaterialType(Enum):
    """Built-in metallic-roughness material preset."""
    GRAY_CARD: PBRMaterialType
    WHITE_PLASTIC: PBRMaterialType
    BLACK_PLASTIC: PBRMaterialType
    BLACK_RUBBER: PBRMaterialType
    CHARCOAL: PBRMaterialType
    CARROT: PBRMaterialType
    CONCRETE: PBRMaterialType
    RED_BRICK: PBRMaterialType
    ALUMINUM: PBRMaterialType
    CHROME: PBRMaterialType
    COPPER: PBRMaterialType
    GOLD: PBRMaterialType
    EMISSIVE_BLUE: PBRMaterialType

class VertexColorStyle(Enum):
    """Built-in rendering style for vertex/display-color materials."""
    UNTEXTURED: VertexColorStyle
    TEXTURED: VertexColorStyle
    CHECKERBOARD: VertexColorStyle
    DEBUG_CHECKER: VertexColorStyle

class NativeMaterial:
    """Native renderer material base used for advanced interop."""

class NativeVertexColorMaterial(NativeMaterial):
    """Native vertex-color material used for advanced interop."""

class NativePhongMaterial(NativeMaterial):
    """Native Phong material used for advanced interop."""

class NativePBRMaterial(NativeMaterial):
    """Native PBR material used for advanced interop."""

class Material:
    """Base facade wrapping a native renderer material."""
    def __init__(self, native: NativeMaterial) -> None: ...
    @property
    def native(self) -> NativeMaterial:
        """Return the wrapped native material."""
        ...

class VertexColorMaterial(Material):
    """Material using vertex/display colors and an optional built-in style."""
    def __init__(self, style: VertexColorStyle = VertexColorStyle.UNTEXTURED, native: NativeVertexColorMaterial | None = None) -> None: ...
    @property
    def style(self) -> VertexColorStyle:
        """Return the active built-in vertex-color style."""
        ...
    def set_style(self, style: VertexColorStyle) -> VertexColorMaterial:
        """Set the rendering style and return this material."""
        ...

class PhongMaterial(Material):
    """Blinn-Phong material with factors and optional texture maps."""
    def __init__(self, preset: PhongMaterialType | None = None, native: NativePhongMaterial | None = None) -> None: ...
    def load_from_preset(self, preset: PhongMaterialType) -> PhongMaterial:
        """Load a built-in preset and return this material."""
        ...
    @property
    def ambient(self) -> Any:
        """Ambient RGB factor."""
        ...
    @ambient.setter
    def ambient(self, value: Any) -> None: ...
    @property
    def diffuse(self) -> Any:
        """Diffuse RGB factor."""
        ...
    @diffuse.setter
    def diffuse(self, value: Any) -> None: ...
    @property
    def specular(self) -> Any:
        """Specular RGB factor."""
        ...
    @specular.setter
    def specular(self, value: Any) -> None: ...
    @property
    def shininess(self) -> float:
        """Specular highlight exponent."""
        ...
    @shininess.setter
    def shininess(self, value: float) -> None: ...
    @property
    def diffuse_map(self) -> Texture | None:
        """Optional diffuse texture."""
        ...
    @diffuse_map.setter
    def diffuse_map(self, texture: Texture | None) -> None: ...
    @property
    def specular_map(self) -> Texture | None:
        """Optional specular texture."""
        ...
    @specular_map.setter
    def specular_map(self, texture: Texture | None) -> None: ...
    @property
    def alpha_map(self) -> Texture | None:
        """Optional alpha-mask texture, such as OBJ ``map_d``."""
        ...
    @alpha_map.setter
    def alpha_map(self, texture: Texture | None) -> None: ...
    @property
    def normal_map(self) -> Texture | None:
        """Optional tangent-space normal map."""
        ...
    @normal_map.setter
    def normal_map(self, texture: Texture | None) -> None: ...
    def set_ambient(self, ambient: Any) -> PhongMaterial:
        """Set the ambient factor and return this material."""
        ...
    def set_diffuse(self, diffuse: Any) -> PhongMaterial:
        """Set the diffuse factor and return this material."""
        ...
    def set_specular(self, specular: Any) -> PhongMaterial:
        """Set the specular factor and return this material."""
        ...
    def set_shininess(self, shininess: float) -> PhongMaterial:
        """Set the highlight exponent and return this material."""
        ...
    def set_diffuse_map(self, texture: Texture | None) -> PhongMaterial:
        """Set the diffuse map and return this material."""
        ...
    def set_specular_map(self, texture: Texture | None) -> PhongMaterial:
        """Set the specular map and return this material."""
        ...
    def set_alpha_map(self, texture: Texture | None) -> PhongMaterial:
        """Set the alpha map and return this material."""
        ...
    def set_normal_map(self, texture: Texture | None) -> PhongMaterial:
        """Set the normal map and return this material."""
        ...

class PBRMaterial(Material):
    """Metallic-roughness PBR material with factors and optional textures."""
    def __init__(self, preset: PBRMaterialType | None = None, native: NativePBRMaterial | None = None) -> None: ...
    def load_from_preset(self, preset: PBRMaterialType) -> PBRMaterial:
        """Load a built-in preset and return this material."""
        ...
    @property
    def base_color(self) -> Any:
        """Base-color RGBA factor."""
        ...
    @base_color.setter
    def base_color(self, value: Any) -> None: ...
    @property
    def metallic(self) -> float:
        """Metallic factor in the range 0..1."""
        ...
    @metallic.setter
    def metallic(self, value: float) -> None: ...
    @property
    def roughness(self) -> float:
        """Roughness factor in the range 0..1."""
        ...
    @roughness.setter
    def roughness(self, value: float) -> None: ...
    @property
    def emissive_color(self) -> Any:
        """Emissive RGB factor."""
        ...
    @emissive_color.setter
    def emissive_color(self, value: Any) -> None: ...
    @property
    def emissive_strength(self) -> float:
        """Multiplier applied to the emissive color."""
        ...
    @emissive_strength.setter
    def emissive_strength(self, value: float) -> None: ...
    @property
    def base_color_texture(self) -> Texture | None:
        """Optional base-color texture."""
        ...
    @base_color_texture.setter
    def base_color_texture(self, texture: Texture | None) -> None: ...
    @property
    def normal_texture(self) -> Texture | None:
        """Optional tangent-space normal texture."""
        ...
    @normal_texture.setter
    def normal_texture(self, texture: Texture | None) -> None: ...
    @property
    def metallic_roughness_texture(self) -> Texture | None:
        """Optional packed metallic-roughness texture."""
        ...
    @metallic_roughness_texture.setter
    def metallic_roughness_texture(self, texture: Texture | None) -> None: ...
    @property
    def metallic_texture(self) -> Texture | None:
        """Optional single-channel metallic texture."""
        ...
    @metallic_texture.setter
    def metallic_texture(self, texture: Texture | None) -> None: ...
    @property
    def roughness_texture(self) -> Texture | None:
        """Optional single-channel roughness texture."""
        ...
    @roughness_texture.setter
    def roughness_texture(self, texture: Texture | None) -> None: ...
    @property
    def ao_texture(self) -> Texture | None:
        """Optional ambient-occlusion texture."""
        ...
    @ao_texture.setter
    def ao_texture(self, texture: Texture | None) -> None: ...
    @property
    def orm_texture(self) -> Texture | None:
        """Optional packed occlusion-roughness-metallic texture."""
        ...
    @orm_texture.setter
    def orm_texture(self, texture: Texture | None) -> None: ...
    @property
    def emissive_texture(self) -> Texture | None:
        """Optional emissive texture."""
        ...
    @emissive_texture.setter
    def emissive_texture(self, texture: Texture | None) -> None: ...
    def set_base_color(self, base_color: Any) -> PBRMaterial:
        """Set the base-color factor and return this material."""
        ...
    def set_metallic(self, metallic: float) -> PBRMaterial:
        """Set the metallic factor and return this material."""
        ...
    def set_roughness(self, roughness: float) -> PBRMaterial:
        """Set the roughness factor and return this material."""
        ...
    def set_emissive_color(self, emissive_color: Any) -> PBRMaterial:
        """Set the emissive color and return this material."""
        ...
    def set_emissive_strength(self, emissive_strength: float) -> PBRMaterial:
        """Set the emissive strength and return this material."""
        ...
    def set_base_color_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the base-color texture and return this material."""
        ...
    def set_normal_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the normal texture and return this material."""
        ...
    def set_metallic_roughness_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the packed metallic-roughness texture and return this material."""
        ...
    def set_metallic_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the metallic texture and return this material."""
        ...
    def set_roughness_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the roughness texture and return this material."""
        ...
    def set_ao_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the ambient-occlusion texture and return this material."""
        ...
    def set_orm_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the packed ORM texture and return this material."""
        ...
    def set_emissive_texture(self, texture: Texture | None) -> PBRMaterial:
        """Set the emissive texture and return this material."""
        ...

_T = TypeVar("_T")

def unwrap_native(value: _T) -> Any:
    """Return a facade's native object, or pass a native value through."""
    ...

__all__: list[str]
