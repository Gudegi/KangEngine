"""Light value types exported by kangengine.scene."""

from enum import Enum
from . import Vec3

class LightType(Enum):
    DIRECTIONAL: LightType
    POINT: LightType
    SPOT: LightType
    RECT: LightType
    DOME: LightType

class DirectionalLight:
    """Infinite-distance light with an ambient contribution."""

    def __init__(self) -> None: ...
    direction: Vec3
    color: Vec3
    intensity: float
    ambient: Vec3

class PointLight:
    """Local light with position and range in scene units."""

    def __init__(self) -> None: ...
    position: Vec3
    color: Vec3
    intensity: float
    range: float

class SpotLight:
    """Local cone light with cone half-angles in radians."""

    def __init__(self) -> None: ...
    position: Vec3
    direction: Vec3
    color: Vec3
    intensity: float
    range: float
    inner_cone_angle: float
    outer_cone_angle: float

class LightEmission:
    """USD-compatible emission, independent of the owning prim's transform."""

    def __init__(self) -> None: ...
    color: Vec3
    intensity: float
    exposure: float
    normalize: bool
    enable_color_temperature: bool
    color_temperature: float
    diffuse: float
    specular: float

class RectLight(LightEmission):
    """Local XY rectangle emitting toward -Z; dimensions are in scene units."""

    width: float
    height: float
    texture_path: str

class DomeLight(LightEmission):
    """Uniform infinite environment, currently evaluated as diffuse illumination."""
    texture_path: str
    texture_format: str
