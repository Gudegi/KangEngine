"""Texture source references for renderers that own their GPU resources."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextureSource:
    """Immutable scene-local texture returned by App texture creation."""

    path: str | None
    srgb: bool
    flip: bool
    resource_handle: int
    _resources: object = field(repr=False)
