"""Color conversion helpers."""

from .._core import _ke


def preset_rgba(color_type, alpha: float = 1.0) -> list[float]:
    """Return a ColorLibrary preset as an RGBA list with optional alpha."""
    color = _ke.ColorLibrary.get(color_type)
    return [color.r, color.g, color.b, float(alpha)]
