"""Optional Newton simulation viewer adapter."""

from ._dependency import NewtonUnavailableError, is_newton_available
from .viewer import NewtonViewer

__all__ = [
    "NewtonUnavailableError",
    "NewtonViewer",
    "is_newton_available",
]
