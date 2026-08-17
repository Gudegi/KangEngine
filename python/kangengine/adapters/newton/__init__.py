"""Optional Newton simulation viewer adapter."""

from ._dependency import NewtonUnavailableError, is_newton_available
from .viewer import ViewerKE

__all__ = [
    "NewtonUnavailableError",
    "ViewerKE",
    "is_newton_available",
]
