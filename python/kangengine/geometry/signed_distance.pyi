from numpy.typing import ArrayLike, NDArray
import numpy as np


class SignedDistanceMesh:
    """Immutable closed-mesh snapshot with native BVH signed-distance queries."""

    content_hash: str
    def __init__(self, vertices: ArrayLike, faces: ArrayLike) -> None: ...
    def query(self, points: ArrayLike, *, chunk_size: int = 2048) -> tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Return distance, closest point, and gradient; negative means inside."""
        ...
