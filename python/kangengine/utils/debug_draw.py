from __future__ import annotations

import numpy as np


def log_debug_axes(
    app,
    path: str,
    origin,
    rotation,
    length: float = 0.18,
    width: float = 1.5,
    alpha: float = 0.95,
    hidden: bool = False,
) -> None:
    """Draw RGB transform axes using the native debug renderer."""
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    _ = alpha

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin

    app.log_debug_axes(path, transform, float(length), float(width), bool(hidden))
