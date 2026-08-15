"""Newton-to-KangEngine transform conventions."""

from __future__ import annotations

import numpy as np


def xyzw_to_rotation_matrix(quaternions) -> np.ndarray:
    """Return conventional row-major 3x3 matrices for xyzw quaternions."""

    q = np.asarray(quaternions, dtype=np.float32).reshape(-1, 4)
    lengths = np.sum(q * q, axis=1, keepdims=True)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("Newton transform contains a zero quaternion")
    q = q / np.sqrt(lengths)
    x, y, z, w = q.T

    result = np.empty((len(q), 3, 3), dtype=np.float32)
    result[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[:, 0, 1] = 2.0 * (x * y - z * w)
    result[:, 0, 2] = 2.0 * (x * z + y * w)
    result[:, 1, 0] = 2.0 * (x * y + z * w)
    result[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[:, 1, 2] = 2.0 * (y * z - x * w)
    result[:, 2, 0] = 2.0 * (x * z - y * w)
    result[:, 2, 1] = 2.0 * (y * z + x * w)
    result[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def transform_array_to_glm_matrices(xforms, scales=None, out=None) -> np.ndarray:
    """Convert Newton ``[px,py,pz,qx,qy,qz,qw]`` to GLM storage matrices.

    Python instance uploads expose GLM's column-major memory layout as
    ``[N, 4, 4]`` arrays. Translation therefore occupies ``matrix[3, :3]``.
    """

    values = np.asarray(xforms, dtype=np.float32).reshape(-1, 7)
    count = len(values)
    if scales is None:
        scale_values = np.ones((count, 3), dtype=np.float32)
    else:
        scale_values = np.asarray(scales, dtype=np.float32).reshape(-1, 3)
        if len(scale_values) == 1 and count > 1:
            scale_values = np.repeat(scale_values, count, axis=0)
        if len(scale_values) != count:
            raise ValueError(
                f"scales has {len(scale_values)} entries; expected 1 or {count}"
            )

    if (
        out is None
        or not isinstance(out, np.ndarray)
        or out.dtype != np.float32
        or out.shape != (count, 4, 4)
        or not out.flags.c_contiguous
    ):
        matrices = np.empty((count, 4, 4), dtype=np.float32)
    else:
        matrices = out

    q = values[:, 3:7]
    lengths = np.sum(q * q, axis=1, keepdims=True)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("Newton transform contains a zero quaternion")
    q = q / np.sqrt(lengths)
    x, y, z, w = q.T
    sx, sy, sz = scale_values.T

    matrices[:, 0, 0] = (1.0 - 2.0 * (y * y + z * z)) * sx
    matrices[:, 0, 1] = (2.0 * (x * y + z * w)) * sx
    matrices[:, 0, 2] = (2.0 * (x * z - y * w)) * sx
    matrices[:, 1, 0] = (2.0 * (x * y - z * w)) * sy
    matrices[:, 1, 1] = (1.0 - 2.0 * (x * x + z * z)) * sy
    matrices[:, 1, 2] = (2.0 * (y * z + x * w)) * sy
    matrices[:, 2, 0] = (2.0 * (x * z + y * w)) * sz
    matrices[:, 2, 1] = (2.0 * (y * z - x * w)) * sz
    matrices[:, 2, 2] = (1.0 - 2.0 * (x * x + y * y)) * sz
    matrices[:, :3, 3] = 0.0
    matrices[:, 3, :3] = values[:, :3]
    matrices[:, 3, 3] = 1.0
    return matrices
