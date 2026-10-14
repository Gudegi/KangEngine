import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_vector_array(value, size: int, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1:] != (size,):
        raise ValueError(f"{name} must have a final dimension of {size}")
    return array


def normalize_vector(value, fallback=None, eps: float = 1e-8) -> np.ndarray:
    """Return a normalized float32 vector or a fallback for a near-zero input."""
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm >= eps:
        return (vector / norm).astype(np.float32)
    if fallback is None:
        return np.zeros_like(vector)
    return np.asarray(fallback, dtype=np.float32).reshape(vector.shape)


def quat_xyzw_normalize(quat, eps: float = 1e-8) -> np.ndarray:
    """Normalize an xyzw quaternion, falling back to identity."""
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def quat_xyzw_multiply(a, b) -> np.ndarray:
    """Multiply two xyzw quaternions."""
    ax, ay, az, aw = np.asarray(a, dtype=np.float32).reshape(4)
    bx, by, bz, bw = np.asarray(b, dtype=np.float32).reshape(4)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def quat_xyzw_conjugate(quat) -> np.ndarray:
    """Return the conjugate of an xyzw quaternion."""
    x, y, z, w = np.asarray(quat, dtype=np.float32).reshape(4)
    return np.array([-x, -y, -z, w], dtype=np.float32)


def quat_xyzw_rotate(quat, vector) -> np.ndarray:
    """Rotate a 3D vector by an xyzw quaternion."""
    quat = quat_xyzw_normalize(quat)
    vector_quat = np.array(
        [*np.asarray(vector, dtype=np.float32).reshape(3), 0.0],
        dtype=np.float32,
    )
    return quat_xyzw_multiply(
        quat_xyzw_multiply(quat, vector_quat),
        quat_xyzw_conjugate(quat),
    )[:3]


def quat_xyzw_from_two_vectors(src, dst) -> np.ndarray:
    """Return the shortest xyzw rotation from src to dst."""
    src = normalize_vector(src, fallback=[1.0, 0.0, 0.0])
    dst = normalize_vector(dst, fallback=[1.0, 0.0, 0.0])
    dot = float(np.dot(src, dst))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if dot < -0.999999:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = normalize_vector(axis, fallback=[1.0, 0.0, 0.0])
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
    axis = np.cross(src, dst)
    return quat_xyzw_normalize([axis[0], axis[1], axis[2], 1.0 + dot])


def quat_wxyz_to_xyzw(quat) -> np.ndarray:
    """Convert arbitrary-shaped wxyz quaternion arrays to xyzw ordering."""
    quat = _as_vector_array(quat, 4, "quat")
    return quat[..., [1, 2, 3, 0]].copy()


def quat_xyzw_to_wxyz(quat) -> np.ndarray:
    """Convert arbitrary-shaped xyzw quaternion arrays to wxyz ordering."""
    quat = _as_vector_array(quat, 4, "quat")
    return quat[..., [3, 0, 1, 2]].copy()


def quat_wxyz_multiply_numpy(first, second) -> np.ndarray:
    """Multiply broadcast-compatible NumPy wxyz quaternion arrays."""
    first = _as_vector_array(first, 4, "first")
    second = _as_vector_array(second, 4, "second")
    first, second = np.broadcast_arrays(first, second)
    aw, ax, ay, az = np.moveaxis(first, -1, 0)
    bw, bx, by, bz = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def quat_wxyz_to_rotation_vector_numpy(quat) -> np.ndarray:
    """Convert NumPy wxyz quaternion arrays to exponential-map vectors."""
    quat = _as_vector_array(quat, 4, "quat")
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-8)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    sin_half = np.linalg.norm(quat[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, quat[..., :1])
    scale = np.full_like(angle, 2.0)
    np.divide(angle, sin_half, out=scale, where=sin_half > 1.0e-8)
    return (quat[..., 1:] * scale).astype(np.float32, copy=False)


def quat_wxyz_from_rotation_vector_numpy(rotation_vector) -> np.ndarray:
    """Convert NumPy exponential-map vectors to wxyz quaternion arrays."""
    rotation_vector = _as_vector_array(rotation_vector, 3, "rotation_vector")
    angle = np.linalg.norm(rotation_vector, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.full_like(angle, 0.5)
    np.divide(np.sin(half), angle, out=scale, where=angle > 1.0e-8)
    return np.concatenate((np.cos(half), rotation_vector * scale), axis=-1).astype(
        np.float32, copy=False
    )


def quat_wxyz_rotate_numpy(quat, vector, *, inverse: bool = False) -> np.ndarray:
    """Rotate broadcast-compatible NumPy vectors by wxyz quaternions."""
    quat = _as_vector_array(quat, 4, "quat")
    vector = _as_vector_array(vector, 3, "vector")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    identity = np.zeros_like(quat)
    identity[..., 0] = 1.0
    quat = np.where(norm > 1e-8, quat / np.maximum(norm, 1e-8), identity)
    xyz = -quat[..., 1:4] if inverse else quat[..., 1:4]
    cross = 2.0 * np.cross(xyz, vector)
    return (vector + quat[..., :1] * cross + np.cross(xyz, cross)).astype(
        np.float32, copy=False
    )


def quat_wxyz_slerp_numpy(first, second, blend) -> np.ndarray:
    """Slerp broadcast-compatible NumPy wxyz quaternion arrays."""
    first = _as_vector_array(first, 4, "first")
    second = _as_vector_array(second, 4, "second")
    blend = np.asarray(blend, dtype=np.float32)
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    blend = blend[..., None]
    close = sin_theta < 1e-6
    first_weight = np.where(
        close,
        1.0 - blend,
        np.sin((1.0 - blend) * theta) / np.maximum(sin_theta, 1e-8),
    )
    second_weight = np.where(
        close,
        blend,
        np.sin(blend * theta) / np.maximum(sin_theta, 1e-8),
    )
    result = first_weight * first + second_weight * second
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    return (result / np.maximum(norm, 1e-8)).astype(np.float32, copy=False)


def quat_wxyz_twist_angle(quat, axis) -> float:
    """Return the signed local-axis twist angle from a wxyz quaternion."""
    w, x, y, z = np.asarray(quat, dtype=np.float32).reshape(4)
    xyz = np.array([x, y, z], dtype=np.float32)
    if w < 0.0:
        w, xyz = -w, -xyz
    axis = normalize_vector(axis, fallback=[1.0, 0.0, 0.0])
    projection = float(np.dot(xyz, axis))
    return float(2.0 * np.arctan2(projection, w))


def quat_wxyz_to_matrix_numpy(quat: ArrayLike) -> NDArray[np.float32]:
    """Convert NumPy wxyz quaternion arrays to float32 rotation matrices.

    Args:
        quat: Shape (..., 4), in wxyz order. Each quaternion is normalized internally.

    Returns:
        Shape (..., 3, 3), preserving batch dimensions. Matrices act on column vectors.

    Raises:
        ValueError: The final dimension is not 4, or any float32 squared norm
            is non-finite or at most 1e-12.
    """
    array = _as_vector_array(quat, 4, "quat")
    w, x, y, z = np.moveaxis(array, -1, 0)
    norm = w * w + x * x + y * y + z * z
    if np.any(~np.isfinite(norm) | (norm <= 1.0e-12)):
        raise ValueError("rotation quaternion must have a finite, non-zero norm")
    scale = 2.0 / norm
    return (
        np.stack(
            (
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ),
            axis=-1,
        )
        .reshape(array.shape[:-1] + (3, 3))
        .astype(np.float32, copy=False)
    )
