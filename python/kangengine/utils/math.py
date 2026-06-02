import numpy as np


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
    """Convert a wxyz quaternion to xyzw ordering."""
    w, x, y, z = np.asarray(quat, dtype=np.float32).reshape(4)
    return np.array([x, y, z, w], dtype=np.float32)


def quat_wxyz_twist_angle(quat, axis) -> float:
    """Return the signed local-axis twist angle from a wxyz quaternion."""
    w, x, y, z = np.asarray(quat, dtype=np.float32).reshape(4)
    xyz = np.array([x, y, z], dtype=np.float32)
    if w < 0.0:
        w, xyz = -w, -xyz
    axis = normalize_vector(axis, fallback=[1.0, 0.0, 0.0])
    projection = float(np.dot(xyz, axis))
    return float(2.0 * np.arctan2(projection, w))
