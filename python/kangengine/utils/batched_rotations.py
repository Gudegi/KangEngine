"""Batched torch quaternion helpers."""

import torch


def _require_last_dim(value: torch.Tensor, width: int, name: str) -> None:
    if value.ndim == 0 or value.shape[-1] != width:
        raise ValueError(f"{name} must have shape [..., {width}]")


def quat_wxyz_normalize(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize wxyz quaternions, falling back to identity for near-zero inputs."""
    _require_last_dim(quat, 4, "quat")
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    normalized = quat / norm.clamp_min(eps)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(norm >= eps, normalized, identity)


def quat_wxyz_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply broadcast-compatible wxyz quaternion batches."""
    _require_last_dim(a, 4, "a")
    _require_last_dim(b, 4, "b")
    a, b = torch.broadcast_tensors(a, b)
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_wxyz_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of wxyz quaternion batches."""
    _require_last_dim(quat, 4, "quat")
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_wxyz_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized wxyz quaternions using a cross-product formula."""
    _require_last_dim(quat, 4, "quat")
    _require_last_dim(vector, 3, "vector")
    xyz, vector = torch.broadcast_tensors(quat[..., 1:], vector)
    w = torch.broadcast_to(quat[..., :1], xyz.shape[:-1] + (1,))
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + w * t + torch.cross(xyz, t, dim=-1)


def quat_wxyz_rotate_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of normalized wxyz quaternions."""
    return quat_wxyz_rotate(quat_wxyz_conjugate(quat), vector)


def quat_wxyz_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Build normalized wxyz quaternions from broadcast-compatible angles and axes."""
    _require_last_dim(axis, 3, "axis")
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    normalized_axis = axis / norm.clamp_min(1e-8)
    fallback_axis = torch.zeros_like(axis)
    fallback_axis[..., 0] = 1.0
    normalized_axis = torch.where(norm >= 1e-8, normalized_axis, fallback_axis)
    half_angle = angle.unsqueeze(-1) * 0.5
    xyz = normalized_axis * torch.sin(half_angle)
    w = torch.ones_like(xyz[..., :1]) * torch.cos(half_angle)
    return quat_wxyz_normalize(torch.cat((w, xyz), dim=-1))


def quat_wxyz_heading_xy(quat: torch.Tensor) -> torch.Tensor:
    """Angle of local +X projected onto XY, measured from world +X."""
    reference = torch.zeros_like(quat[..., 1:])
    reference[..., 0] = 1.0
    direction = quat_wxyz_rotate(quat, reference)
    return torch.atan2(direction[..., 1], direction[..., 0])


def quat_wxyz_heading_yz(quat: torch.Tensor) -> torch.Tensor:
    """Angle of local +Y projected onto YZ, measured from world +Y."""
    reference = torch.zeros_like(quat[..., 1:])
    reference[..., 1] = 1.0
    direction = quat_wxyz_rotate(quat, reference)
    return torch.atan2(direction[..., 2], direction[..., 1])


def quat_wxyz_heading_xz(quat: torch.Tensor) -> torch.Tensor:
    """Angle of local +X projected onto XZ, measured from world +X."""
    reference = torch.zeros_like(quat[..., 1:])
    reference[..., 0] = 1.0
    direction = quat_wxyz_rotate(quat, reference)
    return torch.atan2(direction[..., 2], direction[..., 0])


def _heading_quat_wxyz(
    quat: torch.Tensor, plane: str, inverse: bool
) -> torch.Tensor:
    if plane == "xy":
        angle = quat_wxyz_heading_xy(quat)
        axis_index = 2
    elif plane == "yz":
        angle = quat_wxyz_heading_yz(quat)
        axis_index = 0
    elif plane == "xz":
        angle = quat_wxyz_heading_xz(quat)
        axis_index = 1
    else:
        raise ValueError("plane must be 'xy', 'yz', or 'xz'")
    axis = torch.zeros_like(quat[..., 1:])
    axis[..., axis_index] = 1.0
    return quat_wxyz_from_angle_axis(-angle if inverse else angle, axis)


def quat_wxyz_heading_quat(quat: torch.Tensor, plane: str = "xy") -> torch.Tensor:
    """Encode one projected heading as a WXYZ axis-angle quaternion."""
    return _heading_quat_wxyz(quat, plane, False)


def quat_wxyz_heading_quat_inverse(
    quat: torch.Tensor, plane: str = "xy"
) -> torch.Tensor:
    """Return the inverse projected-heading quaternion for a plane."""
    return _heading_quat_wxyz(quat, plane, True)


def quat_wxyz_to_tangent_normal(quat: torch.Tensor) -> torch.Tensor:
    """Encode rotation by its rotated local +X and +Z basis vectors."""
    tangent = torch.zeros_like(quat[..., 1:])
    tangent[..., 0] = 1.0
    normal = torch.zeros_like(tangent)
    normal[..., 2] = 1.0
    return torch.cat(
        (quat_wxyz_rotate(quat, tangent), quat_wxyz_rotate(quat, normal)),
        dim=-1,
    )


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized WXYZ quaternion batches to 3x3 matrices."""
    quat = quat_wxyz_normalize(quat)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    values = (
        1.0 - two * (y * y + z * z),
        two * (x * y - z * w),
        two * (x * z + y * w),
        two * (x * y + z * w),
        1.0 - two * (x * x + z * z),
        two * (y * z - x * w),
        two * (x * z - y * w),
        two * (y * z + x * w),
        1.0 - two * (x * x + y * y),
    )
    return torch.stack(values, dim=-1).reshape(quat.shape[:-1] + (3, 3))


def matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert proper rotation matrices ``[..., 3, 3]`` to WXYZ quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape [..., 3, 3]")
    result = []
    for value in matrix.reshape(-1, 3, 3):
        trace = torch.trace(value)
        if trace > 0:
            scale = torch.sqrt(trace + 1.0) * 2.0
            quat = torch.stack((0.25 * scale,
                                (value[2, 1] - value[1, 2]) / scale,
                                (value[0, 2] - value[2, 0]) / scale,
                                (value[1, 0] - value[0, 1]) / scale))
        else:
            index = int(torch.argmax(torch.diagonal(value)))
            if index == 0:
                scale = torch.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
                quat = torch.stack(((value[2, 1] - value[1, 2]) / scale,
                                    0.25 * scale,
                                    (value[0, 1] + value[1, 0]) / scale,
                                    (value[0, 2] + value[2, 0]) / scale))
            elif index == 1:
                scale = torch.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
                quat = torch.stack(((value[0, 2] - value[2, 0]) / scale,
                                    (value[0, 1] + value[1, 0]) / scale,
                                    0.25 * scale,
                                    (value[1, 2] + value[2, 1]) / scale))
            else:
                scale = torch.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
                quat = torch.stack(((value[1, 0] - value[0, 1]) / scale,
                                    (value[0, 2] + value[2, 0]) / scale,
                                    (value[1, 2] + value[2, 1]) / scale,
                                    0.25 * scale))
        result.append(quat_wxyz_normalize(quat))
    return torch.stack(result).reshape(matrix.shape[:-2] + (4,))


# KangEngine/PhysX-facing XYZW helpers. Keep the convention in every public
# name because this module also contains WXYZ utilities used by import paths.
def quat_xyzw_normalize(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize XYZW quaternion batches, using identity for zero inputs."""
    _require_last_dim(quat, 4, "quat")
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    normalized = quat / norm.clamp_min(eps)
    identity = torch.zeros_like(quat)
    identity[..., 3] = 1.0
    return torch.where(norm >= eps, normalized, identity)


def quat_xyzw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply broadcast-compatible XYZW quaternion batches."""
    _require_last_dim(a, 4, "a")
    _require_last_dim(b, 4, "b")
    a, b = torch.broadcast_tensors(a, b)
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def quat_xyzw_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of XYZW quaternion batches."""
    _require_last_dim(quat, 4, "quat")
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


def quat_xyzw_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized XYZW quaternion batches."""
    _require_last_dim(quat, 4, "quat")
    _require_last_dim(vector, 3, "vector")
    xyz, vector = torch.broadcast_tensors(quat[..., :3], vector)
    w = torch.broadcast_to(quat[..., 3:], xyz.shape[:-1] + (1,))
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + w * t + torch.cross(xyz, t, dim=-1)


def quat_xyzw_rotate_inverse(
    quat: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors by the inverse of normalized XYZW quaternions."""
    return quat_xyzw_rotate(quat_xyzw_conjugate(quat), vector)


def quat_xyzw_from_angle_axis(
    angle: torch.Tensor, axis: torch.Tensor
) -> torch.Tensor:
    """Build normalized XYZW quaternions from angles and axes."""
    _require_last_dim(axis, 3, "axis")
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    normalized_axis = axis / norm.clamp_min(1e-8)
    fallback_axis = torch.zeros_like(axis)
    fallback_axis[..., 0] = 1.0
    normalized_axis = torch.where(norm >= 1e-8, normalized_axis, fallback_axis)
    half_angle = angle.unsqueeze(-1) * 0.5
    xyz = normalized_axis * torch.sin(half_angle)
    w = torch.cos(half_angle)
    return quat_xyzw_normalize(torch.cat((xyz, w), dim=-1))


def quat_xyzw_heading_xy(quat: torch.Tensor) -> torch.Tensor:
    """Angle of local +X projected onto XY, measured from world +X."""
    reference = torch.zeros_like(quat[..., :3])
    reference[..., 0] = 1.0
    direction = quat_xyzw_rotate(quat, reference)
    return torch.atan2(direction[..., 1], direction[..., 0])


def quat_xyzw_heading_yz(quat: torch.Tensor) -> torch.Tensor:
    """Angle of local +Y projected onto YZ, measured from world +Y."""
    reference = torch.zeros_like(quat[..., :3])
    reference[..., 1] = 1.0
    direction = quat_xyzw_rotate(quat, reference)
    return torch.atan2(direction[..., 2], direction[..., 1])


def quat_xyzw_heading_xz(quat: torch.Tensor) -> torch.Tensor:
    """Angle of local +X projected onto XZ, measured from world +X."""
    reference = torch.zeros_like(quat[..., :3])
    reference[..., 0] = 1.0
    direction = quat_xyzw_rotate(quat, reference)
    return torch.atan2(direction[..., 2], direction[..., 0])


def _heading_quat_xyzw(
    quat: torch.Tensor, plane: str, inverse: bool
) -> torch.Tensor:
    if plane == "xy":
        angle = quat_xyzw_heading_xy(quat)
        axis_index = 2
    elif plane == "yz":
        angle = quat_xyzw_heading_yz(quat)
        axis_index = 0
    elif plane == "xz":
        angle = quat_xyzw_heading_xz(quat)
        axis_index = 1
    else:
        raise ValueError("plane must be 'xy', 'yz', or 'xz'")
    axis = torch.zeros_like(quat[..., :3])
    axis[..., axis_index] = 1.0
    return quat_xyzw_from_angle_axis(-angle if inverse else angle, axis)


def quat_xyzw_heading_quat(quat: torch.Tensor, plane: str = "xy") -> torch.Tensor:
    """Encode one projected heading as an XYZW axis-angle quaternion."""
    return _heading_quat_xyzw(quat, plane, False)


def quat_xyzw_heading_quat_inverse(
    quat: torch.Tensor, plane: str = "xy"
) -> torch.Tensor:
    """Return the inverse projected-heading quaternion for a plane."""
    return _heading_quat_xyzw(quat, plane, True)


def quat_xyzw_to_tangent_normal(quat: torch.Tensor) -> torch.Tensor:
    """Encode rotation by its rotated local +X and +Z basis vectors."""
    tangent = torch.zeros_like(quat[..., :3])
    tangent[..., 0] = 1.0
    normal = torch.zeros_like(tangent)
    normal[..., 2] = 1.0
    return torch.cat(
        (quat_xyzw_rotate(quat, tangent), quat_xyzw_rotate(quat, normal)),
        dim=-1,
    )


def quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized XYZW quaternion batches to 3x3 matrices."""
    quat = quat_xyzw_normalize(quat)
    x, y, z, w = quat.unbind(dim=-1)
    two = 2.0
    values = (
        1.0 - two * (y * y + z * z),
        two * (x * y - z * w),
        two * (x * z + y * w),
        two * (x * y + z * w),
        1.0 - two * (x * x + z * z),
        two * (y * z - x * w),
        two * (x * z - y * w),
        two * (y * z + x * w),
        1.0 - two * (x * x + y * y),
    )
    return torch.stack(values, dim=-1).reshape(quat.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Return the first two matrix columns as a continuous 6D encoding."""
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape [..., 3, 3]")
    return matrix[..., :, :2].transpose(-2, -1).reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a continuous 6D rotation encoding to an orthonormal matrix."""
    _require_last_dim(rotation, 6, "rotation")
    a1, a2 = rotation[..., :3], rotation[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)
