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
