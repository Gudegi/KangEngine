"""Differentiable LBFGS inverse kinematics for MJCF kinematic chains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

@dataclass(frozen=True)
class BodyLink:
    name: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    joint_name: str | None
    joint_axis: tuple[float, float, float] | None
    joint_range: tuple[float, float] | None


@dataclass(frozen=True)
class SiteFrame:
    name: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class IKResult:
    q: torch.Tensor
    ee_pos: torch.Tensor
    target_pos: torch.Tensor
    position_error: float
    loss: float
    iterations: int
    success: bool


def _vec3_tuple(value) -> tuple[float, float, float]:
    return float(value.x), float(value.y), float(value.z)


def _quat_wxyz_tuple(value) -> tuple[float, float, float, float]:
    return float(value.w), float(value.x), float(value.y), float(value.z)


def _parse_chain_from_kangengine(
    xml_path: str | Path,
    site_name: str,
    joint_names: tuple[str, ...],
    *,
    scale: float = 1.0,
    order: str = "DFS",
) -> tuple[list[BodyLink], SiteFrame]:
    from ._core import _ke

    data = _ke.asset.MJCFLoader.load(str(xml_path), scale=scale, order=order)

    sites = getattr(data, "sites", None)
    if not sites or site_name not in sites:
        raise ValueError(f"site '{site_name}' was not found in {xml_path}")

    tree = data.skeleton_tree
    joints = data.joints
    site = sites[site_name]
    body_indices = []
    body_idx = int(site.body_index)
    while body_idx >= 0:
        body_indices.append(body_idx)
        body_idx = int(tree.parent_index(body_idx))
    body_indices.reverse()

    links = []
    for body_idx in body_indices:
        joint_name = None
        joint_axis = None
        joint_range = None
        for joint in joints.get(body_idx, []):
            if joint.name in joint_names:
                joint_name = str(joint.name)
                joint_axis = _vec3_tuple(joint.axis)
                joint_range = float(joint.lo_limit), float(joint.hi_limit)
                break
        links.append(
            BodyLink(
                name=str(tree.node_name(body_idx)),
                pos=_vec3_tuple(tree.local_translation(body_idx)),
                quat_wxyz=_quat_wxyz_tuple(tree.local_rotation(body_idx)),
                joint_name=joint_name,
                joint_axis=joint_axis,
                joint_range=joint_range,
            )
        )

    return links, SiteFrame(
        name=str(site.name),
        pos=_vec3_tuple(site.pos),
        quat_wxyz=_quat_wxyz_tuple(site.quat),
    )


def _load_chain(
    xml_path: str | Path,
    site_name: str,
    joint_names: tuple[str, ...],
    *,
    scale: float = 1.0,
    order: str = "DFS",
) -> tuple[list[BodyLink], SiteFrame]:
    return _parse_chain_from_kangengine(
        xml_path, site_name, joint_names, scale=scale, order=order
    )


def _tensor3(value, *, dtype, device) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.clamp(torch.linalg.vector_norm(quat), min=1e-12)
    w, x, y, z = quat.unbind()
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.vector_norm(axis), min=1e-12)
    x, y, z = axis.unbind()
    zero = angle.new_tensor(0.0)
    k = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + torch.sin(angle) * k + (1 - torch.cos(angle)) * (k @ k)


def _stack_vec3(values, *, dtype, device) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _stack_quat_matrices(values, *, dtype, device) -> torch.Tensor:
    quats = torch.tensor(values, dtype=dtype, device=device)
    return torch.stack([_quat_wxyz_to_matrix(q) for q in quats])


def _transform_from_rot_pos(rot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    bottom = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]], dtype=pos.dtype, device=pos.device
    )
    return torch.cat(
        (
            torch.cat((rot, pos.reshape(3, 1)), dim=1),
            bottom,
        ),
        dim=0,
    )


class MJCFLBFGSIK:
    """Differentiable FK + bounded LBFGS IK for an MJCF site frame."""

    def __init__(
        self,
        xml_path: str | Path,
        *,
        site_name: str,
        joint_names: tuple[str, ...] | list[str],
        scale: float = 1.0,
        order: str = "DFS",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.site_name = site_name
        self.requested_joint_names = tuple(joint_names)
        self.scale = float(scale)
        self.order = order
        self.dtype = dtype
        self.device = torch.device(device)
        self.links, self.site = _load_chain(
            self.xml_path,
            self.site_name,
            self.requested_joint_names,
            scale=self.scale,
            order=self.order,
        )
        self.joint_names = tuple(
            link.joint_name
            for link in self.links
            if link.joint_name in self.requested_joint_names
        )
        self.joint_limits = torch.tensor(
            [
                link.joint_range
                for link in self.links
                if link.joint_name in self.requested_joint_names
            ],
            dtype=self.dtype,
            device=self.device,
        )
        if self.joint_names != self.requested_joint_names:
            raise ValueError(
                "unexpected MJCF joint order: "
                f"expected {self.requested_joint_names}, got {self.joint_names}"
            )
        self._cache_torch_chain()

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def _cache_torch_chain(self) -> None:
        self._link_pos = _stack_vec3(
            [link.pos for link in self.links], dtype=self.dtype, device=self.device
        )
        self._link_rot = _stack_quat_matrices(
            [link.quat_wxyz for link in self.links],
            dtype=self.dtype,
            device=self.device,
        )
        self._site_pos = _tensor3(self.site.pos, dtype=self.dtype, device=self.device)
        self._site_rot = _quat_wxyz_to_matrix(
            torch.tensor(self.site.quat_wxyz, dtype=self.dtype, device=self.device)
        )
        self._joint_axes = _stack_vec3(
            [
                link.joint_axis
                for link in self.links
                if link.joint_name in self.requested_joint_names
            ],
            dtype=self.dtype,
            device=self.device,
        )
        joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self._link_q_indices = tuple(
            joint_index.get(link.joint_name) if link.joint_name else None
            for link in self.links
        )

    def _site_position_and_body_rotation(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.as_tensor(q, dtype=self.dtype, device=self.device).reshape(self.num_joints)
        rot = torch.eye(3, dtype=self.dtype, device=self.device)
        pos = torch.zeros(3, dtype=self.dtype, device=self.device)

        for link_idx, q_idx in enumerate(self._link_q_indices):
            pos = pos + rot @ self._link_pos[link_idx]
            rot = rot @ self._link_rot[link_idx]
            if q_idx is not None:
                joint_rot = _axis_angle_to_matrix(self._joint_axes[q_idx], q[q_idx])
                rot = rot @ joint_rot

        pos = pos + rot @ self._site_pos
        return pos, rot

    def forward_kinematics(self, q: torch.Tensor) -> torch.Tensor:
        pos, rot = self._site_position_and_body_rotation(q)
        rot = rot @ self._site_rot
        return _transform_from_rot_pos(rot, pos)

    def end_effector_position(self, q: torch.Tensor) -> torch.Tensor:
        pos, _ = self._site_position_and_body_rotation(q)
        return pos

    def solve_position(
        self,
        target_pos,
        *,
        initial_q=None,
        max_iter: int = 80,
        tolerance_m: float = 1e-4,
        posture_weight: float = 0.0,
        line_search_fn: str = "strong_wolfe",
    ) -> IKResult:
        target = torch.as_tensor(target_pos, dtype=self.dtype, device=self.device).reshape(3)
        if initial_q is None:
            initial = torch.zeros(self.num_joints, dtype=self.dtype, device=self.device)
        else:
            initial = torch.as_tensor(initial_q, dtype=self.dtype, device=self.device).reshape(self.num_joints)
        initial = self.clamp_to_limits(initial)

        raw = self._q_to_raw(initial).detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [raw],
            lr=1.0,
            max_iter=int(max_iter),
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn=line_search_fn,
        )
        iterations = 0
        last_loss = None

        def closure():
            nonlocal iterations, last_loss
            optimizer.zero_grad()
            q = self._raw_to_q(raw)
            ee_pos = self.end_effector_position(q)
            pos_loss = torch.sum((ee_pos - target) ** 2)
            posture_loss = torch.sum((q - initial) ** 2)
            loss = pos_loss + float(posture_weight) * posture_loss
            loss.backward()
            iterations += 1
            last_loss = float(loss.detach().cpu())
            return loss

        optimizer.step(closure)
        q = self._raw_to_q(raw.detach())
        ee_pos = self.end_effector_position(q).detach()
        error = float(torch.linalg.vector_norm(ee_pos - target).cpu())
        return IKResult(
            q=q.detach(),
            ee_pos=ee_pos,
            target_pos=target.detach(),
            position_error=error,
            loss=float(last_loss if last_loss is not None else error * error),
            iterations=iterations,
            success=error <= float(tolerance_m),
        )

    def clamp_to_limits(self, q: torch.Tensor) -> torch.Tensor:
        q = torch.as_tensor(q, dtype=self.dtype, device=self.device).reshape(self.num_joints)
        return torch.maximum(torch.minimum(q, self.joint_limits[:, 1]), self.joint_limits[:, 0])

    def _raw_to_q(self, raw: torch.Tensor) -> torch.Tensor:
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return mid + half * torch.tanh(raw)

    def _q_to_raw(self, q: torch.Tensor) -> torch.Tensor:
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        x = torch.clamp((q - mid) / half, -0.999999, 0.999999)
        return torch.atanh(x)

    def q_to_joint_positions(
        self, q: torch.Tensor, *, degrees: bool = False
    ) -> dict[str, float]:
        q = torch.as_tensor(q, dtype=self.dtype, device=self.device).reshape(self.num_joints)
        values = torch.rad2deg(q) if degrees else q
        return {
            name: float(values[i].cpu())
            for i, name in enumerate(self.joint_names)
        }

