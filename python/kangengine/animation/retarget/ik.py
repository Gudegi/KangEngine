"""Backend-independent contracts for articulation IK retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...utils.batched_rotations import (
    quat_wxyz_multiply,
    quat_wxyz_normalize,
    quat_wxyz_rotate,
)

if TYPE_CHECKING:
    from .. import SkeletonMotion
    from .ik_profile import IKRetargetConfig


@dataclass(frozen=True)
class IKEffectorMapping:
    """Map one source skeleton joint to one target articulation link."""

    source_joint: str
    target_link: str
    region: str | None = None
    position_weight: float = 1.0
    rotation_weight: float = 0.0
    position_scale: tuple[float, float, float] | None = None
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_offset_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.source_joint or not self.target_link:
            raise ValueError("source_joint and target_link must not be empty")
        if self.position_weight < 0.0 or self.rotation_weight < 0.0:
            raise ValueError("effector weights must be non-negative")


@dataclass(frozen=True)
class AuxiliaryRetargetPoint:
    """Add a solver target at an offset from an existing robot link."""

    name: str
    source_joint: str
    target_link: str
    target_offset: tuple[float, float, float]
    source_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    region: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.source_joint or not self.target_link:
            raise ValueError("auxiliary point names and links must not be empty")
        if self.weight < 0.0:
            raise ValueError("auxiliary point weight must be non-negative")


@dataclass(frozen=True)
class IKTargetMotion:
    """Frame-major world-space targets consumed by articulation IK solvers."""

    fps: float
    source_root_joint: str
    target_root_link: str
    target_link_names: tuple[str, ...]
    target_offsets: torch.Tensor
    primary_effector_count: int
    positions: torch.Tensor
    rotations_wxyz: torch.Tensor
    position_weights: torch.Tensor
    rotation_weights: torch.Tensor
    contact_mask: torch.Tensor | None = None
    initial_root_rotations_wxyz: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        frames = self.positions.shape[0] if self.positions.ndim == 3 else -1
        effectors = len(self.target_link_names)
        if self.target_offsets.shape != (effectors, 3):
            raise ValueError("target_offsets must have shape (effectors, 3)")
        if not 1 <= self.primary_effector_count <= effectors:
            raise ValueError("primary_effector_count must be in [1, effectors]")
        if self.positions.shape != (frames, effectors, 3):
            raise ValueError("positions must have shape (frames, effectors, 3)")
        if self.rotations_wxyz.shape != (frames, effectors, 4):
            raise ValueError("rotations_wxyz must have shape (frames, effectors, 4)")
        if self.position_weights.shape != (effectors,):
            raise ValueError("position_weights must have shape (effectors,)")
        if self.rotation_weights.shape != (effectors,):
            raise ValueError("rotation_weights must have shape (effectors,)")
        if self.contact_mask is not None and self.contact_mask.shape != (
            frames,
            effectors,
        ):
            raise ValueError("contact_mask must have shape (frames, effectors)")
        if self.initial_root_rotations_wxyz is not None and (
            self.initial_root_rotations_wxyz.shape != (frames, 4)
        ):
            raise ValueError("initial_root_rotations_wxyz must have shape (frames, 4)")

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_effectors(self) -> int:
        return len(self.target_link_names)


def build_ik_target_motion(
    motion: SkeletonMotion,
    profile: IKRetargetConfig,
    *,
    contact_mask: torch.Tensor | None = None,
) -> IKTargetMotion:
    """Extract and scale world-space IK targets from a skeleton motion.

    The input motion must already use the target robot's coordinate system.
    Region and per-effector scale are applied to root-relative positions while
    preserving the source root trajectory.
    """

    source_names = tuple(motion.node_names())
    source_indices = {name: index for index, name in enumerate(source_names)}
    requested_names = [profile.source_root_joint]
    requested_names.extend(mapping.source_joint for mapping in profile.link_mappings)
    requested_names.extend(point.source_joint for point in profile.auxiliary_points)
    missing_names = sorted(set(requested_names) - set(source_indices))
    if missing_names:
        raise ValueError(
            "retarget profile references joints absent from source motion: "
            + ", ".join(missing_names)
        )

    global_positions = torch.as_tensor(motion.global_positions(), dtype=torch.float32)
    global_rotations = torch.as_tensor(
        motion.global_rotations_wxyz(), dtype=torch.float32
    )
    if global_positions.ndim != 3 or global_positions.shape[-1] != 3:
        raise ValueError("motion global positions must have shape (frames, joints, 3)")
    if global_rotations.shape != (*global_positions.shape[:2], 4):
        raise ValueError("motion global rotations must have shape (frames, joints, 4)")

    root_index = source_indices[profile.source_root_joint]
    root_positions = global_positions[:, root_index : root_index + 1, :]
    scaled_root_positions = root_positions
    if profile.root_position_scale is not None:
        scaled_root_positions = root_positions * torch.tensor(
            profile.root_position_scale, dtype=root_positions.dtype
        )
    target_positions = []
    target_rotations = []
    for mapping in profile.link_mappings:
        source_index = source_indices[mapping.source_joint]
        position = global_positions[:, source_index : source_index + 1, :]
        scale = mapping.position_scale
        if scale is None and mapping.region is not None:
            scale = profile.region_scales.get(mapping.region)
        if scale is not None:
            scale_tensor = torch.tensor(scale, dtype=position.dtype)
            position = (
                scaled_root_positions + (position - root_positions) * scale_tensor
            )
        elif profile.root_position_scale is not None:
            position = scaled_root_positions + (position - root_positions)
        source_rotation = global_rotations[:, source_index : source_index + 1, :]
        rotation_offset = torch.tensor(
            mapping.rotation_offset_wxyz, dtype=source_rotation.dtype
        )
        calibrated_rotation = quat_wxyz_normalize(
            quat_wxyz_multiply(source_rotation, rotation_offset)
        )
        offset = torch.tensor(mapping.position_offset, dtype=position.dtype)
        target_positions.append(
            position + quat_wxyz_rotate(calibrated_rotation, offset)
        )
        target_rotations.append(calibrated_rotation)

    for point in profile.auxiliary_points:
        source_index = source_indices[point.source_joint]
        source_rotation = global_rotations[:, source_index : source_index + 1, :]
        position = global_positions[:, source_index : source_index + 1, :]
        scale = profile.region_scales.get(point.region) if point.region else None
        if scale is not None:
            position = scaled_root_positions + (
                position - root_positions
            ) * torch.tensor(scale, dtype=position.dtype)
        position = position + quat_wxyz_rotate(
            source_rotation,
            torch.tensor(point.source_offset, dtype=position.dtype),
        )
        target_positions.append(position)
        target_rotations.append(source_rotation)

    positions = torch.cat(target_positions, dim=1)
    rotations = torch.cat(target_rotations, dim=1)
    return IKTargetMotion(
        fps=float(motion.fps()),
        source_root_joint=profile.source_root_joint,
        target_root_link=profile.target_root_link,
        target_link_names=tuple(
            mapping.target_link for mapping in profile.link_mappings
        )
        + tuple(point.target_link for point in profile.auxiliary_points),
        target_offsets=torch.tensor(
            [(0.0, 0.0, 0.0)] * len(profile.link_mappings)
            + [point.target_offset for point in profile.auxiliary_points],
            dtype=torch.float32,
        ),
        primary_effector_count=len(profile.link_mappings),
        positions=positions,
        rotations_wxyz=rotations,
        position_weights=torch.tensor(
            [mapping.position_weight for mapping in profile.link_mappings]
            + [point.weight for point in profile.auxiliary_points],
            dtype=torch.float32,
        ),
        rotation_weights=torch.tensor(
            [mapping.rotation_weight for mapping in profile.link_mappings]
            + [0.0 for point in profile.auxiliary_points],
            dtype=torch.float32,
        ),
        contact_mask=contact_mask,
        initial_root_rotations_wxyz=global_rotations[:, root_index],
    )


def detect_foot_contacts(
    motion: SkeletonMotion,
    profile: IKRetargetConfig,
    *,
    up_axis: int = 2,
    height_threshold: float = 0.045,
    velocity_threshold: float = 0.35,
) -> torch.Tensor:
    """Estimate mapped foot contacts from source height and horizontal speed."""

    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    if height_threshold < 0.0 or velocity_threshold < 0.0:
        raise ValueError("contact thresholds must be non-negative")
    # Keep one implementation of the contact heuristic for the editor and
    # offline retargeting. ContactData uses the motion's canonical sampled
    # positions and velocities, including its semantic/name mapping support.
    from ...motion_module.editor import ContactData

    source_names = tuple(motion.node_names())
    source_indices = {name: index for index, name in enumerate(source_names)}
    positions = np.asarray(motion.global_positions(), dtype=np.float32)
    frames = len(positions)
    contacts = torch.zeros(
        (frames, len(profile.link_mappings) + len(profile.auxiliary_points)),
        dtype=torch.bool,
    )
    contact_slots = [
        index
        for index, mapping in enumerate(profile.link_mappings)
        if mapping.target_link in profile.contact_links
    ]
    if not contact_slots:
        return contacts
    source_contact_names = list(
        dict.fromkeys(
            profile.link_mappings[index].source_joint for index in contact_slots
        )
    )
    missing = sorted(set(source_contact_names) - set(source_indices))
    if missing:
        raise ValueError(f"contact source joint is absent: {', '.join(missing)}")
    selected_indices = [source_indices[name] for name in source_contact_names]
    velocities = np.zeros_like(positions)
    if frames > 1:
        velocities[:-1] = np.diff(positions, axis=0) * float(motion.fps())
        velocities[-1] = velocities[-2]
    data = ContactData(
        positions[:, selected_indices],
        velocities[:, selected_indices],
        selected_indices,
        source_contact_names,
        up_axis,
    )
    detected = data.contacts(height_threshold, velocity_threshold)
    detected_by_name = {
        name: detected[:, index] for index, name in enumerate(data.foot_names)
    }
    active = np.stack(
        [
            detected_by_name[profile.link_mappings[index].source_joint]
            for index in contact_slots
        ],
        axis=1,
    )
    contact_values = torch.from_numpy(active).to(dtype=torch.float32)
    if frames > 1:
        # ProtoMotions uses a five-frame cross-fade so contact costs turn on/off
        # continuously instead of introducing a hard trajectory discontinuity.
        padded = torch.nn.functional.pad(
            contact_values.transpose(0, 1).unsqueeze(1), (2, 2), mode="replicate"
        )
        contact_values = (
            torch.nn.functional.avg_pool1d(padded, kernel_size=5, stride=1)
            .squeeze(1)
            .transpose(0, 1)
        )
    contacts = contacts.to(dtype=torch.float32)
    contacts[:, contact_slots] = contact_values
    return contacts


__all__ = [
    "AuxiliaryRetargetPoint",
    "IKEffectorMapping",
    "IKTargetMotion",
    "build_ik_target_motion",
    "detect_foot_contacts",
]
