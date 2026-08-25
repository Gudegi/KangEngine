"""Motion storage and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
import torch

from .._core import _ke
from ..utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_rotate,
    quat_wxyz_slerp,
    quat_wxyz_to_rotation_vector,
)
from .articulation_motion import ArticulationMotion, ArticulationMotionMapper

SkeletonMotion = _ke.animation.SkeletonMotion

BackendSample = TypeVar("BackendSample")
# Newton adapter -> NewtonMotionTensorSample
# PhysX adapter  -> PhysXMotionTensorSample
# MuJoCo adapter -> MuJoCoMotionTensorSample


class MotionAdapter(Protocol[BackendSample]):
    """Simulator adapter consumed by :class:`MotionLibrary`."""

    def prepare_motion(
        self, motion: SkeletonMotion | ArticulationMotion
    ) -> torch.Tensor | np.ndarray:
        """Precompute frame data needed by simulator-native sampling."""

    def pack_sample(
        self,
        *,
        root_positions: torch.Tensor,
        root_rotations_wxyz: torch.Tensor,
        root_linear_velocities: torch.Tensor,
        root_angular_velocities: torch.Tensor,
        local_rotations_wxyz: torch.Tensor,
        backend_frame_data: torch.Tensor,
        next_backend_frame_data: torch.Tensor,
        blend: torch.Tensor,
    ) -> BackendSample:
        """Convert interpolated quaternion poses to simulator-native tensors."""


@dataclass(frozen=True)
class MotionSample(Generic[BackendSample]):
    """Quaternion pose and optional backend state sampled on one device."""

    root_positions: torch.Tensor
    root_rotations_wxyz: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor
    local_rotations_wxyz: torch.Tensor
    backend_state: BackendSample | None
    motion_ids: torch.Tensor
    frame_indices: torch.Tensor
    next_frame_indices: torch.Tensor
    blend: torch.Tensor


@dataclass(frozen=True)
class MotionKinematics:
    """Robot-body transforms produced by sampled-pose FK."""

    body_positions: torch.Tensor
    body_rotations_wxyz: torch.Tensor


class MotionLibrary(Generic[BackendSample]):
    """Pack quaternion clips and sample them on one Torch device."""

    def __init__(
        self,
        motions: list[SkeletonMotion | ArticulationMotion]
        | tuple[SkeletonMotion | ArticulationMotion, ...] = (),
        *,
        adapter: MotionAdapter[BackendSample] | None = None,
        weights: torch.Tensor | np.ndarray | list[float] | None = None,
        device: str | torch.device = "cpu",
        keep_source_motions: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._adapter = adapter
        self._source_motions: list[SkeletonMotion] | None = (
            [] if keep_source_motions else None
        )
        self._names: list[str] = []
        self._node_names: tuple[str, ...] | None = None
        self._parent_indices: tuple[int, ...] | None = None
        self._body_local_offsets = torch.empty((0, 3), device=self._device)
        self._root_positions = torch.empty((0, 3), device=self._device)
        self._root_linear_velocities = torch.empty((0, 3), device=self._device)
        self._root_angular_velocities = torch.empty((0, 3), device=self._device)
        self._local_rotations = torch.empty((0, 0, 4), device=self._device)
        self._backend_frame_data: torch.Tensor | None = None
        self._frame_offsets = torch.empty(0, dtype=torch.long, device=self._device)
        self._frame_counts = torch.empty(0, dtype=torch.long, device=self._device)
        self._fps = torch.empty(0, device=self._device)
        self._durations = torch.empty(0, device=self._device)
        self._weights = torch.empty(0, device=self._device)

        for motion in motions:
            self.add_motion(motion)
        if weights is not None:
            self.set_weights(weights)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_motions(self) -> int:
        return len(self._names)

    @property
    def keep_source_motions(self) -> bool:
        return self._source_motions is not None

    @property
    def motion_names(self) -> tuple[str, ...]:
        return tuple(self._names)

    @property
    def durations(self) -> torch.Tensor:
        return self._durations

    @property
    def frame_counts(self) -> torch.Tensor:
        return self._frame_counts

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    def to(self, device: str | torch.device) -> MotionLibrary[BackendSample]:
        """Move every packed motion buffer to one Torch device."""

        resolved = torch.device(device)
        if resolved == self._device:
            return self
        self._device = resolved
        for name in (
            "_body_local_offsets",
            "_root_positions",
            "_root_linear_velocities",
            "_root_angular_velocities",
            "_local_rotations",
            "_frame_offsets",
            "_frame_counts",
            "_fps",
            "_durations",
            "_weights",
        ):
            setattr(self, name, getattr(self, name).to(resolved))
        if self._backend_frame_data is not None:
            self._backend_frame_data = self._backend_frame_data.to(resolved)
        return self

    def add_motion(
        self,
        motion: SkeletonMotion | ArticulationMotion,
        *,
        name: str | None = None,
        weight: float = 1.0,
    ) -> int:
        """Upload one clip, converting articulation coordinates when necessary."""

        skeleton_motion = self._as_skeleton_motion(motion)
        count = skeleton_motion.num_frames()
        if count < 1:
            raise ValueError("MotionLibrary does not accept empty clips")
        names = tuple(skeleton_motion.node_names())
        parents = tuple(skeleton_motion.parent_indices())
        if self._node_names is None:
            self._node_names = names
            self._parent_indices = parents
            tree = skeleton_motion.skeleton_tree
            self._body_local_offsets = self._tensor(
                [
                    [
                        tree.local_translation(index).x,
                        tree.local_translation(index).y,
                        tree.local_translation(index).z,
                    ]
                    for index in range(len(names))
                ]
            )
            self._root_positions = torch.empty((0, 3), device=self._device)
            self._root_linear_velocities = torch.empty((0, 3), device=self._device)
            self._root_angular_velocities = torch.empty((0, 3), device=self._device)
            self._local_rotations = torch.empty((0, len(names), 4), device=self._device)
        elif names != self._node_names or parents != self._parent_indices:
            raise ValueError("all MotionLibrary clips must share one skeleton topology")

        resolved_name = skeleton_motion.motion_name() if name is None else str(name)
        if resolved_name in self._names:
            raise ValueError(f"duplicate motion name {resolved_name!r}")
        if weight < 0.0:
            raise ValueError("motion weight must be non-negative")

        roots = self._tensor(skeleton_motion.root_translations())
        root_linear = self._tensor(skeleton_motion.root_linear_velocities())
        rotations = self._tensor(skeleton_motion.local_rotations_wxyz())
        root_angular = self._angular_velocities(
            rotations[:, 0, :], skeleton_motion.fps()
        )
        offset = self._local_rotations.shape[0]
        self._root_positions = torch.cat((self._root_positions, roots))
        self._root_linear_velocities = torch.cat(
            (self._root_linear_velocities, root_linear)
        )
        self._root_angular_velocities = torch.cat(
            (self._root_angular_velocities, root_angular)
        )
        self._local_rotations = torch.cat((self._local_rotations, rotations))

        if self._adapter is not None:
            prepared = self._tensor(self._adapter.prepare_motion(motion))
            if prepared.shape[0] != count:
                raise ValueError("adapter frame data must match the motion frame count")
            if self._backend_frame_data is None:
                self._backend_frame_data = prepared
            elif prepared.shape[1:] != self._backend_frame_data.shape[1:]:
                raise ValueError("adapter frame-data shape differs between clips")
            else:
                self._backend_frame_data = torch.cat(
                    (self._backend_frame_data, prepared)
                )

        self._frame_offsets = self._append(self._frame_offsets, offset, torch.long)
        self._frame_counts = self._append(self._frame_counts, count, torch.long)
        self._fps = self._append(self._fps, skeleton_motion.fps())
        self._durations = self._append(self._durations, skeleton_motion.duration())
        self._weights = self._append(self._weights, weight)
        if self._source_motions is not None:
            self._source_motions.append(skeleton_motion)
        self._names.append(resolved_name)
        return len(self._names) - 1

    def set_weights(self, weights: torch.Tensor | np.ndarray | list[float]) -> None:
        """Replace motion-selection weights."""

        values = torch.as_tensor(weights, dtype=torch.float32, device=self._device)
        if values.shape != (self.num_motions,):
            raise ValueError(f"weights expected shape [{self.num_motions}]")
        if bool(torch.any(values < 0.0)) or float(values.sum()) <= 0.0:
            raise ValueError("weights must be non-negative with a positive sum")
        self._weights = values

    def motion(self, motion_id: int) -> SkeletonMotion:
        """Return a retained source clip when source retention is enabled."""

        index = self._validate_motion_id(motion_id)
        if self._source_motions is None:
            raise RuntimeError(
                "source motions are not retained; construct MotionLibrary with "
                "keep_source_motions=True"
            )
        return self._source_motions[index]

    def motion_id(self, name: str) -> int:
        """Return the stable ID associated with a motion name."""

        try:
            return self._names.index(name)
        except ValueError as error:
            raise KeyError(name) from error

    def sample_motion_ids(self, count: int) -> torch.Tensor:
        """Sample motion IDs using configured clip weights."""

        if self.num_motions == 0:
            raise RuntimeError("cannot sample an empty MotionLibrary")
        return torch.multinomial(self._weights, int(count), replacement=True)

    def sample_times(
        self,
        motion_ids: torch.Tensor | np.ndarray | list[int] | int,
        *,
        truncate_time: float = 0.0,
    ) -> torch.Tensor:
        """Sample random valid times for supplied motion IDs."""

        ids = self._motion_ids(motion_ids)
        durations = self._durations[ids]
        if truncate_time < 0.0 or bool(torch.any(durations < truncate_time)):
            raise ValueError("truncate_time must lie within every selected motion")
        return torch.rand_like(durations) * (durations - truncate_time)

    def sample_frames(
        self,
        motion_ids: torch.Tensor | np.ndarray | list[int] | int,
        frame_indices: torch.Tensor | np.ndarray | list[int] | int,
        *,
        loop: bool = True,
    ) -> MotionSample[BackendSample]:
        """Sample exact frames with broadcastable device inputs."""

        ids = self._motion_ids(motion_ids)
        frames = torch.as_tensor(frame_indices, dtype=torch.long, device=self._device)
        ids, frames = torch.broadcast_tensors(ids, frames)
        counts = self._frame_counts[ids]
        resolved = torch.remainder(frames, counts) if loop else frames.clamp_min(0)
        if not loop:
            resolved = torch.minimum(resolved, counts - 1)
        return self._make_sample(ids, resolved, resolved, torch.zeros_like(ids).float())

    def sample(
        self,
        motion_ids: torch.Tensor | np.ndarray | list[int] | int,
        times: torch.Tensor | np.ndarray | list[float] | float,
        *,
        loop: bool = True,
        interpolate: bool = True,
    ) -> MotionSample[BackendSample]:
        """Sample clips at broadcastable device times in seconds."""

        ids = self._motion_ids(motion_ids)
        values = torch.as_tensor(times, dtype=torch.float32, device=self._device)
        ids, values = torch.broadcast_tensors(ids, values)
        counts = self._frame_counts[ids]
        durations = self._durations[ids]
        if loop:
            duration = durations.clamp_min(torch.finfo(torch.float32).eps)
            position = torch.remainder(values, duration) * self._fps[ids]
            position = torch.where(counts > 1, position, torch.zeros_like(position))
        else:
            position = (values * self._fps[ids]).clamp_min(0.0)
            position = torch.minimum(position, (counts - 1).float())
        first = torch.floor(position).long()
        following = torch.minimum(first + 1, counts - 1)
        blend = position - first.float()
        if not interpolate:
            following = first
            blend = torch.zeros_like(blend)
        return self._make_sample(ids, first, following, blend)

    def forward_kinematics(
        self, sample: MotionSample[BackendSample]
    ) -> MotionKinematics:
        """Run batched FK on the library's robot skeleton after pose sampling."""

        if self._parent_indices is None:
            raise RuntimeError("cannot run FK on an empty MotionLibrary")
        rotations = sample.local_rotations_wxyz
        body_positions = [sample.root_positions]
        body_rotations = [rotations[..., 0, :]]
        for joint in range(1, len(self._parent_indices)):
            parent = self._parent_indices[joint]
            if parent < 0 or parent >= joint:
                raise RuntimeError(
                    "MotionLibrary FK requires parent-before-child joint order"
                )
            parent_rotation = body_rotations[parent]
            offset = quat_wxyz_rotate(parent_rotation, self._body_local_offsets[joint])
            body_positions.append(body_positions[parent] + offset)
            body_rotations.append(
                quat_wxyz_multiply(parent_rotation, rotations[..., joint, :])
            )
        return MotionKinematics(
            body_positions=torch.stack(body_positions, dim=-2),
            body_rotations_wxyz=torch.stack(body_rotations, dim=-2),
        )

    def _make_sample(self, ids, first, following, blend):
        offsets = self._frame_offsets[ids]
        first_packed = offsets + first
        next_packed = offsets + following
        blend_column = blend.unsqueeze(-1)
        roots = (
            self._root_positions[first_packed]
            + (self._root_positions[next_packed] - self._root_positions[first_packed])
            * blend_column
        )
        local_rotations = quat_wxyz_slerp(
            self._local_rotations[first_packed],
            self._local_rotations[next_packed],
            blend.unsqueeze(-1),
        )
        root_rotations = local_rotations[..., 0, :]
        root_linear = self._root_linear_velocities[first_packed]
        root_angular = self._root_angular_velocities[first_packed]
        backend_state = None
        if self._adapter is not None:
            assert self._backend_frame_data is not None
            backend_state = self._adapter.pack_sample(
                root_positions=roots,
                root_rotations_wxyz=root_rotations,
                root_linear_velocities=root_linear,
                root_angular_velocities=root_angular,
                local_rotations_wxyz=local_rotations,
                backend_frame_data=self._backend_frame_data[first_packed],
                next_backend_frame_data=self._backend_frame_data[next_packed],
                blend=blend,
            )
        return MotionSample(
            root_positions=roots,
            root_rotations_wxyz=root_rotations,
            root_linear_velocities=root_linear,
            root_angular_velocities=root_angular,
            local_rotations_wxyz=local_rotations,
            backend_state=backend_state,
            motion_ids=ids,
            frame_indices=first,
            next_frame_indices=following,
            blend=blend,
        )

    def _as_skeleton_motion(
        self, motion: SkeletonMotion | ArticulationMotion
    ) -> SkeletonMotion:
        if isinstance(motion, SkeletonMotion):
            return motion
        if isinstance(motion, ArticulationMotion):
            return ArticulationMotionMapper(motion.layout).to_skeleton_motion(motion)
        raise TypeError("motion must be SkeletonMotion or ArticulationMotion")

    def _tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self._device)

    @staticmethod
    def _angular_velocities(rotations: torch.Tensor, fps: float) -> torch.Tensor:
        """Differentiate WXYZ rotations with MotionLib's forward convention."""

        result = torch.zeros(
            rotations.shape[:-1] + (3,),
            dtype=rotations.dtype,
            device=rotations.device,
        )
        if rotations.shape[0] <= 1:
            return result
        delta = quat_wxyz_multiply(rotations[1:], quat_wxyz_conjugate(rotations[:-1]))
        result[:-1] = quat_wxyz_to_rotation_vector(delta) * float(fps)
        result[-1] = result[-2]
        return result

    def _append(self, tensor, value, dtype=torch.float32):
        item = torch.tensor([value], dtype=dtype, device=self._device)
        return torch.cat((tensor, item))

    def _motion_ids(self, motion_ids) -> torch.Tensor:
        ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self._device)
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= self.num_motions)):
            raise IndexError("motion_ids contain an out-of-range ID")
        return ids

    def _validate_motion_id(self, motion_id: int) -> int:
        motion_id = int(motion_id)
        if motion_id < 0 or motion_id >= self.num_motions:
            raise IndexError(f"motion_id {motion_id} is out of range")
        return motion_id
