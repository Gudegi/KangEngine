"""Motion storage and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, Protocol, Sequence, TypeVar

import numpy as np
import torch
import yaml

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
SkeletonTree = _ke.animation.SkeletonTree

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


@dataclass(frozen=True)
class MotionLibraryBuffers:
    """Borrowed view of frame-major buffers owned by a MotionLibrary.

    The tensors remain owned by the library and can therefore be shared with
    another GPU runtime without copying. Treat them as read-only. Adding a
    motion or moving the library to another device invalidates previously
    acquired views.
    """

    root_positions: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor
    local_rotations_wxyz: torch.Tensor
    frame_offsets: torch.Tensor
    frame_counts: torch.Tensor
    fps: torch.Tensor
    durations: torch.Tensor
    weights: torch.Tensor
    global_positions: torch.Tensor | None
    global_rotations_wxyz: torch.Tensor | None
    global_linear_velocities: torch.Tensor | None
    global_angular_velocities: torch.Tensor | None
    backend_frame_data: torch.Tensor | None


@dataclass(frozen=True)
class _PreparedMotion:
    skeleton_motion: SkeletonMotion
    name: str
    weight: float
    count: int
    fps: float
    duration: float
    root_positions: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor
    local_rotations_wxyz: torch.Tensor
    global_positions: torch.Tensor | None
    global_rotations_wxyz: torch.Tensor | None
    global_linear_velocities: torch.Tensor | None
    global_angular_velocities: torch.Tensor | None
    backend_frame_data: torch.Tensor | None


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
        precompute_kinematics: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._adapter = adapter
        self._source_motions: list[SkeletonMotion] | None = (
            [] if keep_source_motions else None
        )
        self._precompute_kinematics = bool(precompute_kinematics)
        self._names: list[str] = []
        self._node_names: tuple[str, ...] | None = None
        self._parent_indices: tuple[int, ...] | None = None
        self._body_local_offsets = torch.empty((0, 3), device=self._device)
        self._root_positions = torch.empty((0, 3), device=self._device)
        self._root_linear_velocities = torch.empty((0, 3), device=self._device)
        self._root_angular_velocities = torch.empty((0, 3), device=self._device)
        self._local_rotations = torch.empty((0, 0, 4), device=self._device)
        self._global_positions: torch.Tensor | None = None
        self._global_rotations: torch.Tensor | None = None
        self._global_linear_velocities: torch.Tensor | None = None
        self._global_angular_velocities: torch.Tensor | None = None
        self._backend_frame_data: torch.Tensor | None = None
        self._frame_offsets = torch.empty(0, dtype=torch.long, device=self._device)
        self._frame_counts = torch.empty(0, dtype=torch.long, device=self._device)
        self._fps = torch.empty(0, device=self._device)
        self._durations = torch.empty(0, device=self._device)
        self._weights = torch.empty(0, device=self._device)

        self._add_loaded_motions((motion, None, 1.0) for motion in motions)
        if weights is not None:
            self.set_weights(weights)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        adapter: MotionAdapter[BackendSample] | None = None,
        motion_loader: Callable[[Path], SkeletonMotion | ArticulationMotion]
        | None = None,
        skeleton_tree: SkeletonTree | None = None,
        fallback_joint_names: Sequence[str] | None = None,
        device: str | torch.device = "cpu",
        keep_source_motions: bool = False,
        precompute_kinematics: bool = False,
    ) -> MotionLibrary[BackendSample]:
        """Load one motion or a weighted YAML motion collection.

        BVH files are loaded directly. NPY motion dictionaries require a target
        ``skeleton_tree``; use ``fallback_joint_names`` when their rotation
        arrays do not include joint names. Use ``motion_loader`` to support
        additional formats. Relative paths in a YAML collection are resolved
        from the YAML file's directory.
        """

        source = Path(path).expanduser().resolve()
        entries = cls._motion_file_entries(source)
        target_device = torch.device(device)
        library = cls(
            adapter=adapter,
            device="cpu",
            keep_source_motions=keep_source_motions,
            precompute_kinematics=precompute_kinematics,
        )

        def load_entries():
            for motion_id, (motion_path, weight) in enumerate(entries):
                motion = (
                    motion_loader(motion_path)
                    if motion_loader is not None
                    else cls._load_motion_file(
                        motion_path,
                        skeleton_tree=skeleton_tree,
                        fallback_joint_names=fallback_joint_names,
                    )
                )
                yield motion, f"{motion_id}:{motion_path.stem}", weight

        library._add_loaded_motions(load_entries())
        return library.to(target_device)

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
    def fps(self) -> torch.Tensor:
        return self._fps

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
            "_global_positions",
            "_global_rotations",
            "_global_linear_velocities",
            "_global_angular_velocities",
            "_frame_offsets",
            "_frame_counts",
            "_fps",
            "_durations",
            "_weights",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(resolved))
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

        return self._add_loaded_motions([(motion, name, weight)])[0]

    def _add_loaded_motions(
        self,
        entries: Iterable[
            tuple[SkeletonMotion | ArticulationMotion, str | None, float]
        ],
    ) -> list[int]:
        prepared = []
        reserved_names = set(self._names)
        backend_shape = (
            self._backend_frame_data.shape[1:]
            if self._backend_frame_data is not None
            else None
        )
        for motion, name, weight in entries:
            item = self._prepare_motion(motion, name=name, weight=weight)
            if item.name in reserved_names:
                raise ValueError(f"duplicate motion name {item.name!r}")
            if item.backend_frame_data is not None:
                if backend_shape is None:
                    backend_shape = item.backend_frame_data.shape[1:]
                elif item.backend_frame_data.shape[1:] != backend_shape:
                    raise ValueError("adapter frame-data shape differs between clips")
            reserved_names.add(item.name)
            prepared.append(item)
        if not prepared:
            return []

        first_id = self.num_motions
        frame_offset = int(self._local_rotations.shape[0])
        frame_offsets = []
        for item in prepared:
            frame_offsets.append(frame_offset)
            frame_offset += item.count

        self._root_positions = torch.cat(
            (self._root_positions, *(item.root_positions for item in prepared))
        )
        self._root_linear_velocities = torch.cat(
            (
                self._root_linear_velocities,
                *(item.root_linear_velocities for item in prepared),
            )
        )
        self._root_angular_velocities = torch.cat(
            (
                self._root_angular_velocities,
                *(item.root_angular_velocities for item in prepared),
            )
        )
        self._local_rotations = torch.cat(
            (self._local_rotations, *(item.local_rotations_wxyz for item in prepared))
        )

        if self._precompute_kinematics:
            self._global_positions = self._pack_optional(
                self._global_positions,
                [item.global_positions for item in prepared],
            )
            self._global_rotations = self._pack_optional(
                self._global_rotations,
                [item.global_rotations_wxyz for item in prepared],
            )
            self._global_linear_velocities = self._pack_optional(
                self._global_linear_velocities,
                [item.global_linear_velocities for item in prepared],
            )
            self._global_angular_velocities = self._pack_optional(
                self._global_angular_velocities,
                [item.global_angular_velocities for item in prepared],
            )
        if self._adapter is not None:
            self._backend_frame_data = self._pack_optional(
                self._backend_frame_data,
                [item.backend_frame_data for item in prepared],
            )

        self._frame_offsets = self._pack_metadata(
            self._frame_offsets, frame_offsets, torch.long
        )
        self._frame_counts = self._pack_metadata(
            self._frame_counts, [item.count for item in prepared], torch.long
        )
        self._fps = self._pack_metadata(
            self._fps, [item.fps for item in prepared], torch.float32
        )
        self._durations = self._pack_metadata(
            self._durations, [item.duration for item in prepared], torch.float32
        )
        self._weights = self._pack_metadata(
            self._weights, [item.weight for item in prepared], torch.float32
        )
        self._names.extend(item.name for item in prepared)
        if self._source_motions is not None:
            self._source_motions.extend(item.skeleton_motion for item in prepared)
        return list(range(first_id, first_id + len(prepared)))

    def _prepare_motion(
        self,
        motion: SkeletonMotion | ArticulationMotion,
        *,
        name: str | None,
        weight: float,
    ) -> _PreparedMotion:
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
            self._local_rotations = torch.empty((0, len(names), 4), device=self._device)
        elif names != self._node_names or parents != self._parent_indices:
            raise ValueError("all MotionLibrary clips must share one skeleton topology")

        resolved_name = skeleton_motion.motion_name() if name is None else str(name)
        if weight < 0.0:
            raise ValueError("motion weight must be non-negative")

        roots = self._tensor(skeleton_motion.root_translations())
        root_linear = self._tensor(skeleton_motion.root_linear_velocities())
        rotations = self._tensor(skeleton_motion.local_rotations_wxyz())
        root_angular = self._angular_velocities(
            rotations[:, 0, :], skeleton_motion.fps()
        )
        global_positions = None
        global_rotations = None
        global_linear = None
        global_angular = None
        if self._precompute_kinematics:
            global_positions = self._tensor(skeleton_motion.global_positions())
            global_rotations = self._tensor(skeleton_motion.global_rotations_wxyz())
            global_linear = self._tensor(skeleton_motion.global_linear_velocities())
            global_angular = self._tensor(skeleton_motion.global_angular_velocities())

        backend_data = None
        if self._adapter is not None:
            backend_data = self._tensor(self._adapter.prepare_motion(motion))
            if backend_data.shape[0] != count:
                raise ValueError("adapter frame data must match the motion frame count")
        return _PreparedMotion(
            skeleton_motion=skeleton_motion,
            name=resolved_name,
            weight=weight,
            count=count,
            fps=skeleton_motion.fps(),
            duration=skeleton_motion.duration(),
            root_positions=roots,
            root_linear_velocities=root_linear,
            root_angular_velocities=root_angular,
            local_rotations_wxyz=rotations,
            global_positions=global_positions,
            global_rotations_wxyz=global_rotations,
            global_linear_velocities=global_linear,
            global_angular_velocities=global_angular,
            backend_frame_data=backend_data,
        )

    def packed_buffers(self) -> MotionLibraryBuffers:
        """Return tensor views suitable for persistent Torch/Warp interop."""

        return MotionLibraryBuffers(
            root_positions=self._root_positions,
            root_linear_velocities=self._root_linear_velocities,
            root_angular_velocities=self._root_angular_velocities,
            local_rotations_wxyz=self._local_rotations,
            frame_offsets=self._frame_offsets,
            frame_counts=self._frame_counts,
            fps=self._fps,
            durations=self._durations,
            weights=self._weights,
            global_positions=self._global_positions,
            global_rotations_wxyz=self._global_rotations,
            global_linear_velocities=self._global_linear_velocities,
            global_angular_velocities=self._global_angular_velocities,
            backend_frame_data=self._backend_frame_data,
        )

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

    def sample_motion_times(
        self, count: int, *, truncate_time: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample weighted motion IDs and valid times for those motions."""

        motion_ids = self.sample_motion_ids(count)
        return motion_ids, self.sample_times(
            motion_ids,
            truncate_time=truncate_time,
        )

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

    def sample_kinematics(
        self, sample: MotionSample[BackendSample]
    ) -> MotionKinematics:
        """Interpolate precomputed global transforms for a motion sample."""

        if self._global_positions is None or self._global_rotations is None:
            return self.forward_kinematics(sample)
        offsets = self._frame_offsets[sample.motion_ids]
        first = offsets + sample.frame_indices
        following = offsets + sample.next_frame_indices
        position_blend = sample.blend.unsqueeze(-1).unsqueeze(-1)
        positions = (
            self._global_positions[first]
            + (self._global_positions[following] - self._global_positions[first])
            * position_blend
        )
        rotations = quat_wxyz_slerp(
            self._global_rotations[first],
            self._global_rotations[following],
            sample.blend.unsqueeze(-1),
        )
        return MotionKinematics(
            body_positions=positions,
            body_rotations_wxyz=rotations,
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

    @staticmethod
    def _motion_file_entries(path: Path) -> list[tuple[Path, float]]:
        if path.suffix.lower() not in (".yaml", ".yml"):
            return [(path, 1.0)]
        with path.open("r", encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream)
        values = manifest.get("motions") if isinstance(manifest, dict) else None
        if not isinstance(values, list) or not values:
            raise ValueError("motion YAML requires a non-empty 'motions' list")

        entries = []
        total_weight = 0.0
        for value in values:
            if not isinstance(value, dict) or "file" not in value:
                raise ValueError("each motion YAML entry requires a file")
            motion_path = Path(value["file"]).expanduser()
            if not motion_path.is_absolute():
                motion_path = path.parent / motion_path
            weight = float(value.get("weight", 1.0))
            if weight < 0.0:
                raise ValueError("motion weights must be non-negative")
            entries.append((motion_path.resolve(), weight))
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError("at least one motion weight must be positive")
        return entries

    @staticmethod
    def _load_motion_file(
        path: Path,
        *,
        skeleton_tree: SkeletonTree | None,
        fallback_joint_names: Sequence[str] | None,
    ) -> SkeletonMotion:
        if path.suffix.lower() == ".bvh":
            source = _ke.asset.BVHLoader.load_motion(str(path))
            if skeleton_tree is None:
                return source
            return MotionLibrary._motion_on_skeleton(
                path,
                skeleton_tree,
                np.asarray(source.root_translations(), dtype=np.float32),
                np.asarray(source.local_rotations_wxyz(), dtype=np.float32),
                source.fps(),
                source.node_names(),
                fallback_joint_names,
            )
        if path.suffix.lower() == ".npy":
            if skeleton_tree is None:
                raise ValueError("NPY loading requires skeleton_tree")
            value = np.load(path, allow_pickle=True).item()
            names = MotionLibrary._npy_joint_names(value)
            root, rotations, fps = MotionLibrary._npy_arrays(value)
            return MotionLibrary._motion_on_skeleton(
                path,
                skeleton_tree,
                root,
                rotations,
                fps,
                names,
                fallback_joint_names,
            )
        raise ValueError(
            f"unsupported motion file {path}; provide a motion_loader for this format"
        )

    @staticmethod
    def _motion_on_skeleton(
        path: Path,
        skeleton_tree: SkeletonTree,
        root_positions: np.ndarray,
        local_rotations_wxyz: np.ndarray,
        fps: float,
        joint_names: Sequence[str] | None,
        fallback_joint_names: Sequence[str] | None,
    ) -> SkeletonMotion:
        target_names = list(skeleton_tree.node_names())
        names = list(joint_names) if joint_names is not None else None
        if names is None:
            if fallback_joint_names is None:
                if local_rotations_wxyz.shape[1] != len(target_names):
                    raise ValueError(
                        "unnamed motion rotations do not match the target skeleton"
                    )
                names = target_names
            else:
                names = [target_names[0], *fallback_joint_names]
        if len(names) != local_rotations_wxyz.shape[1]:
            raise ValueError("motion joint names do not match the rotation count")
        if len(set(names)) != len(names) or not set(names).issubset(target_names):
            raise ValueError("motion joint names do not match the target skeleton")

        rotations = np.zeros(
            (local_rotations_wxyz.shape[0], len(target_names), 4),
            dtype=np.float32,
        )
        rotations[..., 0] = 1.0
        for source_id, name in enumerate(names):
            rotations[:, target_names.index(name)] = local_rotations_wxyz[:, source_id]
        return SkeletonMotion.from_arrays(
            skeleton_tree,
            root_positions,
            rotations,
            fps,
            path.stem,
        )

    @staticmethod
    def _npy_arrays(value):
        if "rotation" in value and "root_translation" in value:
            rotations = np.asarray(value["rotation"]["arr"], dtype=np.float32)
            root = np.asarray(value["root_translation"]["arr"], dtype=np.float32)
            fps = float(value["fps"])
        else:
            required = {"rootTranslations", "localRotations", "frameRate"}
            missing = required.difference(value)
            if missing:
                raise ValueError(
                    "unsupported motion NPY; missing: " + ", ".join(sorted(missing))
                )
            root = np.asarray(value["rootTranslations"], dtype=np.float32)[:, [2, 0, 1]]
            rotations = np.asarray(value["localRotations"], dtype=np.float32)[
                ..., [0, 3, 1, 2]
            ]
            fps = float(np.asarray(value["frameRate"]).item())
        if fps <= 0.0:
            raise ValueError("motion frame rate must be positive")
        if rotations.ndim != 3 or rotations.shape[-1] != 4:
            raise ValueError("motion rotations must have shape (frames, joints, 4)")
        if root.shape != (rotations.shape[0], 3):
            raise ValueError("motion root positions must have shape (frames, 3)")
        return root, rotations, fps

    @staticmethod
    def _npy_joint_names(value) -> list[str] | None:
        skeleton = value.get("skeleton_tree")
        if not isinstance(skeleton, dict):
            return None
        names = skeleton.get("node_names")
        return list(names) if names is not None else None

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

    @staticmethod
    def _pack_optional(
        current: torch.Tensor | None,
        chunks: list[torch.Tensor | None],
    ) -> torch.Tensor:
        values = [chunk for chunk in chunks if chunk is not None]
        if len(values) != len(chunks):
            raise RuntimeError("prepared motion buffer is missing")
        if current is not None:
            values.insert(0, current)
        return torch.cat(values)

    def _pack_metadata(self, current, values, dtype) -> torch.Tensor:
        appended = torch.tensor(values, dtype=dtype, device=self._device)
        return torch.cat((current, appended))

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
