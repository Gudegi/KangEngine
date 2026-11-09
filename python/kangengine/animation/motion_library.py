"""Motion storage and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index
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
# Newton adapter -> NewtonMotionState
# PhysX adapter  -> PhysXMotionState
# MuJoCo adapter -> MuJoCoMotionState


class MotionAdapter(Protocol[BackendSample]):
    """Simulator adapter consumed by :class:`MotionLibrary`."""

    def prepare_motion(
        self, motion: SkeletonMotion | ArticulationMotion
    ) -> torch.Tensor | np.ndarray:
        """Precompute frame data needed by simulator-native sampling."""

    def frame_buffers(self, data: torch.Tensor) -> BackendSample:
        """Expose prepared frame-major coordinates without copying."""

    def prepare_body_motion(self, data: np.ndarray, *, fps: float) -> SkeletonMotion:
        """Reconstruct robot poses from prepared CPU frames before upload."""

    def pack_sample(
        self,
        *,
        backend_frame_data: torch.Tensor,
        next_backend_frame_data: torch.Tensor,
        blend: torch.Tensor,
    ) -> BackendSample:
        """Interpolate prepared engine-coordinate frames on their device."""


@dataclass(frozen=True)
class SkeletonMotionState:
    """Skeleton pose and root velocities sampled on one device."""

    root_pos: torch.Tensor
    root_quat_wxyz: torch.Tensor
    root_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    local_quat_wxyz: torch.Tensor
    motion_ids: torch.Tensor
    frame_indices: torch.Tensor
    next_frame_indices: torch.Tensor
    blend: torch.Tensor


@dataclass(frozen=True)
class BodyMotionState:
    """World-space body poses and optional frame velocities.

    Positions and velocities use [..., bodies, 3]; WXYZ rotations use
    [..., bodies, 4]. Velocities require precomputed kinematics.
    The _w suffix denotes world coordinates; wxyz denotes quaternion order.
    Units are meters, meters/second, and radians/second.
    """

    body_pos_w: torch.Tensor
    body_quat_wxyz_w: torch.Tensor
    body_vel_w: torch.Tensor | None = None
    body_ang_vel_w: torch.Tensor | None = None


_BODY_FIELDS = ("body_pos_w", "body_quat_wxyz_w", "body_vel_w", "body_ang_vel_w")


@dataclass(frozen=True)
class MotionLibraryBuffers:
    """Read-only, frame-major tensors shared with the owning MotionLibrary.

    No data is copied. Reacquire views after adding clips or changing devices.
    """

    root_pos: torch.Tensor
    root_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    local_quat_wxyz: torch.Tensor
    frame_offsets: torch.Tensor
    frame_counts: torch.Tensor
    fps: torch.Tensor
    durations: torch.Tensor
    weights: torch.Tensor
    body_pos_w: torch.Tensor | None
    body_quat_wxyz_w: torch.Tensor | None
    body_vel_w: torch.Tensor | None
    body_ang_vel_w: torch.Tensor | None


@dataclass(frozen=True)
class _PreparedMotion:
    skeleton_motion: SkeletonMotion
    name: str
    weight: float
    count: int
    fps: float
    duration: float
    root_pos: torch.Tensor
    root_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    local_quat_wxyz: torch.Tensor
    source_body_frames: BodyMotionState | None
    backend_frame_data: torch.Tensor | None
    body_frames: BodyMotionState | None
    body_root_pos: torch.Tensor | None
    body_local_quat_wxyz: torch.Tensor | None


class MotionView(Generic[BackendSample]):
    """Access one clip's frames without copying its buffers.

    Keeps the library alive. Indexing returns (adapted motion, body motion)
    and requires an adapter. Negative indices count from the end; no slices.
    """

    def __init__(self, library: MotionLibrary[BackendSample], motion_id: int) -> None:
        self._library = library
        self._motion_id = motion_id

    def __len__(self) -> int:
        return self._library._frame_counts_cpu[self._motion_id]

    def __getitem__(self, frame_idx: int) -> tuple[BackendSample, BodyMotionState]:
        frame_idx = integer_index(frame_idx)
        count = len(self)
        if frame_idx < 0:
            frame_idx += count
        if not 0 <= frame_idx < count:
            raise IndexError("motion frame index out of range")
        return self._library.sample_robot_state(
            self._motion_id, frame_ids=frame_idx, loop=False
        )


class MotionLibrary(Generic[BackendSample]):
    """Store motion clips and sample skeleton, adapted, or body data on one device.

    Body tracks use adapter-converted poses when an adapter is present; source
    skeleton tracks remain unchanged. Disable precompute_kinematics to store
    local body poses and run FK when queried instead of caching world tracks."""

    def __init__(
        self,
        motions: list[SkeletonMotion | ArticulationMotion]
        | tuple[SkeletonMotion | ArticulationMotion, ...] = (),
        *,
        adapter: MotionAdapter[BackendSample] | None = None,
        weights: torch.Tensor | np.ndarray | list[float] | None = None,
        device: str | torch.device = "cpu",
        keep_source_motions: bool = False,
        precompute_kinematics: bool = True,
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
        self._robot_body_local_offsets: torch.Tensor | None = None
        self._fk_levels: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._root_pos = torch.empty((0, 3), device=self._device)
        self._root_vel = torch.empty((0, 3), device=self._device)
        self._root_ang_vel = torch.empty((0, 3), device=self._device)
        self._local_quat_wxyz = torch.empty((0, 0, 4), device=self._device)
        self._source_body_frames: BodyMotionState | None = None
        self._backend_frame_data: torch.Tensor | None = None
        self._body_frames: BodyMotionState | None = None
        self._body_root_pos: torch.Tensor | None = None
        self._body_local_quat_wxyz: torch.Tensor | None = None
        self._frame_offsets = torch.empty(0, dtype=torch.long, device=self._device)
        self._frame_counts = torch.empty(0, dtype=torch.long, device=self._device)
        self._frame_counts_cpu: list[int] = []
        self._fps = torch.empty(0, device=self._device)
        self._durations = torch.empty(0, device=self._device)
        self._weights = torch.empty(0, device=self._device)
        self._compiled: dict[tuple[str, type], Callable] | None = None

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
        precompute_kinematics: bool = True,
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
    def adapter(self) -> MotionAdapter[BackendSample] | None:
        """Adapter that owns this library's native coordinate layout."""
        return self._adapter

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
            "_root_pos",
            "_root_vel",
            "_root_ang_vel",
            "_local_quat_wxyz",
            "_body_root_pos",
            "_body_local_quat_wxyz",
            "_robot_body_local_offsets",
            "_frame_offsets",
            "_frame_counts",
            "_fps",
            "_durations",
            "_weights",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(resolved))
        self._fk_levels = [
            (ids.to(resolved), parents.to(resolved)) for ids, parents in self._fk_levels
        ]
        if self._backend_frame_data is not None:
            self._backend_frame_data = self._backend_frame_data.to(resolved)
        body_tracks = (
            ("_source_body_frames", "_body_frames")
            if self._adapter is not None
            else ("_source_body_frames",)
        )
        for name in body_tracks:
            frames = getattr(self, name)
            if frames is not None:
                setattr(
                    self,
                    name,
                    BodyMotionState(
                        *(getattr(frames, field).to(resolved) for field in _BODY_FIELDS)
                    ),
                )
        if self._adapter is None:
            self._body_frames = self._source_body_frames
        if self._compiled is not None:
            self._compiled.clear()
        return self

    def compile(self) -> None:
        """Compile sampling and FK on first use; disabled by default.

        Call after loading clips and choosing a device; first calls can take seconds.
        Uses torch.compile with dynamic batch sizes and fullgraph=True.
        Use device tensors; new ranks, dtypes, or query options may recompile.
        Adding clips or moving devices clears the cache but keeps compilation on.
        Compilation errors propagate without an eager fallback.
        """
        if self._compiled is None:
            self._compiled = {}

    def _evaluate(self, function, *args, **kwargs):
        if self._compiled is None:
            return function(*args, **kwargs)

        name = (function.__name__, type(args[0]))
        if name not in self._compiled:
            # Populate adapter metadata outside the full tensor graph.
            function(*args, **kwargs)
            self._compiled[name] = torch.compile(function, dynamic=True, fullgraph=True)
        return self._compiled[name](*args, **kwargs)

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
        frame_offset = int(self._local_quat_wxyz.shape[0])
        frame_offsets = []
        for item in prepared:
            frame_offsets.append(frame_offset)
            frame_offset += item.count

        self._root_pos = torch.cat(
            (self._root_pos, *(item.root_pos for item in prepared))
        )
        self._root_vel = torch.cat(
            (
                self._root_vel,
                *(item.root_vel for item in prepared),
            )
        )
        self._root_ang_vel = torch.cat(
            (
                self._root_ang_vel,
                *(item.root_ang_vel for item in prepared),
            )
        )
        self._local_quat_wxyz = torch.cat(
            (self._local_quat_wxyz, *(item.local_quat_wxyz for item in prepared))
        )

        if self._precompute_kinematics:
            self._source_body_frames = self._pack_body_frames(
                self._source_body_frames, [item.source_body_frames for item in prepared]
            )
        if self._adapter is not None:
            self._backend_frame_data = self._pack_optional(
                self._backend_frame_data,
                [item.backend_frame_data for item in prepared],
            )
            if self._precompute_kinematics:
                self._body_frames = self._pack_body_frames(
                    self._body_frames, [item.body_frames for item in prepared]
                )
            else:
                self._body_root_pos = self._pack_optional(
                    self._body_root_pos, [item.body_root_pos for item in prepared]
                )
                self._body_local_quat_wxyz = self._pack_optional(
                    self._body_local_quat_wxyz,
                    [item.body_local_quat_wxyz for item in prepared],
                )
        else:
            self._body_frames = self._source_body_frames

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
        self._frame_counts_cpu.extend(item.count for item in prepared)
        if self._source_motions is not None:
            self._source_motions.extend(item.skeleton_motion for item in prepared)
        if self._compiled is not None:
            self._compiled.clear()
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
            depths = [0] * len(parents)
            levels = []
            for body_idx, parent in enumerate(parents[1:], start=1):
                if parent < 0 or parent >= body_idx:
                    raise RuntimeError(
                        "MotionLibrary FK requires parent-before-child joint order"
                    )
                depths[body_idx] = depths[parent] + 1
                if len(levels) < depths[body_idx]:
                    levels.append([])
                levels[depths[body_idx] - 1].append(body_idx)
            self._fk_levels = [
                (
                    torch.tensor(ids, dtype=torch.long, device=self._device),
                    torch.tensor(
                        [parents[idx] for idx in ids],
                        dtype=torch.long,
                        device=self._device,
                    ),
                )
                for ids in levels
            ]
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
            self._local_quat_wxyz = torch.empty((0, len(names), 4), device=self._device)
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
        source_body_frames = (
            self._prepare_body_frames(skeleton_motion)
            if self._precompute_kinematics
            else None
        )

        backend_data = None
        body_frames = None
        body_root_pos = None
        body_local_quat_wxyz = None
        if self._adapter is not None:
            # Project the motion onto the robot's joint axes before computing body tracks.
            native_frames = self._adapter.prepare_motion(motion)
            body_motion = self._adapter.prepare_body_motion(
                native_frames, fps=skeleton_motion.fps()
            )
            if self._robot_body_local_offsets is None:
                tree = body_motion.skeleton_tree
                self._robot_body_local_offsets = self._tensor(
                    [
                        [getattr(tree.local_translation(idx), axis) for axis in "xyz"]
                        for idx in range(len(names))
                    ]
                )
            backend_data = self._tensor(native_frames)
            if backend_data.shape[0] != count:
                raise ValueError("adapter frame data must match the motion frame count")
            if self._precompute_kinematics:
                body_frames = self._prepare_body_frames(body_motion)
            else:
                body_root_pos = self._tensor(body_motion.root_translations())
                body_local_quat_wxyz = self._tensor(body_motion.local_rotations_wxyz())
        return _PreparedMotion(
            skeleton_motion=skeleton_motion,
            name=resolved_name,
            weight=weight,
            count=count,
            fps=skeleton_motion.fps(),
            duration=skeleton_motion.duration(),
            root_pos=roots,
            root_vel=root_linear,
            root_ang_vel=root_angular,
            local_quat_wxyz=rotations,
            source_body_frames=source_body_frames,
            backend_frame_data=backend_data,
            body_frames=body_frames,
            body_root_pos=body_root_pos,
            body_local_quat_wxyz=body_local_quat_wxyz,
        )

    def source_buffers(self) -> MotionLibraryBuffers:
        """Borrow source skeleton frames, source body frames, and clip metadata.

        Body frames are source FK results before projection onto robot joints.
        These are library-owned tensors, not the original input-file arrays.
        Reacquire the view after adding clips or moving the library's device.
        """

        return MotionLibraryBuffers(
            root_pos=self._root_pos,
            root_vel=self._root_vel,
            root_ang_vel=self._root_ang_vel,
            local_quat_wxyz=self._local_quat_wxyz,
            frame_offsets=self._frame_offsets,
            frame_counts=self._frame_counts,
            fps=self._fps,
            durations=self._durations,
            weights=self._weights,
            **{
                field: getattr(self._source_body_frames, field)
                if self._source_body_frames is not None
                else None
                for field in _BODY_FIELDS
            },
        )

    def body_buffers(self) -> BodyMotionState:
        """Borrow precomputed world body poses and velocities in skeleton body order.

        With an adapter, FK uses motion projected onto the robot's joints;
        otherwise it uses the source skeleton. This call does not run FK.
        Use sample_robot_state() for time-sampled joint and body targets.
        Adding clips or moving devices invalidates these read-only views.
        Requires precompute_kinematics=True.
        """
        if self._body_frames is None:
            raise RuntimeError("body buffers require precomputed kinematics and a clip")
        return self._body_frames

    def set_weights(self, weights: torch.Tensor | np.ndarray | list[float]) -> None:
        """Replace motion-selection weights."""

        values = torch.as_tensor(weights, dtype=torch.float32, device=self._device)
        if values.shape != (self.num_motions,):
            raise ValueError(f"weights expected shape [{self.num_motions}]")
        if bool(torch.any(values < 0.0)) or float(values.sum()) <= 0.0:
            raise ValueError("weights must be non-negative with a positive sum")
        self._weights = values

    def motion(self, motion_id: int) -> MotionView[BackendSample]:
        """Reference one clip without copying; index it to sample robot frames."""
        return MotionView(self, self._validate_motion_id(motion_id))

    def source_motion(self, motion_id: int) -> SkeletonMotion:
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
        """Sample weighted motion IDs and valid times for those motions.

        truncate_time must be nonnegative and no greater than any selectable
        clip's duration. This input contract is not checked during sampling.
        """

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
        """Sample random times in [0, duration - truncate_time).

        Args:
            motion_ids: Valid clip IDs in [0, num_motions).
            truncate_time: Seconds reserved at the end of each clip. Must lie in
                [0, duration] for every selected clip; not checked during sampling.

        Returns:
            Times with the same shape as motion_ids on the library device.
            When truncate_time equals a clip's duration, its sampled time is zero.
        """

        ids = self._motion_ids(motion_ids)
        durations = self._durations[ids]
        return torch.rand_like(durations) * (durations - truncate_time)

    def sample_skel_state(
        self,
        motion_ids: torch.Tensor | np.ndarray | list[int] | int,
        times: torch.Tensor | np.ndarray | list[float] | float | None = None,
        *,
        frame_ids: torch.Tensor | np.ndarray | list[int] | int | None = None,
        loop: bool = True,
        interpolate: bool = True,
    ) -> SkeletonMotionState:
        """Sample skeleton poses and root velocities without adapter conversion.

        Args:
            motion_ids: IDs in [0, num_motions), broadcastable with times or frame_ids.
            times: Times in seconds. Supply exactly one of times and frame_ids.
            frame_ids: Integer clip-local frames, selected without interpolation.
            loop: Wrap times/frames; otherwise clamp to the clip endpoints.
            interpolate: Interpolate time queries; ignored for explicit frames.

        Returns:
            Root positions [..., 3], local WXYZ rotations [..., joints, 4],
            and world-space root velocities [..., 3] on the library device.
            Velocities interpolate linearly for interpolated time queries.
        """
        return self._evaluate(
            self._query_skel_state,
            motion_ids,
            times,
            frame_ids=frame_ids,
            loop=loop,
            interpolate=interpolate,
        )

    def _query_skel_state(
        self, motion_ids, times, *, frame_ids=None, loop=True, interpolate=True
    ):
        frames = self._resolve_frames(
            motion_ids, times, frame_ids=frame_ids, loop=loop, interpolate=interpolate
        )
        return self._sample_skel_state(*frames)

    def sample_arti_state(
        self,
        motion_ids: torch.Tensor | np.ndarray | list[int] | int,
        times: torch.Tensor | np.ndarray | list[float] | float | None = None,
        *,
        frame_ids: torch.Tensor | np.ndarray | list[int] | int | None = None,
        loop: bool = True,
        interpolate: bool = True,
    ) -> BackendSample:
        """Sample adapter-native coordinates and velocities, including a free root.

        Args:
            motion_ids: IDs in [0, num_motions), broadcastable with times or frame_ids.
            times: Times in seconds. Supply exactly one of times and frame_ids.
            frame_ids: Integer clip-local frames, selected without interpolation.
            loop: Wrap times/frames; otherwise clamp to the clip endpoints.
            interpolate: Interpolate time queries; ignored for explicit frames.

        Returns:
            Adapter-native tensors. The adapter defines coordinate order and
            velocity frames. Requires an adapter.
        """
        return self._evaluate(
            self._query_arti_state,
            motion_ids,
            times,
            frame_ids=frame_ids,
            loop=loop,
            interpolate=interpolate,
        )

    def _query_arti_state(
        self, motion_ids, times, *, frame_ids=None, loop=True, interpolate=True
    ):
        frames = self._resolve_frames(
            motion_ids, times, frame_ids=frame_ids, loop=loop, interpolate=interpolate
        )
        return self._sample_arti_state(*frames)

    def sample_body_state(
        self,
        motion_ids: torch.Tensor | np.ndarray | list[int] | int,
        times: torch.Tensor | np.ndarray | list[float] | float | None = None,
        *,
        frame_ids: torch.Tensor | np.ndarray | list[int] | int | None = None,
        loop: bool = True,
        interpolate: bool = True,
    ) -> BodyMotionState:
        """Sample world body tracks prepared from robot poses, or source poses without an adapter.

        Args:
            motion_ids: IDs in [0, num_motions), broadcastable with times or frame_ids.
            times: Times in seconds. Supply exactly one of times and frame_ids.
            frame_ids: Integer clip-local frames, selected without interpolation.
            loop: Wrap times/frames; otherwise clamp to the clip endpoints.
            interpolate: Interpolate time queries; ignored for explicit frames.

        Returns:
            Body frames are computed from converted robot poses at load time.
            World poses and velocities interpolate between those frames;
            interpolated body poses need not equal FK of interpolated joints.
            With precompute_kinematics=False, interpolated local robot poses
            are evaluated with FK and body velocities are None.
        """
        return self._evaluate(
            self._query_body_state,
            motion_ids,
            times,
            frame_ids=frame_ids,
            loop=loop,
            interpolate=interpolate,
        )

    def _query_body_state(
        self, motion_ids, times, *, frame_ids=None, loop=True, interpolate=True
    ):
        frames = self._resolve_frames(
            motion_ids, times, frame_ids=frame_ids, loop=loop, interpolate=interpolate
        )
        return self._sample_body_state(*frames)

    def sample_robot_state(
        self,
        motion_ids: torch.Tensor | np.ndarray | list[int] | int,
        times: torch.Tensor | np.ndarray | list[float] | float | None = None,
        *,
        frame_ids: torch.Tensor | np.ndarray | list[int] | int | None = None,
        loop: bool = True,
        interpolate: bool = True,
    ) -> tuple[BackendSample, BodyMotionState]:
        """Return (adapted motion, body motion) using one shared frame lookup.

        Args:
            motion_ids: IDs in [0, num_motions), broadcastable with times or frame_ids.
            times: Times in seconds. Supply exactly one of times and frame_ids.
            frame_ids: Integer clip-local frames, selected without interpolation.
            loop: Wrap times/frames; otherwise clamp to the clip endpoints.
            interpolate: Interpolate time queries; ignored for explicit frames.

        Returns:
            Body tracks follow sample_body_state(): converted robot frames,
            interpolated in world space by default. Disabling precomputation
            uses FK of interpolated local robot poses and no body velocities.
            The first result uses adapter-native coordinates; requires an adapter.
        """
        return self._evaluate(
            self._query_robot_state,
            motion_ids,
            times,
            frame_ids=frame_ids,
            loop=loop,
            interpolate=interpolate,
        )

    def _query_robot_state(
        self, motion_ids, times, *, frame_ids=None, loop=True, interpolate=True
    ):
        frames = self._resolve_frames(
            motion_ids, times, frame_ids=frame_ids, loop=loop, interpolate=interpolate
        )
        return self._sample_arti_state(*frames), self._sample_body_state(*frames)

    def adapted_buffers(self) -> BackendSample:
        """Borrow motion frames converted to the engine's joint order and coordinates.

        Requires an adapter. Fields follow its native state format.
        Treat them as read-only. Adding clips or moving devices invalidates them.
        """
        if self._adapter is None or self._backend_frame_data is None:
            raise RuntimeError("adapted buffers require a motion adapter")
        return self._adapter.frame_buffers(self._backend_frame_data)

    def _resolve_frames(
        self, motion_ids, times, *, frame_ids=None, loop=True, interpolate=True
    ):
        if (times is None) == (frame_ids is None):
            raise ValueError("supply exactly one of times and frame_ids")
        ids = self._motion_ids(motion_ids)
        if frame_ids is not None:
            frames = torch.as_tensor(frame_ids, device=self._device)
            if (
                frames.dtype.is_floating_point
                or frames.dtype.is_complex
                or frames.dtype == torch.bool
            ):
                raise TypeError("frame_ids must contain integers")
            ids, frames = torch.broadcast_tensors(ids, frames.long())
            counts = self._frame_counts[ids]
            resolved = torch.remainder(frames, counts) if loop else frames.clamp_min(0)
            if not loop:
                resolved = torch.minimum(resolved, counts - 1)
            # Sharing the index tensor marks exact queries for the samplers.
            return ids, resolved, resolved, torch.zeros_like(ids, dtype=torch.float32)

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
        return ids, first, following, blend

    def _sample_arti_state(self, ids, first, following, blend):
        if self._adapter is None or self._backend_frame_data is None:
            raise RuntimeError("articulation sampling requires a motion adapter")
        offsets = self._frame_offsets[ids]
        if first is following:
            return self._adapter.frame_buffers(
                self._backend_frame_data[offsets + first]
            )
        return self._adapter.pack_sample(
            backend_frame_data=self._backend_frame_data[offsets + first],
            next_backend_frame_data=self._backend_frame_data[offsets + following],
            blend=blend,
        )

    def forward_kinematics(
        self, sample: SkeletonMotionState | BackendSample
    ) -> BodyMotionState:
        """Compute world body poses from a skeleton or supported adapter state.

        PhysXMotionState uses the library adapter's logical joint order. Other
        adapter states are currently unsupported. Leading batch dimensions and
        device are preserved; body velocities are None.
        """
        return self._evaluate(self._forward_kinematics, sample)

    def _forward_kinematics(self, sample):
        # TODO: Implement PhysX-state batched FK in C++ as well.

        if self._parent_indices is None:
            raise RuntimeError("cannot run FK on an empty MotionLibrary")
        if isinstance(sample, SkeletonMotionState):
            root_pos, rotations = sample.root_pos, sample.local_quat_wxyz
            offsets = self._body_local_offsets
        else:
            to_pose = getattr(self._adapter, "to_skeleton_pose", None)
            if to_pose is None:
                raise NotImplementedError(
                    "this motion adapter does not support state FK"
                )
            root_pos, rotations = to_pose(sample)
            offsets = self._robot_body_local_offsets

        return self._pose_fk(root_pos, rotations, offsets)

    def _pose_fk(self, root_pos, rotations, offsets):
        # Parents are complete before all children at the next depth are evaluated.
        body_pos_w = root_pos.new_empty((*rotations.shape[:-1], 3))
        body_rotations = torch.empty_like(rotations)
        body_pos_w[..., 0, :] = root_pos
        body_rotations[..., 0, :] = rotations[..., 0, :]
        for ids, parents in self._fk_levels:
            parent_rotation = body_rotations[..., parents, :]
            offset = quat_wxyz_rotate(parent_rotation, offsets[ids])
            body_pos_w[..., ids, :] = body_pos_w[..., parents, :] + offset
            body_rotations[..., ids, :] = quat_wxyz_multiply(
                parent_rotation, rotations[..., ids, :]
            )
        return BodyMotionState(body_pos_w=body_pos_w, body_quat_wxyz_w=body_rotations)

    def _sample_body_state(self, ids, first, following, blend):
        frames = self._body_frames
        if frames is None:
            if self._adapter is not None:
                start = self._frame_offsets[ids] + first
                end = self._frame_offsets[ids] + following
                roots = torch.lerp(
                    self._body_root_pos[start],
                    self._body_root_pos[end],
                    blend.unsqueeze(-1),
                )
                rotations = quat_wxyz_slerp(
                    self._body_local_quat_wxyz[start],
                    self._body_local_quat_wxyz[end],
                    blend.unsqueeze(-1),
                )
                return self._pose_fk(roots, rotations, self._robot_body_local_offsets)
            return self._forward_kinematics(
                self._sample_skel_state(ids, first, following, blend)
            )
        positions, rotations = frames.body_pos_w, frames.body_quat_wxyz_w
        velocities, angular = frames.body_vel_w, frames.body_ang_vel_w
        offsets = self._frame_offsets[ids]
        if first is following:
            frame = offsets + first
            return BodyMotionState(
                positions[frame],
                rotations[frame],
                velocities[frame],
                angular[frame],
            )
        first = offsets + first
        following = offsets + following
        position_blend = blend.unsqueeze(-1).unsqueeze(-1)
        return BodyMotionState(
            body_pos_w=torch.lerp(
                positions[first], positions[following], position_blend
            ),
            body_quat_wxyz_w=quat_wxyz_slerp(
                rotations[first], rotations[following], blend.unsqueeze(-1)
            ),
            body_vel_w=torch.lerp(
                velocities[first], velocities[following], position_blend
            ),
            body_ang_vel_w=torch.lerp(
                angular[first],
                angular[following],
                position_blend,
            ),
        )

    def _sample_skel_state(self, ids, first, following, blend):
        offsets = self._frame_offsets[ids]
        first_packed = offsets + first
        next_packed = offsets + following
        if first is following:
            roots = self._root_pos[first_packed]
            local_rotations = self._local_quat_wxyz[first_packed]
            root_linear = self._root_vel[first_packed]
            root_angular = self._root_ang_vel[first_packed]
        else:
            blend_column = blend.unsqueeze(-1)
            roots = torch.lerp(
                self._root_pos[first_packed], self._root_pos[next_packed], blend_column
            )
            local_rotations = quat_wxyz_slerp(
                self._local_quat_wxyz[first_packed],
                self._local_quat_wxyz[next_packed],
                blend.unsqueeze(-1),
            )
            root_linear = torch.lerp(
                self._root_vel[first_packed], self._root_vel[next_packed], blend_column
            )
            root_angular = torch.lerp(
                self._root_ang_vel[first_packed],
                self._root_ang_vel[next_packed],
                blend_column,
            )
        root_rotations = local_rotations[..., 0, :]
        return SkeletonMotionState(
            root_pos=roots,
            root_quat_wxyz=root_rotations,
            root_vel=root_linear,
            root_ang_vel=root_angular,
            local_quat_wxyz=local_rotations,
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
        root_pos: np.ndarray,
        local_quat_wxyz: np.ndarray,
        fps: float,
        joint_names: Sequence[str] | None,
        fallback_joint_names: Sequence[str] | None,
    ) -> SkeletonMotion:
        target_names = list(skeleton_tree.node_names())
        names = list(joint_names) if joint_names is not None else None
        if names is None:
            if fallback_joint_names is None:
                if local_quat_wxyz.shape[1] != len(target_names):
                    raise ValueError(
                        "unnamed motion rotations do not match the target skeleton"
                    )
                names = target_names
            else:
                names = [target_names[0], *fallback_joint_names]
        if len(names) != local_quat_wxyz.shape[1]:
            raise ValueError("motion joint names do not match the rotation count")
        if len(set(names)) != len(names) or not set(names).issubset(target_names):
            raise ValueError("motion joint names do not match the target skeleton")

        rotations = np.zeros(
            (local_quat_wxyz.shape[0], len(target_names), 4),
            dtype=np.float32,
        )
        rotations[..., 0] = 1.0
        for source_id, name in enumerate(names):
            rotations[:, target_names.index(name)] = local_quat_wxyz[:, source_id]
        return SkeletonMotion.from_arrays(
            skeleton_tree,
            root_pos,
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

    def _prepare_body_frames(self, motion: SkeletonMotion) -> BodyMotionState:
        return BodyMotionState(
            self._tensor(motion.global_positions()),
            self._tensor(motion.global_rotations_wxyz()),
            self._tensor(motion.global_linear_velocities()),
            self._tensor(motion.global_angular_velocities()),
        )

    def _pack_body_frames(
        self, current: BodyMotionState | None, chunks: list[BodyMotionState]
    ) -> BodyMotionState:
        return BodyMotionState(
            *(
                self._pack_optional(
                    getattr(current, field) if current is not None else None,
                    [getattr(chunk, field) for chunk in chunks],
                )
                for field in _BODY_FIELDS
            )
        )

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
        # Trust clip IDs during batch queries to avoid CUDA-to-host synchronization.
        return torch.as_tensor(motion_ids, dtype=torch.long, device=self._device)

    def _validate_motion_id(self, motion_id: int) -> int:
        motion_id = int(motion_id)
        if motion_id < 0 or motion_id >= self.num_motions:
            raise IndexError(f"motion_id {motion_id} is out of range")
        return motion_id
