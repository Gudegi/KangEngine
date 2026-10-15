"""CPU feature construction with native and reference Motion Matching search.

- Walking features: joint positions (3P), velocities (3V), future planar
  positions (2H), and headings (2H), expressed in the current heading frame.
- Reference tracks leave the source skeleton, root motion, and skin binding intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from operator import index
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt

from ..._core import _ke
from ._arrays import (
    _readonly,
    _unit_quaternions,
    _validate_quaternions,
    _interpolate,
    _interpolate_rotation,
)
from .features import WalkingFeatures

SkeletonMotion = _ke.animation.SkeletonMotion
MotionMatch = _ke.animation.MotionMatch

_EPSILON = 1.0e-6
_SEARCH_CHUNK_SIZE = 8192


def _describe_features(builder, config) -> dict:
    """Validate a builder's JSON data contract before storing any features."""
    spec = json.loads(json.dumps(builder.describe(config), allow_nan=False))
    if not spec["type"]:
        raise ValueError("feature description requires a nonempty type")
    blocks = spec["blocks"]
    if not blocks:
        raise ValueError("feature description requires nonempty blocks")
    names = set()
    for block in blocks:
        name, size, weight = block["name"], block["size"], block["weight"]
        if not name or name in names:
            raise ValueError("feature block names must be nonempty and unique")
        names.add(name)
        if type(size) is not int or size < 0:
            raise ValueError("feature block sizes must be nonnegative integers")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("feature block weights must be finite and nonnegative")
    if not any(block["size"] > 0 and block["weight"] > 0 for block in blocks):
        raise ValueError(
            "at least one nonempty feature block must have positive weight"
        )
    return spec


def _validate_motion(motion, config, motion_id, skeleton_signature=None):
    label = f"motion {motion_id}"
    if not isinstance(motion, SkeletonMotion):
        raise TypeError(f"{label} must be SkeletonMotion")
    fps = float(motion.fps())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{label} fps must be finite and positive")
    names = tuple(motion.node_names())
    parents = tuple(motion.parent_indices())
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{label} must have nonempty, unique joint names")
    if (
        len(parents) != len(names)
        or parents[0] != -1
        or any(
            parent < 0 or parent >= joint for joint, parent in enumerate(parents[1:], 1)
        )
    ):
        raise ValueError(
            f"{label} must have one root at index zero and parent-first joints"
        )
    tree = motion.skeleton_tree
    offsets = np.array(
        [
            tuple(getattr(tree.local_translation(j), axis) for axis in "xyz")
            for j in range(len(names))
        ],
        dtype=np.float64,
    )
    rest_rotations = _unit_quaternions(
        [
            tuple(getattr(tree.local_rotation(j), axis) for axis in "wxyz")
            for j in range(len(names))
        ],
        f"{label} bind rotations",
    )
    if not np.isfinite(offsets).all():
        raise ValueError(f"{label} bind offsets must be finite")
    if skeleton_signature is None:
        skeleton_signature = (names, parents, offsets, rest_rotations)
    else:
        old_names, old_parents, old_offsets, old_rest = skeleton_signature
        if (
            names != old_names
            or parents != old_parents
            or offsets.shape != old_offsets.shape
            or rest_rotations.shape != old_rest.shape
        ):
            raise ValueError(f"{label} skeleton names and topology must match motion 0")
        signs = np.where(
            np.sum(old_rest * rest_rotations, axis=-1, keepdims=True) < 0, -1, 1
        )
        if not np.allclose(
            offsets, old_offsets, atol=1e-6, rtol=1e-5
        ) or not np.allclose(rest_rotations * signs, old_rest, atol=1e-6, rtol=1e-5):
            raise ValueError(
                f"{label} skeleton names, topology, bind offsets, and bind rotations must match motion 0"
            )
    roots = np.array(motion.root_translations(), dtype=np.float64, copy=True)
    rotations = np.asarray(motion.local_rotations_wxyz(), dtype=np.float64)
    frames = int(motion.num_frames())
    if roots.shape != (frames, 3) or rotations.shape != (frames, len(names), 4):
        raise ValueError(f"{label} has inconsistent root/rotation frame arrays")
    if not np.isfinite(roots).all():
        raise ValueError(f"{label} root translations must be finite")
    horizon_frames = config.trajectory_times[-1] * fps
    candidate_count = int(np.floor(frames - 1 - horizon_frames + 1e-7)) + 1
    # Even a sub-tolerance horizon needs a future sample; exclude the final frame.
    candidate_count = min(candidate_count, frames - 1)
    if frames < 2 or candidate_count <= 0:
        raise ValueError(
            f"{label} needs at least {config.trajectory_times[-1]:g} seconds of sampled motion for the future trajectory horizon"
        )
    _validate_quaternions(rotations, f"{label} local rotations")
    return fps, roots, frames, candidate_count, skeleton_signature


@dataclass(frozen=True)
class MotionMatchingConfig:
    """Walking defaults and clip timing shared by construction and queries.

    - Units: meters, seconds, m/s; convert source data before building the DB.
    - Joint positions retain height above the zero-height ground plane.
      World velocities are expressed in the heading frame, preserving speed.
    """

    position_joints: tuple[str, ...]
    """Joints whose positions form the pose feature block, in this order."""

    velocity_joints: tuple[str, ...]
    """Joints whose velocities form the pose feature block, in this order."""

    trajectory_times: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0, 1.0)
    """Positive, increasing future sample offsets in seconds."""

    up_axis: str = "Z"
    """Positive source up axis: Y or Z."""

    forward_axis: str = "X"
    """Positive source forward axis: X, Y, or Z; distinct from up_axis."""

    feature_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    """Squared-cost weights for joint position/velocity and trajectory position/direction.

    Zero disables a block; at least one nonempty block must have positive weight.
    """

    def __post_init__(self) -> None:
        for name in ("position_joints", "velocity_joints"):
            source = getattr(self, name)
            if isinstance(source, str):
                raise ValueError(f"{name} must be a sequence of joint names")
            names = tuple(source)
            if any(not isinstance(item, str) or not item for item in names):
                raise ValueError(f"{name} must contain nonempty joint names")
            if len(set(names)) != len(names):
                raise ValueError(f"{name} must not contain duplicate joint names")
            object.__setattr__(self, name, names)
        times = tuple(float(value) for value in self.trajectory_times)
        if (
            not times
            or not np.isfinite(times).all()
            or times[0] <= 0.0
            or any(after <= before for before, after in zip(times, times[1:]))
        ):
            raise ValueError(
                "trajectory_times must be finite, positive, and increasing"
            )
        object.__setattr__(self, "trajectory_times", times)
        if self.up_axis not in ("Y", "Z"):
            raise ValueError("up_axis must be 'Y' or 'Z'")
        if self.forward_axis not in ("X", "Y", "Z"):
            raise ValueError("forward_axis must be 'X', 'Y', or 'Z'")
        if self.forward_axis == self.up_axis:
            raise ValueError("forward_axis must be orthogonal to up_axis")
        weights = tuple(float(value) for value in self.feature_weights)
        if len(weights) != 4 or not np.isfinite(weights).all() or min(weights) < 0:
            raise ValueError(
                "feature_weights must contain four finite nonnegative values"
            )
        if not any(
            weight > 0 and size > 0
            for weight, size in zip(
                weights, (len(self.position_joints), len(self.velocity_joints), 1, 1)
            )
        ):
            raise ValueError(
                "at least one nonempty feature block must have positive weight"
            )
        object.__setattr__(self, "feature_weights", weights)


@dataclass(frozen=True, eq=False)
class _ClipFeatures:
    fps: float
    roots: np.ndarray
    headings: np.ndarray
    pose_features: np.ndarray
    candidate_count: int
    reference_positions: np.ndarray | None = None


@dataclass(frozen=True, eq=False, init=False)
class MotionMatchingDatabase:
    """Immutable clip-local features with native and NumPy exact search.

    - Public NumPy arrays are owned and read-only.
    - Walking order follows config joints/times; planar axes follow XYZ minus up.
    - features = (raw - mean) / scale * weights: mean is per dimension, scale is
      each block's mean standard deviation (1 for constant blocks), and weights
      repeat sqrt(block weight).
    - Custom builders own layout, coordinates, sampling, and query inputs;
      normalization and search stay common.
    """

    motions: tuple[SkeletonMotion, ...]
    config: MotionMatchingConfig
    features: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    motion_indices: np.ndarray
    frame_indices: np.ndarray
    _clips: tuple[_ClipFeatures, ...]
    _native: _ke.animation.MotionMatchingIndex
    _schema: dict
    _feature_builder: object
    _feature_spec: dict

    def save_npz(self, path: str | Path, *, write_summary: bool = False) -> Path:
        """Save source clips and precomputed features without Python objects.

        Args:
            path: Destination for the uncompressed NPZ; .npz is appended if needed.
            write_summary: Also write clip names, frame counts, and FPS to
                .summary.json. Loading ignores this file.

        Returns:
            The resolved path of the saved NPZ.
        """
        from ._io import save_database

        return save_database(self, path, write_summary=write_summary)

    @classmethod
    def from_npz(
        cls, path: str | Path, *, feature_builder=None
    ) -> MotionMatchingDatabase:
        """Restore without extracting features or recomputing normalization.

        Args:
            path: Source NPZ. Archives contain no executable code.
            feature_builder: Custom builder with matching describe(config) metadata.
                None restores WalkingFeatures and its saved filter settings.

        Returns:
            A database with restored clips and caches, and a rebuilt search index.

        Raises:
            ValueError: The archive version or feature schema is incompatible.
        """
        from ._io import load_database

        return load_database(cls, path, feature_builder=feature_builder)

    @classmethod
    def from_motions(
        cls,
        motions: Iterable[SkeletonMotion],
        *,
        config: MotionMatchingConfig,
        feature_builder=None,
    ) -> MotionMatchingDatabase:
        """Build features without joining velocity or trajectory across clips.

        Args:
            motions: Clips sharing names, topology, and bind transforms; FPS and
                q/-q may differ. Require finite data, unit WXYZ rotations within
                1e-3, one root in parent-first order, and a full future horizon.
            config: Clip timing and walking feature defaults.
            feature_builder: Defaults to WalkingFeatures. See matching.features
                for the builder contract; keep its settings unchanged after build.
                Explicit reference positions must be supplied for all clips or none.

        Returns:
            A database with normalized features and a native search index.

        Raises:
            ValueError: Clips are incompatible or motion/feature data is invalid.
        """
        if not isinstance(config, MotionMatchingConfig):
            raise TypeError("config must be MotionMatchingConfig")
        feature_builder = (
            WalkingFeatures() if feature_builder is None else feature_builder
        )
        feature_spec = _describe_features(feature_builder, config)
        blocks = feature_spec["blocks"]
        sizes = tuple(block["size"] for block in blocks)
        block_weights = tuple(block["weight"] for block in blocks)
        dimensions = sum(sizes)
        source_motions = tuple(motions)
        if not source_motions:
            raise ValueError("motions must contain at least one SkeletonMotion")
        clips: list[_ClipFeatures] = []
        feature_blocks: list[np.ndarray] = []
        skeleton_signature = None

        for motion_id, motion in enumerate(source_motions):
            label = f"motion {motion_id}"
            fps, roots, frames, candidate_count, skeleton_signature = _validate_motion(
                motion, config, motion_id, skeleton_signature
            )
            built = feature_builder.build_clip(
                motion, config=config, candidate_count=candidate_count
            )
            if len(built) not in (3, 4):
                raise ValueError(
                    "build_clip must return features, pose, rotations, and optionally reference positions"
                )
            raw, pose, headings = built[:3]
            reference_positions = None
            if len(built) == 4:
                reference_positions = np.asarray(built[3], dtype=np.float64)
                if (
                    reference_positions.shape != (frames, 3)
                    or not np.isfinite(reference_positions).all()
                ):
                    raise ValueError(
                        f"{label} reference positions must be finite with shape {(frames, 3)}"
                    )
                reference_positions = _readonly(reference_positions)
            if clips and (reference_positions is None) != (
                clips[0].reference_positions is None
            ):
                raise ValueError(
                    "all clips must use the same reference-position policy"
                )
            raw = np.asarray(raw, dtype=np.float64)
            pose = np.asarray(pose, dtype=np.float64)
            headings = _unit_quaternions(headings, "reference rotations")
            if raw.shape != (candidate_count, dimensions) or not np.isfinite(raw).all():
                raise ValueError(
                    f"{label} feature rows must be finite with shape {(candidate_count, dimensions)}"
                )
            if pose.ndim != 2 or len(pose) != frames or not np.isfinite(pose).all():
                raise ValueError(
                    f"{label} pose feature cache must be finite with shape [source frames, pose dimensions]"
                )
            if clips and pose.shape[1] != clips[0].pose_features.shape[1]:
                raise ValueError(
                    "all clips must share the pose feature cache dimension"
                )
            if headings.shape != (frames, 4):
                raise ValueError(
                    f"{label} reference rotations must have shape {(frames, 4)}"
                )
            clips.append(
                _ClipFeatures(
                    fps,
                    _readonly(roots),
                    _readonly(headings),
                    _readonly(pose),
                    candidate_count,
                    reference_positions,
                )
            )
            feature_blocks.append(raw)

        raw = np.concatenate(feature_blocks)
        mean = raw.mean(axis=0)
        scale = np.empty(raw.shape[1], dtype=np.float64)
        weights = np.empty_like(scale)
        offset = 0
        for size, weight in zip(sizes, block_weights):
            if size:
                block_std = float(raw[:, offset : offset + size].std(axis=0).mean())
                scale[offset : offset + size] = (
                    block_std if block_std > _EPSILON else 1.0
                )
                weights[offset : offset + size] = np.sqrt(weight)
            offset += size
        return cls._from_features(
            source_motions,
            config,
            feature_builder,
            feature_spec,
            clips,
            (raw - mean) / scale * weights,
            mean,
            scale,
            skeleton_signature,
        )

    @classmethod
    def _from_features(
        cls,
        source_motions,
        config,
        feature_builder,
        feature_spec,
        clips,
        features,
        mean,
        scale,
        skeleton_signature,
    ):
        """Assemble validated precomputed data without extracting features again."""
        blocks = feature_spec["blocks"]
        weights = np.concatenate(
            [np.full(block["size"], np.sqrt(block["weight"])) for block in blocks]
        )
        motion_ids = [
            np.full(clip.candidate_count, i, dtype=np.int64)
            for i, clip in enumerate(clips)
        ]
        frame_ids = [np.arange(clip.candidate_count) for clip in clips]
        result = object.__new__(cls)
        for key, value in {
            "motions": source_motions,
            "config": config,
            "features": _readonly(features),
            "mean": _readonly(mean),
            "scale": _readonly(scale),
            "weights": _readonly(weights),
            "motion_indices": _readonly(np.concatenate(motion_ids), np.int64),
            "frame_indices": _readonly(np.concatenate(frame_ids), np.int64),
            "_clips": tuple(clips),
            "_feature_builder": feature_builder,
            "_feature_spec": deepcopy(feature_spec),
        }.items():
            object.__setattr__(result, key, value)
        object.__setattr__(
            result,
            "_native",
            _ke.animation.MotionMatchingIndex(
                result.features,
                clip_frame_counts=[len(clip.roots) for clip in clips],
                clip_candidate_counts=[clip.candidate_count for clip in clips],
                clip_fps=[clip.fps for clip in clips],
            ),
        )
        object.__setattr__(result, "_schema", result._build_schema(skeleton_signature))
        if _describe_features(feature_builder, config) != feature_spec:
            raise ValueError("feature builder schema changed during construction")
        return result

    def _build_schema(self, skeleton_signature) -> dict:
        """Describe the assembled database for inspection and export."""
        blocks = self._feature_spec["blocks"]
        names, parents, offsets, rest = skeleton_signature
        # Canonical quaternion signs make the recorded rig independent of q/-q.
        rest = rest.copy()
        for rotation in rest:
            nonzero = np.flatnonzero(rotation)
            if rotation[nonzero[0]] < 0:
                rotation *= -1
        layout = []
        offset = 0
        for block in blocks:
            name, size = block["name"], block["size"]
            layout.append({"name": name, "offset": offset, "size": size})
            offset += size
        return {
            "format": "kangengine.motion_matching",
            "schema_version": 1,
            "feature_type": self._feature_spec["type"],
            "feature_dimension": int(self.features.shape[1]),
            "length_unit": "m",
            "time_unit": "s",
            "up_axis": self.config.up_axis,
            "forward_axis": self.config.forward_axis,
            "quaternion_order": "wxyz",
            "skeleton": {
                "joint_names": list(names),
                "parent_indices": list(parents),
                "rest_translations": offsets.tolist(),
                "rest_rotations": rest.tolist(),
            },
            "features": {
                **{
                    key: value
                    for key, value in self._feature_spec.items()
                    if key not in ("type", "blocks")
                },
                "layout": layout,
            },
            "normalization": {
                "formula": "(raw - mean) / scale * multipliers",
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
                "multipliers": self.weights.tolist(),
                "block_weights": [block["weight"] for block in blocks],
            },
            "clips": [
                {
                    "name": motion.motion_name(),
                    "fps": clip.fps,
                    "frame_count": len(clip.roots),
                    "candidate_count": clip.candidate_count,
                }
                for motion, clip in zip(self.motions, self._clips)
            ],
        }

    @property
    def num_frames(self) -> int:
        """Number of searchable source frames, excluding incomplete horizons."""
        return len(self.features)

    @property
    def native(self) -> _ke.animation.MotionMatchingIndex:
        """Shared native index with its own float64 copy; accepts normalized queries."""
        return self._native

    @property
    def schema(self) -> dict:
        """Detached JSON-compatible data contract for export/training tools.

        - Records rig, feature semantics, normalization, units (m, s), and clip clocks.
        - NPZ stores only the inputs needed to reconstruct this view.
        """
        return deepcopy(self._schema)

    def _clip(self, motion_index: int) -> _ClipFeatures:
        motion_index = index(motion_index)
        if motion_index < 0 or motion_index >= len(self._clips):
            raise ValueError("motion_index is outside the database")
        return self._clips[motion_index]

    @staticmethod
    def _frame_at_time(clip: _ClipFeatures, time: float) -> float:
        time = float(time)
        if (
            not np.isfinite(time)
            or time < 0
            or time > (len(clip.roots) - 1) / clip.fps + 1e-7
        ):
            raise ValueError("time must be within the sampled clip interval (seconds)")
        return min(time * clip.fps, len(clip.roots) - 1)

    def reference_quats_wxyz(self, motion_index: int) -> np.ndarray:
        """Builder-defined reference-to-source rotations [frames,4] in WXYZ."""
        return self._clip(motion_index).headings

    def root_positions(self, motion_index: int) -> np.ndarray:
        """Read-only full source root translations as [source frames, 3]."""
        return self._clip(motion_index).roots

    def reference_quat_wxyz(self, motion_index: int, time: float) -> np.ndarray:
        """Slerp the builder-defined reference rotation without looping."""
        clip = self._clip(motion_index)
        return _interpolate_rotation(clip.headings, self._frame_at_time(clip, time))

    def root_position(self, motion_index: int, time: float) -> np.ndarray:
        """Interpolate the full source root translation without looping."""
        clip = self._clip(motion_index)
        return _interpolate(clip.roots, self._frame_at_time(clip, time))

    @property
    def has_reference_track(self) -> bool:
        """Whether playback aligns to explicit reference positions/rotations."""
        return self._clips[0].reference_positions is not None

    def reference_positions(self, motion_index: int) -> np.ndarray:
        """Read-only reference positions; source roots when no track was supplied."""
        clip = self._clip(motion_index)
        return (
            clip.roots if clip.reference_positions is None else clip.reference_positions
        )

    def reference_position(self, motion_index: int, time: float) -> np.ndarray:
        """Interpolate the reference position without looping."""
        clip = self._clip(motion_index)
        return _interpolate(
            self.reference_positions(motion_index), self._frame_at_time(clip, time)
        )

    def make_query(
        self, motion_index: int, time: float, **feature_inputs
    ) -> np.ndarray:
        """Build a query with the same feature policy used to construct the DB.

        Args:
            motion_index: Source clip index.
            time: Clip-local seconds, including the unsearchable tail.
            **feature_inputs: Builder-specific inputs. Walking uses
                trajectory_positions/directions [H,3] in the reference frame;
                positions are relative displacements in meters. Pass world-space
                joint_positions/joint_velocities [J,3], skeleton_tree, and
                reference_position/reference_rotation_wxyz to use external state
                instead of sampling the source clip cache.

        Returns:
            A normalized, weighted feature vector [D] ready for search().
        """
        clip = self._clip(motion_index)
        frame = self._frame_at_time(clip, time)
        raw = np.asarray(
            self._feature_builder.make_query(
                clip.pose_features,
                frame=frame,
                config=self.config,
                **feature_inputs,
            ),
            dtype=np.float64,
        )
        if raw.shape != self.mean.shape or not np.isfinite(raw).all():
            raise ValueError(
                f"raw query must have shape {self.mean.shape} with finite values"
            )
        query = (raw - self.mean) / self.scale * self.weights
        if not np.isfinite(query).all():
            raise ValueError("feature magnitude produced a nonfinite query")
        return query

    def search(
        self,
        query: npt.ArrayLike,
        *,
        current_motion_index: int | None = None,
        current_time: float | None = None,
        continuation_bias: float = 0.0,
        allowed_motion_indices: Iterable[int] | None = None,
        backend: str = "native",
    ) -> MotionMatch:
        """Return the exact lowest weighted squared distance over all frames.

        Args:
            query: Normalized, weighted features [D], normally from make_query().
            current_motion_index: Current clip; supply together with current_time.
            current_time: Clip-local seconds for continuation bias.
            continuation_bias: Nonnegative cost reduction for the nearest current
                frame with a full horizon; requires both current arguments if nonzero.
            allowed_motion_indices: Eligible clips, including continuation.
                None allows all; an empty selection is invalid.
            backend: "native" or NumPy "python"; float64 roundoff may differ.

        Returns:
            The lowest-cost match; ties choose the first row. Cost includes bias.
        """
        if backend not in ("native", "python"):
            raise ValueError("backend must be 'native' or 'python'")
        allowed = (
            None
            if allowed_motion_indices is None
            else tuple(index(value) for value in allowed_motion_indices)
        )
        if current_motion_index is not None:
            current_motion_index = index(current_motion_index)
        if backend == "native":
            return self._native.search(
                query,
                current_motion_index=current_motion_index,
                current_time=current_time,
                continuation_bias=continuation_bias,
                allowed_motion_indices=allowed,
            )
        values = np.asarray(query, dtype=np.float64)
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError(
                f"query must have shape {self.mean.shape} with finite values"
            )
        continuation_bias = float(continuation_bias)
        if not np.isfinite(continuation_bias) or continuation_bias < 0:
            raise ValueError("continuation_bias must be finite and nonnegative")
        if (current_motion_index is None) != (current_time is None):
            raise ValueError(
                "current_motion_index and current_time must be supplied together"
            )
        if continuation_bias and current_motion_index is None:
            raise ValueError(
                "continuation_bias requires current_motion_index and current_time"
            )
        if allowed is not None:
            if not allowed:
                raise ValueError(
                    "allowed_motion_indices must select at least one motion"
                )
            for motion_index in allowed:
                self._clip(motion_index)
        if current_motion_index is not None:
            clip = self._clip(current_motion_index)
        continuation_row = -1
        if current_motion_index is not None:
            # Stable half-frame ties despite accumulated playback-clock roundoff.
            frame = int(np.floor(self._frame_at_time(clip, current_time) + 0.5 + 1e-7))
            if frame < clip.candidate_count:
                continuation_row = (
                    sum(
                        item.candidate_count
                        for item in self._clips[:current_motion_index]
                    )
                    + frame
                )
        best_row = 0
        best_cost = np.inf
        for start in range(0, self.num_frames, _SEARCH_CHUNK_SIZE):
            difference = self.features[start : start + _SEARCH_CHUNK_SIZE] - values
            costs = np.einsum("ij,ij->i", difference, difference)
            if start <= continuation_row < start + len(costs):
                costs[continuation_row - start] -= continuation_bias
            if allowed is not None:
                eligible = np.isin(
                    self.motion_indices[start : start + len(costs)], allowed
                )
                costs[~eligible] = np.inf
            local_row = int(np.argmin(costs))
            if costs[local_row] < best_cost:
                best_row, best_cost = start + local_row, float(costs[local_row])
        if not np.isfinite(best_cost):
            raise ValueError("query magnitude produced nonfinite squared distances")
        motion_id = int(self.motion_indices[best_row])
        frame_id = int(self.frame_indices[best_row])
        return MotionMatch(
            row=best_row,
            motion_index=motion_id,
            frame_index=frame_id,
            time=frame_id / self._clips[motion_id].fps,
            cost=best_cost,
        )
