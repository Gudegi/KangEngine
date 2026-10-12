"""NPZ source data and precomputed feature caches."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..._core import _ke
from ._arrays import _readonly, _validate_quaternions
from .database import (
    MotionMatchingConfig,
    _ClipFeatures,
    _describe_features,
    _validate_motion,
)
from .features import WalkingFeatures

_FORMAT = "kangengine.motion_matching"
_VERSION = 1


def save_database(database, path, *, write_summary: bool = False) -> Path:
    if (
        _describe_features(database._feature_builder, database.config)
        != database._feature_spec
    ):
        raise ValueError("feature builder schema changed; rebuild the database")
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        destination = destination.with_name(destination.name + ".npz")

    arrays = {
        "features": np.asarray(database.features, dtype="<f8"),
        "mean": np.asarray(database.mean, dtype="<f8"),
        "scale": np.asarray(database.scale, dtype="<f8"),
        "parents": np.asarray(database.motions[0].parent_indices(), dtype="<i8"),
    }
    rest_translations, rest_rotations, rig_ids, clips = [], [], {}, []
    for i, (motion, cache) in enumerate(zip(database.motions, database._clips)):
        tree = motion.skeleton_tree
        offsets = np.array(
            [
                [getattr(tree.local_translation(j), axis) for axis in "xyz"]
                for j in range(len(motion.node_names()))
            ],
            dtype="<f4",
        )
        rotations = np.array(
            [
                [getattr(tree.local_rotation(j), axis) for axis in "wxyz"]
                for j in range(len(motion.node_names()))
            ],
            dtype="<f4",
        )
        # Deduplicate exact rigs; preserve compatible but distinct bind transforms.
        identity = (offsets.tobytes(), rotations.tobytes())
        if identity not in rig_ids:
            rig_ids[identity] = len(rest_translations)
            rest_translations.append(offsets)
            rest_rotations.append(rotations)
        clips.append(
            {
                "name": motion.motion_name(),
                "fps": cache.fps,
                "skeleton": rig_ids[identity],
            }
        )
        arrays[f"clip_{i}_roots"] = np.asarray(motion.root_translations(), dtype="<f4")
        arrays[f"clip_{i}_rotations"] = np.asarray(
            motion.local_rotations_wxyz(), dtype="<f4"
        )
        arrays[f"clip_{i}_pose_features"] = np.asarray(cache.pose_features, dtype="<f8")
        arrays[f"clip_{i}_reference_rotations"] = np.asarray(
            cache.headings, dtype="<f8"
        )
        if database.has_reference_track:
            arrays[f"clip_{i}_reference_positions"] = np.asarray(
                cache.reference_positions, dtype="<f8"
            )
    arrays["rest_translations"] = np.asarray(np.stack(rest_translations), dtype="<f4")
    arrays["rest_rotations"] = np.asarray(np.stack(rest_rotations), dtype="<f4")
    metadata = {
        "format": _FORMAT,
        "version": _VERSION,
        "has_reference_track": database.has_reference_track,
        "config": asdict(database.config),
        "feature_spec": database._feature_spec,
        "joint_names": list(database.motions[0].node_names()),
        "clips": clips,
    }
    arrays["metadata"] = np.frombuffer(
        json.dumps(
            metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8"),
        dtype=np.uint8,
    )
    # A failed write must not replace an existing valid database.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".npz", delete=False
        ) as file:
            temporary = Path(file.name)
            np.savez(file, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if write_summary:
        summary = {
            "clips": [
                {
                    "name": motion.motion_name(),
                    "num_frames": motion.num_frames(),
                    "fps": motion.fps(),
                }
                for motion in database.motions
            ]
        }
        destination.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return destination


def _array(archive, key, dtype, shape):
    value = archive[key]
    if (
        value.dtype != np.dtype(dtype)
        or value.ndim != len(shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape)
        )
    ):
        raise ValueError(f"{key} must have dtype {dtype} and shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} must contain finite values")
    return value


def load_database(cls, path, *, feature_builder=None):
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        if metadata["format"] != _FORMAT or metadata["version"] != _VERSION:
            raise ValueError("unsupported motion matching archive format/version")
        config = MotionMatchingConfig(**metadata["config"])
        has_reference_track = metadata["has_reference_track"]
        builder = feature_builder
        if builder is None:
            builder = WalkingFeatures(
                **metadata["feature_spec"].get("reference_filter", {})
            )
        spec = _describe_features(builder, config)
        if spec != metadata["feature_spec"]:
            raise ValueError(
                "feature schema mismatch; supply the matching feature_builder"
            )
        if "reference_filter" in spec and not has_reference_track:
            raise ValueError("filtered walking features require reference positions")
        names, clip_info = metadata["joint_names"], metadata["clips"]
        if not clip_info:
            raise ValueError("archive must contain at least one clip")
        joints = len(names)
        parents = _array(archive, "parents", "<i8", (joints,))
        offsets = _array(archive, "rest_translations", "<f4", (None, joints, 3))
        rest = _array(archive, "rest_rotations", "<f4", (len(offsets), joints, 4))
        _validate_quaternions(rest, "rest_rotations")
        trees = [
            _ke.animation.SkeletonTree(names, parents.tolist(), p, q)
            for p, q in zip(offsets, rest)
        ]
        motions, clips = [], []
        signature = None
        for i, info in enumerate(clip_info):
            if not 0 <= info["skeleton"] < len(trees):
                raise ValueError(f"invalid skeleton index for clip {i}")
            roots = _array(archive, f"clip_{i}_roots", "<f4", (None, 3))
            rotations = _array(
                archive, f"clip_{i}_rotations", "<f4", (len(roots), joints, 4)
            )
            motion = _ke.animation.SkeletonMotion.from_arrays(
                trees[info["skeleton"]], roots, rotations, info["fps"], info["name"]
            )
            fps, _, frames, count, signature = _validate_motion(
                motion, config, i, signature
            )
            if fps != info["fps"]:
                raise ValueError(
                    f"clip {i} fps cannot be represented by the native motion"
                )
            pose = _array(
                archive,
                f"clip_{i}_pose_features",
                "<f8",
                (frames, clips[0].pose_features.shape[1] if clips else None),
            )
            reference = _array(
                archive, f"clip_{i}_reference_rotations", "<f8", (frames, 4)
            )
            _validate_quaternions(reference, f"clip {i} reference rotations")
            reference_positions = None
            if has_reference_track:
                reference_positions = _readonly(
                    _array(archive, f"clip_{i}_reference_positions", "<f8", (frames, 3))
                )
            motions.append(motion)
            clips.append(
                _ClipFeatures(
                    fps,
                    _readonly(roots),
                    _readonly(reference),
                    _readonly(pose),
                    count,
                    reference_positions,
                )
            )
        dimension = sum(block["size"] for block in spec["blocks"])
        rows = sum(clip.candidate_count for clip in clips)
        features = _array(archive, "features", "<f8", (rows, dimension))
        mean = _array(archive, "mean", "<f8", (dimension,))
        scale = _array(archive, "scale", "<f8", (dimension,))
        if np.any(scale <= 0):
            raise ValueError("normalization scale must be positive")
        # Custom cache layouts belong to their builders.
        if type(builder) is WalkingFeatures:
            if not set(config.position_joints + config.velocity_joints).issubset(names):
                raise ValueError("walking feature joints are missing from the skeleton")
            pose_dimension = 3 * (
                len(config.position_joints) + len(config.velocity_joints)
            )
            if clips[0].pose_features.shape[1] != pose_dimension:
                raise ValueError("walking pose cache has an incompatible dimension")
        return cls._from_features(
            tuple(motions),
            config,
            builder,
            spec,
            clips,
            features,
            mean,
            scale,
            signature,
        )
