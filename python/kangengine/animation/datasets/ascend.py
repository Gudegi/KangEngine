"""Ascend SMPL motion dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..._core import _ke
from ...utils import (
    CoordinateSystem,
    coordinate_conversion_matrix,
    matrix_to_quat_wxyz,
)
from ...utils.math import quat_wxyz_from_rotation_vector_numpy
from ..transform import transform_motion
from ._base import DatasetConfig

if TYPE_CHECKING:
    from .. import SkeletonMotion, SkeletonTree


@dataclass(frozen=True)
class AscendConfig(DatasetConfig):
    """Ascend SMPL field selection and skeleton construction options."""

    person_key: str = "second_person"
    pose_key: str = "opt_pose"
    translation_key: str = "opt_trans"
    beta_key: str = "beta"
    default_fps: float = 30.0
    smpl_model_path: str | Path | None = None
    skeleton: SkeletonTree | None = None
    root_translation_offset: tuple[float, float, float] | None = None


# Ascend capture coordinates are Z-up/-Y-forward. SMPL skeleton and child
# local joint frames remain in their native Y-up representation.
ASCEND_NATIVE_TO_Y_UP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    dtype=np.float32,
)


def _dictionary(value: object, label: str) -> dict:
    if isinstance(value, dict):
        return value
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, dict):
            return item
    raise ValueError(f"{label} must be a dictionary")


def _load_archive(path: Path) -> dict:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            return {name: loaded[name] for name in loaded.files}
        finally:
            loaded.close()
    return _dictionary(loaded, "Ascend archive")


def _fps(person: dict, default_fps: float) -> float:
    value = person.get("mocap_framerate", person.get("mocap_frame_rate", default_fps))
    result = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"frame rate must be positive and finite, got {result}")
    return result


def _skeleton(person: dict, config: AscendConfig) -> SkeletonTree:
    if config.skeleton is not None:
        return config.skeleton

    from ...asset.smpl import SMPLModel, repository_smpl_model_path

    model_path = (
        repository_smpl_model_path("neutral")
        if config.smpl_model_path is None
        else Path(config.smpl_model_path).expanduser().resolve()
    )
    betas = np.asarray(person.get(config.beta_key, []), np.float32)
    return SMPLModel.load(model_path).create_body(betas).skeleton_tree


def load_motion(
    path: Path,
    *,
    target_world: CoordinateSystem | None,
    config: DatasetConfig | None,
) -> SkeletonMotion:
    """Load one Ascend clip with an optional world-root conversion."""
    if config is not None and not isinstance(config, AscendConfig):
        raise TypeError("Ascend datasets require config=AscendConfig(...)")
    effective = AscendConfig() if config is None else config
    archive = _load_archive(path)
    if effective.person_key not in archive:
        raise KeyError(
            f"Ascend person {effective.person_key!r} is absent; "
            f"available={sorted(archive)}"
        )
    person = _dictionary(
        archive[effective.person_key], f"Ascend person {effective.person_key!r}"
    )
    if effective.pose_key not in person or effective.translation_key not in person:
        raise KeyError(
            f"Ascend motion requires {effective.pose_key!r} and "
            f"{effective.translation_key!r}; available={sorted(person)}"
        )

    skeleton = _skeleton(person, effective)
    poses = np.asarray(person[effective.pose_key], np.float32)
    translations = np.asarray(person[effective.translation_key], np.float32)
    joints = skeleton.num_joints()
    if poses.ndim != 2 or poses.shape[1] != joints * 3:
        raise ValueError(
            f"Ascend poses must have shape [frames, {joints * 3}], got {poses.shape}"
        )
    if translations.shape != (len(poses), 3):
        raise ValueError(
            f"Ascend translations must have shape [{len(poses)}, 3], got "
            f"{translations.shape}"
        )
    if not np.isfinite(poses).all() or not np.isfinite(translations).all():
        raise ValueError("Ascend pose/translation data contains NaN or infinity")

    result = _ke.animation.SkeletonMotion.from_arrays(
        skeleton,
        np.ascontiguousarray(translations, dtype=np.float32),
        np.ascontiguousarray(
            quat_wxyz_from_rotation_vector_numpy(poses.reshape(len(poses), joints, 3)),
            dtype=np.float32,
        ),
        _fps(person, effective.default_fps),
        path.stem,
    )
    if target_world is not None:
        target = CoordinateSystem(target_world)
        y_up_to_target = coordinate_conversion_matrix(
            CoordinateSystem.Y_UP_Z_FORWARD, target
        )
        native_to_target = y_up_to_target @ ASCEND_NATIVE_TO_Y_UP
        rotation = matrix_to_quat_wxyz(
            torch.from_numpy(native_to_target.astype(np.float64))
        ).numpy()
        result = transform_motion(result, rotation_wxyz=rotation)
    if effective.root_translation_offset is not None:
        result = transform_motion(
            result,
            translation=np.asarray(effective.root_translation_offset, dtype=np.float64),
        )
    return result


__all__ = ["ASCEND_NATIVE_TO_Y_UP", "AscendConfig", "load_motion"]
