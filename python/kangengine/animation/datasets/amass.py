"""AMASS dataset adapter producing native ``SkeletonMotion`` objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..._core import _ke
from ...utils import CoordinateSystem
from ..transform import transform_motion
from ._base import DatasetConfig, resolve_dataset_path

if TYPE_CHECKING:
    from .. import SkeletonMotion, SkeletonTree


@dataclass(frozen=True)
class AMASSConfig(DatasetConfig):
    """AMASS body-model, scale, and output adjustment options."""

    model_type: str = "smplx"
    model_path: str | Path | None = None
    skeleton: SkeletonTree | None = None
    scale: float = 1.0
    root_translation_offset: tuple[float, float, float] | None = None


def _skeleton(path: Path, config: AMASSConfig) -> SkeletonTree:
    if config.skeleton is not None:
        return config.skeleton

    from ...asset import AMASSLoader
    from ...asset.smpl import (
        SMPLHModel,
        SMPLModel,
        SMPLXModel,
        repository_smpl_model_path,
        repository_smplh_model_path,
        repository_smplx_model_path,
    )

    info = AMASSLoader.inspect(path)
    kind = config.model_type.lower()
    models = {
        "smpl": (SMPLModel, repository_smpl_model_path),
        "smplh": (SMPLHModel, repository_smplh_model_path),
        "smplx": (SMPLXModel, repository_smplx_model_path),
    }
    if kind not in models:
        raise ValueError("AMASS model_type must be 'smpl', 'smplh', or 'smplx'")
    model_class, default_path = models[kind]
    model_path = (
        default_path(info.gender)
        if config.model_path is None
        else resolve_dataset_path(config.model_path)
    )
    return model_class.load(model_path).create_body(info.betas).skeleton_tree


def _up_axis(target_world: CoordinateSystem | None):
    if target_world is None:
        return _ke.UpAxis.Z
    target = CoordinateSystem(target_world)
    if target is CoordinateSystem.Y_UP_Z_FORWARD:
        return _ke.UpAxis.Y
    if target is CoordinateSystem.Z_UP_X_FORWARD:
        return _ke.UpAxis.Z
    raise ValueError("AMASS target_world supports y_up_z_forward or z_up_x_forward")


def load_motion(
    path: Path,
    *,
    target_world: CoordinateSystem | None,
    config: DatasetConfig | None,
) -> SkeletonMotion:
    """Load one AMASS NPZ using its body shape and coordinate convention."""
    if config is not None and not isinstance(config, AMASSConfig):
        raise TypeError("AMASS datasets require config=AMASSConfig(...)")
    effective = AMASSConfig() if config is None else config
    scale = float(effective.scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("AMASS scale must be finite and positive")

    from ...asset import AMASSLoader

    result = AMASSLoader.load_motion(
        path,
        _skeleton(path, effective),
        model_type=effective.model_type,
        up_axis=_up_axis(target_world),
        scale=scale,
    )
    if effective.root_translation_offset is not None:
        result = transform_motion(
            result,
            translation=np.asarray(effective.root_translation_offset, dtype=np.float64),
        )
    return result


__all__ = ["AMASSConfig", "load_motion"]
