"""Dataset adapters that produce ``SkeletonMotion`` objects."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from ...utils import CoordinateSystem
from . import amass as amass, ascend as ascend
from ._base import (
    DATASET_URI_PREFIX,
    DatasetConfig,
    MOTION_DATASETS_DIR_ENV,
    dataset_uri_from_path,
    resolve_dataset_path,
)
from .ascend import ASCEND_NATIVE_TO_Y_UP, AscendConfig
from .amass import AMASSConfig

if TYPE_CHECKING:
    from .. import SkeletonMotion


class DatasetType(str, Enum):
    """Dataset adapters supported by :func:`load_motion`."""

    AMASS = "amass"
    ASCEND = "ascend"


DatasetLoader = Callable[..., "SkeletonMotion"]
_LOADERS: dict[DatasetType, DatasetLoader] = {
    DatasetType.AMASS: amass.load_motion,
    DatasetType.ASCEND: ascend.load_motion,
}


def detect_dataset_type(path: str | Path) -> DatasetType | None:
    """Identify a supported motion dataset from its container and schema."""

    source_path = resolve_dataset_path(path)
    suffix = source_path.suffix.lower()
    if suffix in {".npy", ".pkl"}:
        return DatasetType.ASCEND
    if suffix != ".npz":
        return None
    with np.load(source_path, allow_pickle=False) as archive:
        keys = set(archive.files)
    if {"poses", "trans"}.issubset(keys):
        return DatasetType.AMASS
    return DatasetType.ASCEND


def load_motion(
    path: str | Path,
    *,
    dataset: DatasetType | str,
    target_world: CoordinateSystem | None = None,
    config: DatasetConfig | None = None,
) -> SkeletonMotion:
    """Load a dataset motion, optionally rotating its world-space root.

    ``target_world=None`` preserves the dataset's native world convention.
    World conversion changes root translation and orientation only; the
    skeleton and non-root local rotations are preserved.
    """
    dataset_type = DatasetType(dataset)
    loader = _LOADERS[dataset_type]
    return loader(
        resolve_dataset_path(path),
        target_world=target_world,
        config=config,
    )


__all__ = [
    "ASCEND_NATIVE_TO_Y_UP",
    "AMASSConfig",
    "DATASET_URI_PREFIX",
    "AscendConfig",
    "DatasetConfig",
    "DatasetType",
    "MOTION_DATASETS_DIR_ENV",
    "amass",
    "ascend",
    "dataset_uri_from_path",
    "detect_dataset_type",
    "load_motion",
    "resolve_dataset_path",
]
