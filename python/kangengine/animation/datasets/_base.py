"""Shared contracts for motion dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


MOTION_DATASETS_DIR_ENV = "KE_MOTION_DATASETS_DIR"
DATASET_URI_PREFIX = "dataset://"


@dataclass(frozen=True)
class DatasetConfig:
    """Base type for dataset-specific load configuration."""


def resolve_dataset_path(path: str | Path) -> Path:
    """Resolve a motion dataset path without relying on packaged assets.

    Absolute paths and existing paths relative to the current working directory
    are preserved. Other relative paths are resolved below the directory named
    by ``KE_MOTION_DATASETS_DIR``.
    """
    value = str(path)
    is_dataset_uri = value.startswith(DATASET_URI_PREFIX)
    if is_dataset_uri:
        value = value.removeprefix(DATASET_URI_PREFIX)
        if not value:
            raise ValueError("dataset URI must include a relative path")
    candidate = Path(value).expanduser()
    if is_dataset_uri and candidate.is_absolute():
        raise ValueError("dataset URI path must be relative")
    if not is_dataset_uri and (candidate.is_absolute() or candidate.exists()):
        return candidate.resolve()

    root = os.environ.get(MOTION_DATASETS_DIR_ENV)
    if root:
        resolved_root = Path(root).expanduser().resolve()
        resolved = (resolved_root / candidate).resolve()
        if is_dataset_uri and not resolved.is_relative_to(resolved_root):
            raise ValueError("dataset URI must remain inside the dataset root")
        return resolved
    raise FileNotFoundError(
        f"relative motion dataset path {str(candidate)!r} was not found; "
        f"set {MOTION_DATASETS_DIR_ENV} or pass an absolute path"
    )


def dataset_uri_from_path(path: str | Path) -> str | None:
    """Return a portable dataset URI when ``path`` is below the configured root."""
    root_value = os.environ.get(MOTION_DATASETS_DIR_ENV)
    if not root_value:
        return None
    root = Path(root_value).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None
    return f"{DATASET_URI_PREFIX}{relative.as_posix()}"


__all__ = [
    "DATASET_URI_PREFIX",
    "DatasetConfig",
    "MOTION_DATASETS_DIR_ENV",
    "dataset_uri_from_path",
    "resolve_dataset_path",
]
