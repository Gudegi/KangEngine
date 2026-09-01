"""Shared source-motion and target-skeleton profiles for retargeting."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from ..coordinates import CoordinateSystem
from ..datasets._base import dataset_uri_from_path, resolve_dataset_path


_MOTION_SUFFIX = "_motion.json"
_ANGLE_TARGET_SUFFIX = "_angle_target.json"


def _validated_path(path: str | Path, suffix: str, label: str) -> Path:
    result = Path(path).expanduser()
    if not result.name.endswith(suffix):
        raise ValueError(f"{label} filename must end with {suffix!r}")
    return result


def _resolve_reference(owner: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    if value.startswith("dataset://"):
        return resolve_dataset_path(value)
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = owner.parent / result
    return result.resolve()


def _relative_reference(owner: Path, referenced: Path) -> str:
    dataset_uri = dataset_uri_from_path(referenced)
    if dataset_uri is not None:
        return dataset_uri
    return os.path.relpath(referenced.resolve(), owner.parent.resolve())


def _load_object(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} root must be a JSON object")
    return data


def _save_object(path: Path, data: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return path


@dataclass(frozen=True)
class MotionSourceProfile:
    """Properties shared by motions using one source skeleton convention."""

    name: str
    coordinate_system: str = CoordinateSystem.Y_UP_Z_FORWARD.value
    translation_unit_scale: float = 1.0
    has_armature_joint: bool = False
    reference_skeleton: Path | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("motion source profile name must not be empty")
        object.__setattr__(
            self,
            "coordinate_system",
            CoordinateSystem(self.coordinate_system).value,
        )
        scale = float(self.translation_unit_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("translation_unit_scale must be finite and positive")
        object.__setattr__(self, "translation_unit_scale", scale)
        object.__setattr__(self, "has_armature_joint", bool(self.has_armature_joint))
        if self.reference_skeleton is not None:
            object.__setattr__(
                self, "reference_skeleton", Path(self.reference_skeleton).expanduser()
            )

    def to_dict(self, *, owner_path: Path | None = None) -> dict[str, object]:
        reference = self.reference_skeleton
        if reference is not None and owner_path is not None:
            reference_value = _relative_reference(owner_path, reference)
        else:
            reference_value = None if reference is None else str(reference)
        return {
            "version": 1,
            "name": self.name,
            "coordinate_system": self.coordinate_system,
            "translation_unit_scale": self.translation_unit_scale,
            "has_armature_joint": self.has_armature_joint,
            "reference_skeleton": reference_value,
        }

    @classmethod
    def from_dict(
        cls, data: object, *, owner_path: str | Path | None = None
    ) -> MotionSourceProfile:
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported motion source profile")
        reference = data.get("reference_skeleton")
        if reference is not None and owner_path is not None:
            reference = _resolve_reference(
                Path(owner_path), reference, "reference_skeleton"
            )
        return cls(
            name=str(data.get("name", "")),
            coordinate_system=str(
                data.get("coordinate_system", CoordinateSystem.Y_UP_Z_FORWARD.value)
            ),
            translation_unit_scale=float(data.get("translation_unit_scale", 1.0)),
            has_armature_joint=bool(data.get("has_armature_joint", False)),
            reference_skeleton=(None if reference is None else Path(str(reference))),
        )

    @classmethod
    def load(cls, path: str | Path) -> MotionSourceProfile:
        profile_path = _validated_path(path, _MOTION_SUFFIX, "motion source profile")
        return cls.from_dict(_load_object(profile_path), owner_path=profile_path)

    def save(self, path: str | Path) -> Path:
        profile_path = _validated_path(path, _MOTION_SUFFIX, "motion source profile")
        return _save_object(profile_path, self.to_dict(owner_path=profile_path))


@dataclass(frozen=True)
class AngleTargetProfile:
    """Properties shared by one angle-retarget target skeleton."""

    name: str
    skeleton: Path
    coordinate_system: str = CoordinateSystem.Y_UP_Z_FORWARD.value

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("skeleton profile name must not be empty")
        object.__setattr__(self, "skeleton", Path(self.skeleton).expanduser())
        object.__setattr__(
            self,
            "coordinate_system",
            CoordinateSystem(self.coordinate_system).value,
        )

    def to_dict(self, *, owner_path: Path | None = None) -> dict[str, object]:
        reference = self.skeleton
        reference_value = (
            _relative_reference(owner_path, reference)
            if owner_path is not None
            else str(reference)
        )
        return {
            "version": 1,
            "name": self.name,
            "skeleton": reference_value,
            "coordinate_system": self.coordinate_system,
        }

    @classmethod
    def from_dict(
        cls, data: object, *, owner_path: str | Path | None = None
    ) -> AngleTargetProfile:
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported skeleton profile")
        reference = data.get("skeleton")
        if owner_path is not None:
            reference = _resolve_reference(Path(owner_path), reference, "skeleton")
        return cls(
            name=str(data.get("name", "")),
            skeleton=Path(str(reference)),
            coordinate_system=str(
                data.get("coordinate_system", CoordinateSystem.Y_UP_Z_FORWARD.value)
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> AngleTargetProfile:
        profile_path = _validated_path(
            path, _ANGLE_TARGET_SUFFIX, "angle target profile"
        )
        return cls.from_dict(_load_object(profile_path), owner_path=profile_path)

    def save(self, path: str | Path) -> Path:
        profile_path = _validated_path(
            path, _ANGLE_TARGET_SUFFIX, "angle target profile"
        )
        return _save_object(profile_path, self.to_dict(owner_path=profile_path))


__all__ = ["AngleTargetProfile", "MotionSourceProfile"]
