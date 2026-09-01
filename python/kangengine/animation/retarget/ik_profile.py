"""Serializable source, robot, and pair configuration for IK retargeting."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..coordinates import CoordinateSystem
from .ik import AuxiliaryRetargetPoint, IKEffectorMapping
from .profile import MotionSourceProfile


_TARGET_SUFFIX = "_ik_target.json"
_RETARGET_SUFFIX = "_ik_retarget.json"


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


def _validated_path(path: str | Path, suffix: str, label: str) -> Path:
    result = Path(path).expanduser()
    if not result.name.endswith(suffix):
        raise ValueError(f"{label} filename must end with {suffix!r}")
    return result


def _relative_reference(owner: Path, referenced: Path) -> str:
    return os.path.relpath(referenced.resolve(), owner.parent.resolve())


def _resolve_reference(owner: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = owner.parent / result
    return result.resolve()


def _finite_positive(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _vec3(
    value: object, name: str, *, positive: bool = False
) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite numbers")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return tuple(float(item) for item in array)


def _quat(value: object, name: str) -> tuple[float, float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain four finite WXYZ values")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must not be a zero quaternion")
    return tuple(float(item) for item in array / norm)


@dataclass(frozen=True)
class IKTargetProfile:
    """Articulation structure and IK defaults shared by one robot."""

    name: str
    skeleton: Path
    joint_order: tuple[str, ...]
    coordinate_system: str = CoordinateSystem.Z_UP_X_FORWARD.value
    visual_skeleton: Path | None = None
    rest_configuration: tuple[float, ...] | None = None
    joint_rest_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("robot profile name must not be empty")
        if not self.joint_order or len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("joint_order must contain unique joint names")
        object.__setattr__(self, "skeleton", Path(self.skeleton))
        if self.visual_skeleton is not None:
            object.__setattr__(self, "visual_skeleton", Path(self.visual_skeleton))
        object.__setattr__(
            self,
            "coordinate_system",
            CoordinateSystem(self.coordinate_system).value,
        )
        weights = {
            str(name): float(value) for name, value in self.joint_rest_weights.items()
        }
        unknown = set(weights) - set(self.joint_order)
        if unknown:
            raise ValueError(f"joint_rest_weights contains unknown joints: {unknown}")
        if any(value < 0.0 or not np.isfinite(value) for value in weights.values()):
            raise ValueError("joint rest weights must be finite and non-negative")
        object.__setattr__(self, "joint_rest_weights", MappingProxyType(weights))
        if self.rest_configuration is not None:
            rest = tuple(float(value) for value in self.rest_configuration)
            if len(rest) != len(self.joint_order) or not np.isfinite(rest).all():
                raise ValueError(
                    "rest_configuration must contain one finite value per joint"
                )
            object.__setattr__(self, "rest_configuration", rest)

    def to_dict(self, *, owner_path: Path | None = None) -> dict[str, object]:
        target = self.skeleton
        visual = self.visual_skeleton
        if owner_path is not None:
            target_value = _relative_reference(owner_path, target)
            visual_value = (
                None if visual is None else _relative_reference(owner_path, visual)
            )
        else:
            target_value = str(target)
            visual_value = None if visual is None else str(visual)
        return {
            "version": 1,
            "name": self.name,
            "skeleton": target_value,
            "visual_skeleton": visual_value,
            "coordinate_system": self.coordinate_system,
            "joint_order": self.joint_order,
            "rest_configuration": self.rest_configuration,
            "joint_rest_weights": dict(self.joint_rest_weights),
        }

    @classmethod
    def from_dict(
        cls, data: object, *, owner_path: str | Path | None = None
    ) -> IKTargetProfile:
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported IK robot profile")
        owner = Path(owner_path) if owner_path is not None else None
        target_value = data.get("skeleton")
        if owner is None:
            target = Path(str(target_value))
        else:
            target = _resolve_reference(owner, target_value, "skeleton")
        visual_value = data.get("visual_skeleton")
        visual = None
        if visual_value is not None:
            visual = (
                Path(str(visual_value))
                if owner is None
                else _resolve_reference(owner, visual_value, "visual_skeleton")
            )
        joint_order = data.get("joint_order")
        if not isinstance(joint_order, list):
            raise ValueError("joint_order must be a JSON array")
        rest = data.get("rest_configuration")
        return cls(
            name=str(data.get("name", "")),
            skeleton=target,
            visual_skeleton=visual,
            coordinate_system=str(
                data.get(
                    "coordinate_system",
                    CoordinateSystem.Z_UP_X_FORWARD.value,
                )
            ),
            joint_order=tuple(str(name) for name in joint_order),
            rest_configuration=(
                None if rest is None else tuple(float(value) for value in rest)
            ),
            joint_rest_weights=data.get("joint_rest_weights", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> IKTargetProfile:
        profile_path = _validated_path(path, _TARGET_SUFFIX, "IK target profile")
        return cls.from_dict(_load_object(profile_path), owner_path=profile_path)

    def save(self, path: str | Path) -> Path:
        profile_path = _validated_path(path, _TARGET_SUFFIX, "IK target profile")
        return _save_object(profile_path, self.to_dict(owner_path=profile_path))


def _mapping_from_dict(data: object, name: str) -> IKEffectorMapping:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    scale = data.get("position_scale")
    return IKEffectorMapping(
        source_joint=str(data.get("source_joint", "")),
        target_link=str(data.get("target_link", "")),
        region=None if data.get("region") is None else str(data["region"]),
        position_weight=float(data.get("position_weight", 1.0)),
        rotation_weight=float(data.get("rotation_weight", 0.0)),
        position_scale=(
            None
            if scale is None
            else _vec3(scale, f"{name}.position_scale", positive=True)
        ),
        position_offset=_vec3(
            data.get("position_offset", (0, 0, 0)), f"{name}.position_offset"
        ),
        rotation_offset_wxyz=_quat(
            data.get("rotation_offset_wxyz", (1, 0, 0, 0)),
            f"{name}.rotation_offset_wxyz",
        ),
    )


def _mapping_to_dict(mapping: IKEffectorMapping) -> dict[str, object]:
    return {
        "source_joint": mapping.source_joint,
        "target_link": mapping.target_link,
        "region": mapping.region,
        "position_weight": mapping.position_weight,
        "rotation_weight": mapping.rotation_weight,
        "position_scale": mapping.position_scale,
        "position_offset": mapping.position_offset,
        "rotation_offset_wxyz": mapping.rotation_offset_wxyz,
    }


def _auxiliary_from_dict(data: object, name: str) -> AuxiliaryRetargetPoint:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    return AuxiliaryRetargetPoint(
        name=str(data.get("name", "")),
        source_joint=str(data.get("source_joint", "")),
        target_link=str(data.get("target_link", "")),
        target_offset=_vec3(data.get("target_offset"), f"{name}.target_offset"),
        source_offset=_vec3(
            data.get("source_offset", (0, 0, 0)), f"{name}.source_offset"
        ),
        region=None if data.get("region") is None else str(data["region"]),
        weight=float(data.get("weight", 1.0)),
    )


def _auxiliary_to_dict(point: AuxiliaryRetargetPoint) -> dict[str, object]:
    return {
        "name": point.name,
        "source_joint": point.source_joint,
        "target_link": point.target_link,
        "target_offset": point.target_offset,
        "source_offset": point.source_offset,
        "region": point.region,
        "weight": point.weight,
    }


@dataclass(frozen=True)
class IKRetargetConfig:
    """Serializable source-to-robot IK mapping, analogous to AngleRetargetConfig."""

    source_profile: MotionSourceProfile
    target_profile: IKTargetProfile
    link_mappings: tuple[IKEffectorMapping, ...]
    source_root_joint: str
    target_root_link: str
    source_profile_path: Path | None = None
    target_profile_path: Path | None = None
    translation_scale: float = 1.0
    region_scales: Mapping[str, tuple[float, float, float]] = field(
        default_factory=dict
    )
    auxiliary_points: tuple[AuxiliaryRetargetPoint, ...] = ()
    local_alignment_pairs: tuple[tuple[str, str], ...] = ()
    foot_shape_pairs: tuple[tuple[str, str], ...] = ()
    contact_links: tuple[str, ...] = ()
    root_position_scale: tuple[float, float, float] | None = None
    root_height_offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.link_mappings:
            raise ValueError("link_mappings must contain at least one mapping")
        if not self.source_root_joint or not self.target_root_link:
            raise ValueError("source and target root names must not be empty")
        object.__setattr__(
            self,
            "translation_scale",
            _finite_positive(self.translation_scale, "translation_scale"),
        )
        regions = {
            str(name): _vec3(value, f"region_scales.{name}", positive=True)
            for name, value in self.region_scales.items()
        }
        if any(not name for name in regions):
            raise ValueError("region scale names must not be empty")
        object.__setattr__(self, "region_scales", MappingProxyType(regions))
        if self.root_position_scale is not None:
            object.__setattr__(
                self,
                "root_position_scale",
                _vec3(
                    self.root_position_scale,
                    "root_position_scale",
                    positive=True,
                ),
            )
        root_height = float(self.root_height_offset)
        if not np.isfinite(root_height):
            raise ValueError("root_height_offset must be finite")
        object.__setattr__(self, "root_height_offset", root_height)
        if len(set(self.contact_links)) != len(self.contact_links):
            raise ValueError("contact_links contains duplicate link names")
        mapped_links = {mapping.target_link for mapping in self.link_mappings}
        for label, pairs in (
            ("local_alignment_pairs", self.local_alignment_pairs),
            ("foot_shape_pairs", self.foot_shape_pairs),
        ):
            unknown = {
                link for pair in pairs for link in pair if link not in mapped_links
            }
            if unknown:
                raise ValueError(f"{label} contains unmapped links: {sorted(unknown)}")

    @property
    def urdf_path(self) -> Path:
        return self.target_profile.skeleton

    @property
    def joint_order(self) -> tuple[str, ...]:
        return self.target_profile.joint_order

    @property
    def rest_configuration(self) -> tuple[float, ...] | None:
        return self.target_profile.rest_configuration

    @property
    def joint_rest_weights(self) -> Mapping[str, float]:
        return self.target_profile.joint_rest_weights

    @property
    def source_translation_scale(self) -> float:
        return self.source_profile.translation_unit_scale * self.translation_scale

    def validate_urdf_path(self) -> None:
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file does not exist: {self.urdf_path}")

    def to_dict(self, *, owner_path: Path | None = None) -> dict[str, object]:
        if self.source_profile_path is None or self.target_profile_path is None:
            raise ValueError(
                "saving requires source_profile_path and target_profile_path"
            )
        source_path = self.source_profile_path
        target_path = self.target_profile_path
        if owner_path is not None:
            source_value = _relative_reference(owner_path, source_path)
            target_value = _relative_reference(owner_path, target_path)
        else:
            source_value = str(source_path)
            target_value = str(target_path)
        return {
            "version": 1,
            "source_profile": source_value,
            "target_profile": target_value,
            "source_root_joint": self.source_root_joint,
            "target_root_link": self.target_root_link,
            "translation_scale": self.translation_scale,
            "link_mappings": [_mapping_to_dict(value) for value in self.link_mappings],
            "region_scales": dict(self.region_scales),
            "root_position_scale": self.root_position_scale,
            "root_height_offset": self.root_height_offset,
            "auxiliary_points": [
                _auxiliary_to_dict(value) for value in self.auxiliary_points
            ],
            "local_alignment_pairs": self.local_alignment_pairs,
            "foot_shape_pairs": self.foot_shape_pairs,
            "contact_links": self.contact_links,
        }

    @classmethod
    def load(cls, path: str | Path) -> IKRetargetConfig:
        config_path = _validated_path(path, _RETARGET_SUFFIX, "IK retarget config")
        data = _load_object(config_path)
        if data.get("version") != 1:
            raise ValueError(f"unsupported IK retarget config: {data.get('version')!r}")
        source_path = _resolve_reference(
            config_path, data.get("source_profile"), "source_profile"
        )
        target_path = _resolve_reference(
            config_path, data.get("target_profile"), "target_profile"
        )
        mappings = data.get("link_mappings")
        if not isinstance(mappings, list):
            raise ValueError("link_mappings must be a JSON array")
        auxiliary = data.get("auxiliary_points", [])
        if not isinstance(auxiliary, list):
            raise ValueError("auxiliary_points must be a JSON array")
        return cls(
            source_profile=MotionSourceProfile.load(source_path),
            target_profile=IKTargetProfile.load(target_path),
            source_profile_path=source_path,
            target_profile_path=target_path,
            source_root_joint=str(data.get("source_root_joint", "")),
            target_root_link=str(data.get("target_root_link", "")),
            translation_scale=float(data.get("translation_scale", 1.0)),
            link_mappings=tuple(
                _mapping_from_dict(value, f"link_mappings[{index}]")
                for index, value in enumerate(mappings)
            ),
            region_scales=data.get("region_scales", {}),
            root_position_scale=data.get("root_position_scale"),
            root_height_offset=float(data.get("root_height_offset", 0.0)),
            auxiliary_points=tuple(
                _auxiliary_from_dict(value, f"auxiliary_points[{index}]")
                for index, value in enumerate(auxiliary)
            ),
            local_alignment_pairs=tuple(
                tuple(str(link) for link in pair)
                for pair in data.get("local_alignment_pairs", [])
            ),
            foot_shape_pairs=tuple(
                tuple(str(link) for link in pair)
                for pair in data.get("foot_shape_pairs", [])
            ),
            contact_links=tuple(str(link) for link in data.get("contact_links", [])),
        )

    def save(self, path: str | Path) -> Path:
        config_path = _validated_path(path, _RETARGET_SUFFIX, "IK retarget config")
        return _save_object(config_path, self.to_dict(owner_path=config_path))


__all__ = ["IKRetargetConfig", "IKTargetProfile"]
