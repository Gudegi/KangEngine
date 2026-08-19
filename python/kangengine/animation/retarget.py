"""Skeleton retargeting configuration and bind-relative motion transfer."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .._core import _ke
from ..utils.batched_rotations import (
    quat_wxyz_conjugate,
    quat_wxyz_multiply,
    quat_wxyz_normalize,
)

if TYPE_CHECKING:
    from . import SkeletonMotion, SkeletonState, SkeletonTree


_RETARGET_SUFFIX = "_retarget.json"


def _vec3(value: object, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(float(item) for item in array)


def _quaternion(value: object, name: str) -> tuple[float, float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain four finite WXYZ values")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must not be a zero quaternion")
    array /= norm
    return tuple(float(item) for item in array)


def _quaternion_map(
    value: object, name: str
) -> dict[str, tuple[float, float, float, float]]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must be an object mapping joint names to WXYZ quaternions"
        )
    return {
        str(joint): _quaternion(quaternion, f"{name}.{joint}")
        for joint, quaternion in value.items()
    }


@dataclass
class RetargetConfig:
    """Serializable calibration between a source and target skeleton.

    ``joint_map`` maps source joint names to target joint names. Bind rotation
    dictionaries replace the corresponding skeleton local bind rotations; any
    omitted joint uses the bind rotation stored by its ``SkeletonTree``.
    ``translation_scale`` converts source root displacement into target units.
    """

    joint_map: dict[str, str]
    source_bind_local_wxyz: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    target_bind_local_wxyz: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    source_bind_root: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_bind_root: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_scale: float = 1.0
    source_skeleton: str = ""
    target_skeleton: str = ""

    def __post_init__(self) -> None:
        joint_map = {
            str(source): str(target) for source, target in self.joint_map.items()
        }
        if any(not source or not target for source, target in joint_map.items()):
            raise ValueError("joint_map names must not be empty")
        if len(set(joint_map.values())) != len(joint_map):
            raise ValueError(
                "joint_map must not map multiple source joints to one target joint"
            )
        scale = float(self.translation_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("translation_scale must be finite and positive")

        self.joint_map = joint_map
        self.source_bind_local_wxyz = _quaternion_map(
            self.source_bind_local_wxyz, "source_bind_local_wxyz"
        )
        self.target_bind_local_wxyz = _quaternion_map(
            self.target_bind_local_wxyz, "target_bind_local_wxyz"
        )
        self.source_bind_root = _vec3(self.source_bind_root, "source_bind_root")
        self.target_bind_root = _vec3(self.target_bind_root, "target_bind_root")
        self.translation_scale = scale
        self.source_skeleton = str(self.source_skeleton)
        self.target_skeleton = str(self.target_skeleton)

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible representation."""
        return {
            "version": 1,
            "source_skeleton": self.source_skeleton,
            "target_skeleton": self.target_skeleton,
            "joint_map": self.joint_map,
            "source_bind_local_wxyz": self.source_bind_local_wxyz,
            "target_bind_local_wxyz": self.target_bind_local_wxyz,
            "source_bind_root": self.source_bind_root,
            "target_bind_root": self.target_bind_root,
            "translation_scale": self.translation_scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RetargetConfig:
        """Construct a config from its versioned JSON representation."""
        if not isinstance(data, dict):
            raise ValueError("retarget config root must be a JSON object")
        if data.get("version") != 1:
            raise ValueError(
                f"unsupported retarget config version: {data.get('version')!r}"
            )
        joint_map = data.get("joint_map")
        if not isinstance(joint_map, dict):
            raise ValueError("joint_map must be a JSON object")
        return cls(
            joint_map={
                str(source): str(target) for source, target in joint_map.items()
            },
            source_bind_local_wxyz=_quaternion_map(
                data.get("source_bind_local_wxyz", {}), "source_bind_local_wxyz"
            ),
            target_bind_local_wxyz=_quaternion_map(
                data.get("target_bind_local_wxyz", {}), "target_bind_local_wxyz"
            ),
            source_bind_root=_vec3(
                data.get("source_bind_root", (0, 0, 0)), "source_bind_root"
            ),
            target_bind_root=_vec3(
                data.get("target_bind_root", (0, 0, 0)), "target_bind_root"
            ),
            translation_scale=float(data.get("translation_scale", 1.0)),
            source_skeleton=str(data.get("source_skeleton", "")),
            target_skeleton=str(data.get("target_skeleton", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> RetargetConfig:
        """Load a ``*_retarget.json`` calibration file."""
        config_path = _validated_path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        """Save this calibration to a ``*_retarget.json`` file."""
        config_path = _validated_path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        return config_path


def _validated_path(path: str | Path) -> Path:
    result = Path(path)
    if not result.name.endswith(_RETARGET_SUFFIX):
        raise ValueError(f"retarget config filename must end with {_RETARGET_SUFFIX!r}")
    return result


def _normalized_rotations(values: object) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float64)
    if result.ndim == 0 or result.shape[-1] != 4:
        raise ValueError("rotation data must have shape [..., 4]")
    if torch.any(torch.linalg.vector_norm(result, dim=-1) <= 1.0e-12):
        raise ValueError("rotation data contains a zero quaternion")
    return quat_wxyz_normalize(result)


def _tree_bind_local(tree: SkeletonTree) -> torch.Tensor:
    values = torch.empty((tree.num_joints(), 4), dtype=torch.float64)
    for joint in range(tree.num_joints()):
        rotation = tree.local_rotation(joint)
        values[joint] = torch.tensor(
            (rotation.w, rotation.x, rotation.y, rotation.z), dtype=torch.float64
        )
    return _normalized_rotations(values)


def _configured_bind_local(
    tree: SkeletonTree,
    overrides: dict[str, tuple[float, float, float, float]],
    label: str,
) -> torch.Tensor:
    output = _tree_bind_local(tree)
    indices = {name: index for index, name in enumerate(tree.node_names())}
    unknown = sorted(set(overrides) - set(indices))
    if unknown:
        raise ValueError(f"unknown {label} bind-pose joints: {unknown}")
    for name, quaternion in overrides.items():
        output[indices[name]] = torch.tensor(quaternion, dtype=torch.float64)
    return output


def _global_rotations(local: torch.Tensor, parents: list[int]) -> torch.Tensor:
    output = torch.empty_like(local)
    for joint, parent in enumerate(parents):
        output[..., joint, :] = (
            local[..., joint, :]
            if parent < 0
            else quat_wxyz_multiply(output[..., parent, :], local[..., joint, :])
        )
    return quat_wxyz_normalize(output)


def _hemisphere_continuous(rotations: torch.Tensor) -> None:
    for frame in range(1, rotations.shape[0]):
        flip = torch.sum(rotations[frame - 1] * rotations[frame], dim=-1) < 0.0
        rotations[frame, flip] *= -1.0


class Retargeter:
    """Compiled bind-relative mapping reusable for motions and live poses.

    Joint names, bind globals, and bind correction rotations are resolved once
    during construction. Create a new retargeter after editing its config.
    """

    def __init__(
        self,
        source_skeleton: SkeletonTree,
        target_skeleton: SkeletonTree,
        config: RetargetConfig,
    ) -> None:
        self.source_skeleton = source_skeleton
        self.target_skeleton = target_skeleton
        self.config = config
        self._source_names = tuple(source_skeleton.node_names())
        self._target_names = tuple(target_skeleton.node_names())
        self._source_parents = source_skeleton.parent_indices()
        self._target_parents = target_skeleton.parent_indices()

        source_index = {name: index for index, name in enumerate(self._source_names)}
        target_index = {name: index for index, name in enumerate(self._target_names)}
        missing_source = sorted(set(config.joint_map) - set(source_index))
        missing_target = sorted(set(config.joint_map.values()) - set(target_index))
        if missing_source or missing_target:
            raise ValueError(
                "unknown retarget joints: "
                f"source={missing_source}, target={missing_target}"
            )

        source_bind_global = _global_rotations(
            _configured_bind_local(
                source_skeleton, config.source_bind_local_wxyz, "source"
            ),
            self._source_parents,
        )
        self._target_bind_local = _configured_bind_local(
            target_skeleton, config.target_bind_local_wxyz, "target"
        )
        target_bind_global = _global_rotations(
            self._target_bind_local, self._target_parents
        )
        reverse_map = {
            target_index[target]: source_index[source]
            for source, target in config.joint_map.items()
        }
        self._source_joint_for_target = tuple(
            reverse_map.get(joint, -1) for joint in range(len(self._target_names))
        )
        self._bind_delta = torch.empty(
            (len(self._target_names), 4), dtype=torch.float64
        )
        self._bind_delta[:, 0] = 1.0
        self._bind_delta[:, 1:] = 0.0
        for target_joint, source_joint in enumerate(self._source_joint_for_target):
            if source_joint >= 0:
                self._bind_delta[target_joint] = quat_wxyz_multiply(
                    quat_wxyz_conjugate(source_bind_global[source_joint]),
                    target_bind_global[target_joint],
                )

        self._source_bind_root = np.asarray(config.source_bind_root, dtype=np.float32)
        self._target_bind_root = np.asarray(config.target_bind_root, dtype=np.float32)
        self._translation_scale = config.translation_scale

    def _retarget_global(self, source_global: torch.Tensor) -> torch.Tensor:
        if source_global.shape[-2:] != (len(self._source_names), 4):
            raise ValueError(
                f"source rotations must have shape [..., {len(self._source_names)}, 4]"
            )
        output_shape = source_global.shape[:-2] + (len(self._target_names), 4)
        target_global = torch.empty(output_shape, dtype=source_global.dtype)
        target_local = torch.empty_like(target_global)
        for joint, parent in enumerate(self._target_parents):
            source_joint = self._source_joint_for_target[joint]
            if source_joint < 0:
                target_local[..., joint, :] = self._target_bind_local[joint]
                target_global[..., joint, :] = (
                    self._target_bind_local[joint]
                    if parent < 0
                    else quat_wxyz_multiply(
                        target_global[..., parent, :],
                        self._target_bind_local[joint],
                    )
                )
                continue

            desired_global = quat_wxyz_multiply(
                source_global[..., source_joint, :], self._bind_delta[joint]
            )
            target_global[..., joint, :] = desired_global
            target_local[..., joint, :] = (
                desired_global
                if parent < 0
                else quat_wxyz_multiply(
                    quat_wxyz_conjugate(target_global[..., parent, :]),
                    desired_global,
                )
            )
        return quat_wxyz_normalize(target_local)

    def _retarget_root(self, source_root: object) -> np.ndarray:
        root = np.asarray(source_root, dtype=np.float32)
        if root.shape[-1:] != (3,):
            raise ValueError("root_translation must have shape [..., 3]")
        return (
            root - self._source_bind_root
        ) * self._translation_scale + self._target_bind_root

    def retarget_pose(
        self,
        root_translation: object,
        local_rotations_wxyz: object,
    ) -> SkeletonState:
        """Retarget one raw local pose for real-time use."""
        source_local = _normalized_rotations(local_rotations_wxyz)
        if source_local.shape != (len(self._source_names), 4):
            raise ValueError(
                f"local_rotations_wxyz must have shape [{len(self._source_names)}, 4]"
            )
        source_global = _global_rotations(source_local, self._source_parents)
        target_local = self._retarget_global(source_global)
        target_root = self._retarget_root(root_translation)
        if target_root.shape != (3,):
            raise ValueError("root_translation must have shape [3]")
        return _ke.animation.SkeletonState.from_rotation_and_root_translation(
            self.target_skeleton,
            target_local.to(dtype=torch.float32),
            target_root,
            True,
        )

    def retarget_state(self, source_state: SkeletonState) -> SkeletonState:
        """Retarget one ``SkeletonState`` using its stored rotation space."""
        if source_state.num_joints() != len(self._source_names):
            raise ValueError("source_state joint count does not match source_skeleton")
        rotations = torch.empty((len(self._source_names), 4), dtype=torch.float64)
        if source_state.is_local():
            for joint in range(len(self._source_names)):
                rotation = source_state.rotation(joint)
                rotations[joint] = torch.tensor(
                    (rotation.w, rotation.x, rotation.y, rotation.z),
                    dtype=torch.float64,
                )
            source_global = _global_rotations(
                _normalized_rotations(rotations), self._source_parents
            )
        else:
            for joint in range(len(self._source_names)):
                rotation = source_state.rotation(joint)
                rotations[joint] = torch.tensor(
                    (rotation.w, rotation.x, rotation.y, rotation.z),
                    dtype=torch.float64,
                )
            source_global = _normalized_rotations(rotations)

        root = source_state.root_translation()
        target_root = self._retarget_root((root.x, root.y, root.z))
        target_local = self._retarget_global(source_global)
        return _ke.animation.SkeletonState.from_rotation_and_root_translation(
            self.target_skeleton,
            target_local.to(dtype=torch.float32),
            target_root,
            True,
        )

    def retarget_motion(self, source_motion: SkeletonMotion) -> SkeletonMotion:
        """Retarget a complete motion using the compiled mapping."""
        if tuple(source_motion.node_names()) != self._source_names:
            raise ValueError("source_motion skeleton does not match source_skeleton")
        source_global = _normalized_rotations(source_motion.global_rotations_wxyz())
        target_local = self._retarget_global(source_global)
        _hemisphere_continuous(target_local)
        root_translations = self._retarget_root(source_motion.root_translations())
        return _ke.animation.SkeletonMotion.from_arrays(
            self.target_skeleton,
            root_translations,
            target_local.to(dtype=torch.float32),
            source_motion.fps(),
            source_motion.motion_name(),
        )


def retarget_motion(
    source_motion: SkeletonMotion,
    target_skeleton: SkeletonTree,
    config: RetargetConfig,
) -> SkeletonMotion:
    """Retarget a motion with a temporary ``Retargeter``."""
    return Retargeter(
        source_motion.skeleton_tree, target_skeleton, config
    ).retarget_motion(source_motion)


__all__ = ["RetargetConfig", "Retargeter", "retarget_motion"]
