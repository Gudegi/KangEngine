"""Headless file and batch processing for configured motion retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .coordinates import (
    CoordinateSystem,
    convert_motion_coordinates,
    convert_skeleton_coordinates,
)
from .retarget import RetargetConfig, Retargeter


def _asset_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path


def _default_scale(path: Path) -> float:
    return 0.01 if path.suffix.lower() == ".fbx" else 1.0


def _load_motion(path: Path, *, has_armature_joint: bool = False):
    from .. import asset

    if path.suffix.lower() == ".bvh":
        return asset.BVHLoader.load_motion(
            bvh_path=str(path),
            scale=_default_scale(path),
            has_armature_joint=has_armature_joint,
        )
    if path.suffix.lower() == ".fbx":
        return asset.FBXLoader.load_motion(
            fbx_path=str(path), scale=_default_scale(path)
        )
    raise ValueError(f"unsupported source motion format: {path.suffix}")


def _load_skeleton(path: Path, *, has_armature_joint: bool = False):
    from .. import asset

    if path.suffix.lower() == ".bvh":
        return asset.BVHLoader.load_skeleton(
            bvh_path=str(path),
            scale=_default_scale(path),
            has_armature_joint=has_armature_joint,
        )
    if path.suffix.lower() == ".fbx":
        return asset.FBXLoader.load_skeleton(
            fbx_path=str(path), scale=_default_scale(path)
        )
    if path.suffix.lower() == ".xml":
        return asset.MJCFLoader.load(
            str(path), scale=_default_scale(path), order="DFS"
        ).skeleton_tree
    raise ValueError(f"unsupported skeleton format: {path.suffix}")


@dataclass(frozen=True)
class RetargetBatchResult:
    input_path: Path
    output_path: Path
    frame_count: int


class RetargetBatchProcessor:
    """Reusable, renderer-independent processor compiled from one config."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = RetargetConfig.load(self.config_path)
        source_path = _asset_path(self.config_path, self.config.source_skeleton)
        target_path = _asset_path(self.config_path, self.config.target_skeleton)
        self.source_coordinates = CoordinateSystem(self.config.source_coordinate_system)
        self.target_coordinates = CoordinateSystem(self.config.target_coordinate_system)
        self.output_coordinates = CoordinateSystem(self.config.output_coordinate_system)
        canonical = CoordinateSystem.Y_UP_Z_FORWARD
        self.source_skeleton = convert_skeleton_coordinates(
            _load_skeleton(
                source_path,
                has_armature_joint=self.config.source_has_armature_joint,
            ),
            source=self.source_coordinates,
            target=canonical,
        )
        self.target_skeleton = convert_skeleton_coordinates(
            _load_skeleton(target_path),
            source=self.target_coordinates,
            target=canonical,
        )
        self.retargeter = Retargeter(
            self.source_skeleton, self.target_skeleton, self.config
        )

    def _check_compatible(self, motion) -> None:
        if motion.node_names() != self.source_skeleton.node_names():
            raise ValueError(
                "motion joint names do not match the config source skeleton"
            )
        if motion.parent_indices() != self.source_skeleton.parent_indices():
            raise ValueError(
                "motion hierarchy does not match the config source skeleton"
            )

    def process_motion(self, motion):
        canonical = CoordinateSystem.Y_UP_Z_FORWARD
        source = convert_motion_coordinates(
            motion, source=self.source_coordinates, target=canonical
        )
        self._check_compatible(source)
        result = self.retargeter.retarget_motion(source)
        return convert_motion_coordinates(
            result, source=canonical, target=self.output_coordinates
        )

    def process_file(
        self, input_path: str | Path, output_path: str | Path
    ) -> RetargetBatchResult:
        from ..exports import save_motion_bvh

        source_path = Path(input_path).expanduser().resolve()
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = self.process_motion(
            _load_motion(
                source_path,
                has_armature_joint=self.config.source_has_armature_joint,
            )
        )
        saved = save_motion_bvh(destination, result)
        return RetargetBatchResult(source_path, Path(saved), result.num_frames())

    def process_files(
        self,
        input_paths: Iterable[str | Path],
        output_directory: str | Path,
        *,
        suffix: str = "_retargeted",
    ) -> list[RetargetBatchResult]:
        directory = Path(output_directory).expanduser().resolve()
        return [
            self.process_file(path, directory / f"{Path(path).stem}{suffix}.bvh")
            for path in input_paths
        ]
