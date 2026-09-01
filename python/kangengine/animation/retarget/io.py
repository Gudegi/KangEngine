"""Source-motion loading shared by angle and IK retarget processors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .profile import MotionSourceProfile

if TYPE_CHECKING:
    from .. import SkeletonMotion, SkeletonTree
    from ..coordinates import CoordinateSystem


def load_retarget_motion(
    path: str | Path,
    profile: MotionSourceProfile,
    *,
    target_coordinate_system: CoordinateSystem | str | None = None,
    pair_scale: float = 1.0,
) -> SkeletonMotion:
    from ... import asset
    from ..coordinates import CoordinateSystem, convert_motion_coordinates

    source_path = Path(path).expanduser().resolve()
    scale = profile.translation_unit_scale * float(pair_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("combined source motion scale must be finite and positive")
    suffix = source_path.suffix.lower()
    if suffix == ".bvh":
        motion = asset.BVHLoader.load_motion(
            bvh_path=str(source_path),
            scale=scale,
            has_armature_joint=profile.has_armature_joint,
        )
    elif suffix == ".fbx":
        motion = asset.FBXLoader.load_motion(
            fbx_path=str(source_path),
            scale=scale,
        )
    else:
        raise ValueError(f"unsupported source motion format: {suffix}")
    if target_coordinate_system is None:
        return motion
    return convert_motion_coordinates(
        motion,
        source=CoordinateSystem(profile.coordinate_system),
        target=CoordinateSystem(target_coordinate_system),
    )


def load_retarget_dataset_motion(
    path: str | Path,
    skeleton: SkeletonTree,
    *,
    target_coordinate_system: CoordinateSystem,
    root_translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ascend_person: str = "second_person",
    ascend_pose_key: str = "opt_pose",
    ascend_translation_key: str = "opt_trans",
    ascend_fps: float = 30.0,
) -> SkeletonMotion:
    """Load an AMASS or Ascend clip against a retarget reference skeleton."""

    from .. import datasets

    dataset_type = datasets.detect_dataset_type(path)
    if dataset_type is None:
        raise ValueError(f"unsupported dataset motion format: {Path(path).suffix}")
    if dataset_type is datasets.DatasetType.AMASS:
        config = datasets.AMASSConfig(
            model_type="smpl",
            skeleton=skeleton,
            root_translation_offset=root_translation_offset,
        )
    else:
        config = datasets.AscendConfig(
            person_key=ascend_person,
            pose_key=ascend_pose_key,
            translation_key=ascend_translation_key,
            default_fps=ascend_fps,
            skeleton=skeleton,
            root_translation_offset=root_translation_offset,
        )
    return datasets.load_motion(
        path,
        dataset=dataset_type,
        target_world=target_coordinate_system,
        config=config,
    )


def scale_skeleton_motion(motion: SkeletonMotion, scale: float) -> SkeletonMotion:
    """Uniformly scale skeleton offsets and root translations."""
    from .. import SkeletonMotion, SkeletonTree

    factor = float(scale)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("motion scale must be finite and positive")
    if abs(factor - 1.0) <= 1.0e-12:
        return motion
    tree = motion.skeleton_tree
    translations = np.asarray(
        [
            (
                tree.local_translation(index).x,
                tree.local_translation(index).y,
                tree.local_translation(index).z,
            )
            for index in range(tree.num_joints())
        ],
        dtype=np.float32,
    )
    rotations = np.asarray(
        [
            (
                tree.local_rotation(index).w,
                tree.local_rotation(index).x,
                tree.local_rotation(index).y,
                tree.local_rotation(index).z,
            )
            for index in range(tree.num_joints())
        ],
        dtype=np.float32,
    )
    scaled_tree = SkeletonTree(
        tree.node_names(),
        tree.parent_indices(),
        translations * factor,
        rotations,
    )
    return SkeletonMotion.from_arrays(
        scaled_tree,
        np.asarray(motion.root_translations(), dtype=np.float32) * factor,
        motion.local_rotations_wxyz(),
        motion.fps(),
        motion.motion_name(),
    )


__all__ = [
    "load_retarget_motion",
    "scale_skeleton_motion",
]
