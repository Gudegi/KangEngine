"""AMASS motion archive inspection and skeleton-motion loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .._core import _ke

if TYPE_CHECKING:
    from ..animation import SkeletonMotion, SkeletonTree


@dataclass(frozen=True)
class AMASSInfo:
    """Metadata read from an AMASS NPZ archive."""

    path: Path
    gender: str
    betas: np.ndarray
    fps: float
    num_frames: int


class AMASSLoader:
    """Load AMASS body and hand poses as a native ``SkeletonMotion``."""

    @staticmethod
    def inspect(path: str | Path) -> AMASSInfo:
        source_path = Path(path).expanduser().resolve()
        with np.load(source_path, allow_pickle=False) as source:
            poses = source["poses"]
            return AMASSInfo(
                source_path,
                str(np.asarray(source["gender"]).item()).lower(),
                np.asarray(source["betas"], dtype=np.float32),
                float(np.asarray(source["mocap_framerate"]).item()),
                int(poses.shape[0]),
            )

    @staticmethod
    def load_motion(
        path: str | Path,
        skeleton_tree: SkeletonTree,
        *,
        model_type: str = "smplx",
        up_axis=_ke.UpAxis.Y,
        scale: float = 1.0,
        motion_name: str | None = None,
    ) -> SkeletonMotion:
        """Load an AMASS NPZ directly into ``animation.SkeletonMotion``."""
        import torch

        from ..utils import quat_wxyz_from_angle_axis, quat_wxyz_multiply

        source_path = Path(path).expanduser().resolve()
        with np.load(source_path, allow_pickle=False) as source:
            poses = np.asarray(source["poses"], dtype=np.float32)
            translations = np.asarray(source["trans"], dtype=np.float32)
            fps = float(np.asarray(source["mocap_framerate"]).item())
        if poses.ndim != 2 or poses.shape[1] < 156:
            raise ValueError("AMASS poses must have shape [frames, at least 156]")
        if translations.shape != (len(poses), 3):
            raise ValueError("AMASS trans must have shape [frames, 3]")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("AMASS mocap_framerate must be positive")

        rotvec = torch.from_numpy(poses[:, :156].reshape(-1, 52, 3))
        angles = torch.linalg.vector_norm(rotvec, dim=-1)
        source_rotations = quat_wxyz_from_angle_axis(angles, rotvec)

        kind = str(model_type).lower()
        if kind == "smpl":
            if skeleton_tree.num_joints() != 24:
                raise ValueError("SMPL motion requires a 24-joint skeleton_tree")
            rotations = torch.zeros((len(poses), 24, 4), dtype=torch.float32)
            rotations[..., 0] = 1.0
            rotations[:, :22] = source_rotations[:, :22]
        elif kind == "smplh":
            if skeleton_tree.num_joints() != 52:
                raise ValueError("SMPL-H motion requires a 52-joint skeleton_tree")
            rotations = source_rotations
        elif kind == "smplx":
            if skeleton_tree.num_joints() != 55:
                raise ValueError("SMPL-X motion requires a 55-joint skeleton_tree")
            rotations = torch.zeros((len(poses), 55, 4), dtype=torch.float32)
            rotations[..., 0] = 1.0
            rotations[:, :22] = source_rotations[:, :22]
            rotations[:, 25:] = source_rotations[:, 22:]
        else:
            raise ValueError("model_type must be 'smpl', 'smplh', or 'smplx'")

        if up_axis == _ke.UpAxis.Y:
            angle = torch.tensor(-np.pi / 2.0, dtype=torch.float32)
            axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            basis = quat_wxyz_from_angle_axis(angle, axis)
            rotations[:, 0] = quat_wxyz_multiply(basis, rotations[:, 0])
            translations = translations[:, [0, 2, 1]]
            translations[:, 2] *= -1.0
        elif up_axis != _ke.UpAxis.Z:
            raise ValueError("AMASSLoader supports only Y-up or Z-up output")
        translations *= float(scale)

        return _ke.animation.SkeletonMotion.from_arrays(
            skeleton_tree,
            translations,
            rotations,
            fps,
            motion_name or source_path.name,
        )


__all__ = ["AMASSInfo", "AMASSLoader"]
