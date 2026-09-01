"""Load/Save functions for canonical articulation motion arrays."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from .articulation_motion import ArticulationCoordinateLayout, ArticulationMotion


def save_articulation_motion_npz(motion: ArticulationMotion, path: str | Path) -> Path:
    """Save canonical q/qd arrays and layout identity without pickled objects."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        version=np.asarray(1, dtype=np.int32),
        q=np.asarray(motion.q, dtype=np.float32),
        qd=np.asarray(motion.qd, dtype=np.float32),
        fps=np.asarray(motion.fps(), dtype=np.float64),
        motion_name=np.asarray(motion.motion_name()),
        model_signature=np.asarray(motion.layout.model_signature),
    )
    return destination


def load_articulation_motion_npz(
    path: str | Path,
    layout: ArticulationCoordinateLayout,
) -> ArticulationMotion:
    """Load canonical motion, warning when the target model identity differs."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        version = int(archive["version"])
        if version != 1:
            raise ValueError(f"unsupported articulation motion version: {version}")
        q = np.asarray(archive["q"], dtype=np.float32)
        qd = np.asarray(archive["qd"], dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != layout.nq:
            raise ValueError(
                f"articulation motion q shape {q.shape} is incompatible with "
                f"target layout nq={layout.nq}"
            )
        if qd.ndim != 2 or qd.shape != (q.shape[0], layout.nv):
            raise ValueError(
                f"articulation motion qd shape {qd.shape} is incompatible with "
                f"{q.shape[0]} frames and target layout nv={layout.nv}"
            )
        signature = str(archive["model_signature"])
        if signature != layout.model_signature:
            warnings.warn(
                "articulation motion model signature differs from the target "
                "layout; loading because q/qd dimensions match, but joint "
                "ordering or coordinate semantics may differ "
                f"(motion={signature}, target={layout.model_signature})",
                RuntimeWarning,
                stacklevel=2,
            )
        return ArticulationMotion.from_arrays(
            layout=layout,
            q=q,
            qd=qd,
            fps=float(archive["fps"]),
            motion_name=str(archive["motion_name"]),
        )


__all__ = ["load_articulation_motion_npz", "save_articulation_motion_npz"]
