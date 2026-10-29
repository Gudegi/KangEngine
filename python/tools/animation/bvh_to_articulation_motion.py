"""Convert a target-aligned BVH motion to canonical ArticulationMotion NPZ.

The BVH must already use the target MJCF body names and hierarchy. Run angle
retargeting first when converting a motion from a different skeleton.

Example::

    python python/tools/animation/bvh_to_articulation_motion.py \
        path/to/retargeted_motion.bvh --mjcf path/to/robot.xml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import kangengine as ke


_COORDINATE_CHOICES = tuple(system.value for system in ke.utils.CoordinateSystem)


def convert(
    bvh_path: Path,
    mjcf_path: Path,
    output_path: Path,
    *,
    input_coordinates: ke.utils.CoordinateSystem,
    clamp_to_limits: bool,
) -> tuple[Path, int, float, tuple[int, int], tuple[int, int]]:
    """Convert one BVH and return output metadata for the CLI summary."""

    data = ke.asset.MJCFLoader.load(str(mjcf_path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data,
        free_root=True,
    )
    source = ke.asset.BVHLoader.load_motion(str(bvh_path))
    motion = ke.animation.convert_motion_coordinates(
        source,
        source=input_coordinates,
        target=ke.utils.CoordinateSystem.Z_UP_X_FORWARD,
    )

    mapper = ke.animation.ArticulationMotionMapper(layout)
    mapped = mapper.to_articulation_motion(
        motion,
        clamp_to_limits=clamp_to_limits,
    )
    destination = ke.animation.save_articulation_motion_npz(
        mapped.motion,
        output_path,
    )
    residuals = np.asarray(mapped.residual_angles, dtype=np.float32)
    max_residual_degrees = (
        0.0 if residuals.size == 0 else math.degrees(float(np.max(residuals)))
    )
    return (
        Path(destination),
        mapped.motion.num_frames(),
        max_residual_degrees,
        tuple(mapped.motion.q.shape),
        tuple(mapped.motion.qd.shape),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Target-aligned BVH motion")
    parser.add_argument("--mjcf", type=Path, required=True, help="Target MJCF model")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ (default: input filename with .npz suffix)",
    )
    parser.add_argument(
        "--input-coordinates",
        choices=_COORDINATE_CHOICES,
        default=ke.utils.CoordinateSystem.Y_UP_Z_FORWARD.value,
        help="BVH coordinate system (default: y_up_z_forward)",
    )
    parser.add_argument(
        "--clamp-to-limits",
        action="store_true",
        help="Clamp revolute coordinates to MJCF joint limits",
    )
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    mjcf = args.mjcf.expanduser().resolve()
    output = (
        source.with_suffix(".npz")
        if args.output is None
        else args.output.expanduser().resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if not mjcf.is_file():
        raise FileNotFoundError(mjcf)

    destination, frames, max_residual, q_shape, qd_shape = convert(
        source,
        mjcf,
        output,
        input_coordinates=ke.utils.CoordinateSystem(args.input_coordinates),
        clamp_to_limits=args.clamp_to_limits,
    )
    print(f"{source} -> {destination}")
    print(
        f"frames={frames} q={q_shape} qd={qd_shape} "
        f"max_residual={max_residual:.6g} deg "
        f"clamped={'yes' if args.clamp_to_limits else 'no'}"
    )
    if max_residual > 5.0:
        print(
            "warning: some BVH rotations cannot be represented by the target "
            "joint axes or requested limits; the reported residual is discarded"
        )


if __name__ == "__main__":
    main()
