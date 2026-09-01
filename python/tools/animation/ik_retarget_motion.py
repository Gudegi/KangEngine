"""Headless articulation IK retargeting from a versioned pair config."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

# Avoid JAX reserving most GPU memory for this windowed offline tool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import kangengine as ke
import numpy as np
from kangengine.adapters.pyroki import (
    PyrokiTrajectoryState,
    PyrokiTrajectoryIKConfig,
)


def _slice_motion(
    motion, start: int, end: int | None, stride: int, target_frames: int | None
):
    if stride <= 0:
        raise ValueError("stride must be positive")
    stop = motion.num_frames() if end is None else min(end, motion.num_frames())
    if start < 0 or start >= stop:
        raise ValueError(f"invalid frame range: [{start}, {stop})")
    root = motion.root_translations()[start:stop:stride]
    rotations = motion.local_rotations_wxyz()[start:stop:stride]
    if target_frames is not None:
        root = root[:target_frames]
        rotations = rotations[:target_frames]
    return ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=motion.skeleton_tree,
        root_translations=root,
        local_rotations_wxyz=rotations,
        fps=motion.fps() / stride,
        motion_name=motion.motion_name(),
    )


def _slice_contiguous_motion(motion, start: int, end: int):
    return ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=motion.skeleton_tree,
        root_translations=motion.root_translations()[start:end],
        local_rotations_wxyz=motion.local_rotations_wxyz()[start:end],
        fps=motion.fps(),
        motion_name=f"{motion.motion_name()}_{start:06d}_{end:06d}",
    )


def _clip_output_path(path: Path, index: int) -> Path:
    suffix = path.suffix or ".npz"
    return path.with_name(f"{path.stem}_{index:03d}{suffix}")


def _trajectory_state(motion, retarget_config) -> PyrokiTrajectoryState:
    scalar_types = {
        ke.animation.ArticulationCoordinateType.REVOLUTE,
        ke.animation.ArticulationCoordinateType.PRISMATIC,
    }
    blocks = {
        block.joint_name: block
        for block in motion.layout.blocks
        if block.type in scalar_types
    }
    missing = sorted(set(retarget_config.joint_order) - set(blocks))
    if missing:
        raise ValueError(
            "initial motion is missing config joints: " + ", ".join(missing)
        )
    joints = np.stack(
        [motion.q[:, blocks[name].q_offset] for name in retarget_config.joint_order],
        axis=1,
    ).astype(np.float32, copy=False)
    free_blocks = [
        block
        for block in motion.layout.blocks
        if block.type == ke.animation.ArticulationCoordinateType.FREE
    ]
    if len(free_blocks) != 1:
        raise ValueError("initial motion requires one free-root block")
    free = free_blocks[0]
    return PyrokiTrajectoryState(
        joint_configurations=joints,
        root_positions=motion.q[:, free.q_offset : free.q_offset + 3],
        root_rotations_wxyz=motion.q[:, free.q_offset + 3 : free.q_offset + 7],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-motion",
        type=Path,
        help="Canonical ArticulationMotion NPZ used as the initial q(t).",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride; defaults to 1 to preserve the source FPS.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=0,
        help="Maximum frames after subsampling; use 0 for the full clip.",
    )
    parser.add_argument(
        "--no-contacts",
        action="store_true",
        help="Disable motion-module foot contacts and contact-aware grounding.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--window-overlap", type=int, default=32)
    parser.add_argument("--window-continuity-weight", type=float, default=10.0)
    parser.add_argument("--whole-trajectory-max-frames", type=int, default=1000)
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=0,
        help=(
            "Solve independent clips of this length and write OUTPUT_000.npz, "
            "OUTPUT_001.npz, ...; use 0 for one continuous output."
        ),
    )
    args = parser.parse_args()

    if not args.verbose_solver:
        from loguru import logger

        logger.disable("jaxls")
        logger.disable("pyroki")

    processor = ke.animation.IKRetargetProcessor(args.config)
    retarget_config = processor.config
    source = processor.load_source(args.input)
    stride = args.stride
    source = _slice_motion(
        source,
        args.start_frame,
        args.end_frame,
        stride,
        None if args.target_frames == 0 else args.target_frames,
    )
    layout = processor.layout
    initial_trajectory = None
    if args.initial_motion is not None:
        initial_motion = ke.animation.load_articulation_motion_npz(
            args.initial_motion.expanduser().resolve(), layout
        )
        initial_trajectory = _trajectory_state(initial_motion, retarget_config)
        if (
            np.asarray(initial_trajectory.joint_configurations).shape[0]
            != source.num_frames()
        ):
            raise ValueError(
                "initial motion frame count must match the sliced source motion"
            )
    config = PyrokiTrajectoryIKConfig(
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        window_continuity_weight=args.window_continuity_weight,
        whole_trajectory_max_frames=args.whole_trajectory_max_frames,
    )
    if args.clip_frames < 0:
        raise ValueError("clip_frames must be non-negative")
    if args.clip_frames:
        frame_ranges = [
            (start, min(start + args.clip_frames, source.num_frames()))
            for start in range(0, source.num_frames(), args.clip_frames)
        ]
    else:
        frame_ranges = [(0, source.num_frames())]

    progress = None
    if not args.no_progress:
        from tqdm.auto import tqdm

        progress = tqdm(total=source.num_frames(), unit="frame", desc="Retargeting")

    outputs = []
    try:
        for clip_index, (start, end) in enumerate(frame_ranges):
            clip = _slice_contiguous_motion(source, start, end)
            progress_callback = None
            if progress is not None:
                completed_before = start

                def update_progress(state, completed_before=completed_before) -> None:
                    progress.set_postfix(
                        clip=f"{clip_index + 1}/{len(frame_ranges)}",
                        window=f"{state.window_index}/{state.window_count}",
                        rmse=f"{state.position_rmse:.4f}m",
                        iterations=state.iterations,
                        converged=state.converged,
                        refresh=False,
                    )
                    completed = completed_before + state.frames_completed
                    progress.update(max(0, completed - progress.n))

                progress_callback = update_progress

            clip_config = config
            if args.clip_frames:
                clip_config = replace(
                    config,
                    whole_trajectory_max_frames=max(
                        config.whole_trajectory_max_frames, clip.num_frames()
                    ),
                )
            result = processor.process_motion(
                clip,
                initial_state=(
                    None
                    if initial_trajectory is None
                    else initial_trajectory.slice(start, end)
                ),
                detect_contacts=not args.no_contacts,
                solver_config=clip_config,
                progress_callback=progress_callback,
            )
            output_path = (
                _clip_output_path(args.output, clip_index)
                if args.clip_frames
                else args.output
            )
            output = ke.animation.save_articulation_motion_npz(
                result.motion, output_path
            )
            outputs.append((output, result))
    finally:
        if progress is not None:
            progress.close()
    for output, result in outputs:
        print(
            f"saved {result.motion.num_frames()} frames to {output}; "
            f"position RMSE {result.solve.initial_position_rmse:.6f} -> "
            f"{result.solve.position_rmse:.6f} m; "
            f"iterations={result.solve.iterations}, "
            f"converged={result.solve.converged}"
        )


if __name__ == "__main__":
    main()
