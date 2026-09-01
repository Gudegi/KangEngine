"""Run angle retargeting without creating a window or renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from kangengine.animation import (
    AngleRetargetProcessor,
    AngleRetargetResult,
    datasets,
)
from kangengine.exports import save_motion_bvh
from kangengine.utils import CoordinateSystem


_MOTION_SUFFIXES = {".bvh", ".fbx", ".npz", ".npy", ".pkl"}
_ASCEND_SUFFIXES = {".npz", ".npy", ".pkl"}


def _process_path(
    processor: AngleRetargetProcessor,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> AngleRetargetResult:
    if input_path.suffix.lower() not in _ASCEND_SUFFIXES:
        return processor.process_file(input_path, output_path)
    source = datasets.load_motion(
        input_path,
        dataset=datasets.DatasetType.ASCEND,
        target_world=CoordinateSystem.Y_UP_Z_FORWARD,
        config=datasets.AscendConfig(
            person_key=args.ascend_person,
            pose_key=args.ascend_pose_key,
            translation_key=args.ascend_translation_key,
            default_fps=args.ascend_fps,
            skeleton=processor.source_skeleton,
            root_translation_offset=processor.config.source_bind_root,
        ),
    )
    result = processor.process_motion(source)
    saved = save_motion_bvh(output_path.expanduser().resolve(), result)
    return AngleRetargetResult(
        input_path.expanduser().resolve(), Path(saved), result.num_frames()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, nargs="+")
    inputs.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suffix", default="_retargeted")
    parser.add_argument("--ascend-person", default="second_person")
    parser.add_argument("--ascend-pose-key", default="opt_pose")
    parser.add_argument("--ascend-translation-key", default="opt_trans")
    parser.add_argument("--ascend-fps", type=float, default=30.0)
    args = parser.parse_args()

    paths = args.input
    if args.input_dir is not None:
        paths = sorted(
            path
            for path in args.input_dir.iterdir()
            if path.suffix.lower() in _MOTION_SUFFIXES
        )
    assert paths is not None
    processor = AngleRetargetProcessor(args.config)
    if args.output is not None:
        if len(paths) != 1:
            parser.error("--output requires exactly one input motion")
        results = [_process_path(processor, paths[0], args.output, args)]
    else:
        if args.output_dir is None:
            parser.error("provide --output or --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results = [
            _process_path(
                processor,
                path,
                args.output_dir / f"{path.stem}{args.suffix}.bvh",
                args,
            )
            for path in paths
        ]
    for result in results:
        print(
            f"{result.input_path} -> {result.output_path} ({result.frame_count} frames)"
        )


if __name__ == "__main__":
    main()
