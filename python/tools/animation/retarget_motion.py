"""Retarget one or more motions without creating a window or renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from kangengine.animation import RetargetBatchProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, nargs="+")
    inputs.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suffix", default="_retargeted")
    args = parser.parse_args()

    paths = args.input
    if args.input_dir is not None:
        paths = sorted(
            path
            for path in args.input_dir.iterdir()
            if path.suffix.lower() in {".bvh", ".fbx"}
        )
    assert paths is not None
    processor = RetargetBatchProcessor(args.config)
    if args.output is not None:
        if len(paths) != 1:
            parser.error("--output requires exactly one input motion")
        results = [processor.process_file(paths[0], args.output)]
    else:
        if args.output_dir is None:
            parser.error("provide --output or --output-dir")
        results = processor.process_files(paths, args.output_dir, suffix=args.suffix)
    for result in results:
        print(
            f"{result.input_path} -> {result.output_path} ({result.frame_count} frames)"
        )


if __name__ == "__main__":
    main()
