"""Plan an offline convex-decomposition cache without running a backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjcf_convex_cache import plan_mjcf_convex_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan MJCF collision meshes and emit the manifest that a future "
            "CoACD offline builder will consume. No decomposition is run."
        )
    )
    parser.add_argument("mjcf", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="coacd")
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Record a backend option in the plan manifest.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write manifest.json under --output-dir; otherwise print only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    options = dict(_parse_option(item) for item in args.backend_option)
    manifest = plan_mjcf_convex_cache(
        args.mjcf,
        args.output_dir,
        backend=args.backend,
        backend_options=options,
    )
    text = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    print(text)
    if args.write_manifest:
        destination = manifest.write(args.output_dir / "manifest.json")
        print(f"Wrote plan manifest: {destination}")


def _parse_option(value: str) -> tuple[str, object]:
    key, separator, raw = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("backend options must use KEY=VALUE")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


if __name__ == "__main__":
    main()
