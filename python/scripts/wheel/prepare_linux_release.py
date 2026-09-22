"""Repair Linux wheels and check metadata before a separate upload step."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("python/dist/release"))
    args = parser.parse_args()
    for wheel in args.wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata_path = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
            if any("@" in dep for dep in metadata.get_all("Requires-Dist", [])):
                raise ValueError(
                    f"Direct URL dependencies cannot be published: {wheel}"
                )

    # Use a fresh directory so stale release artifacts cannot pass these checks.
    with tempfile.TemporaryDirectory(prefix="kangengine-release-") as temp:
        for wheel in args.wheels:
            subprocess.run(
                [
                    "auditwheel",
                    "repair",
                    "--plat",
                    "manylinux_2_39_x86_64",
                    # Supplied by the pinned NVIDIA pip dependency.
                    "--exclude",
                    "libcudart.so.13",
                    "--wheel-dir",
                    temp,
                    str(wheel.resolve()),
                ],
                check=True,
            )
        repaired = sorted(Path(temp).glob("*.whl"))
        if len(repaired) != len(args.wheels):
            raise RuntimeError("Expected one repaired artifact per input wheel")
        for wheel in repaired:
            if wheel.stat().st_size > 100_000_000:
                raise ValueError(f"Wheel exceeds 100 MB: {wheel.name}")
        subprocess.run(["twine", "check", "--strict", *map(str, repaired)], check=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for wheel in repaired:
            output = args.output_dir / wheel.name
            shutil.copy2(wheel, output)
            print(f"Prepared: {output} ({output.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()
