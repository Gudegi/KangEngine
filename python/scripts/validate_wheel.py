"""Build and validate a KangEngine wheel without modifying the active venv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


STALE_MODULES = {
    "kangengine/app.py",
    "kangengine/motion_editor.py",
    "kangengine/motion_modules.py",
    "kangengine/physics.py",
    "kangengine/sensor.py",
    "kangengine/sim.py",
    "kangengine/terrain.py",
}


def run(*args: str, **kwargs) -> None:
    subprocess.run(args, check=True, **kwargs)


def stage_project(source: Path, destination: Path) -> None:
    shutil.copy2(source / "pyproject.toml", destination)
    shutil.copy2(source / "setup.py", destination)
    shutil.copy2(source / "MANIFEST.in", destination)
    readme = source / "README.md"
    if not readme.exists():
        readme = source.parent / "README.md"
    shutil.copy2(readme, destination / "README.md")

    shutil.copytree(
        source / "kangengine",
        destination / "kangengine",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".*",
            "external",
        ),
    )


def inspect_wheel(wheel: Path) -> None:
    if wheel.name.endswith("-none-any.whl"):
        raise AssertionError(f"native wheel has a pure-Python tag: {wheel.name}")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        if "kangengine/_kangengine.so" not in names:
            raise AssertionError("wheel does not contain kangengine/_kangengine.so")
        external = sorted(
            name for name in names if name.startswith("kangengine/assets/external/")
        )
        if external:
            raise AssertionError(f"wheel contains external assets: {external[0]}")
        stale = sorted(names & STALE_MODULES)
        if stale:
            raise AssertionError(f"wheel contains stale modules: {', '.join(stale)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--expect-no-usd", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    source = Path(__file__).resolve().parents[1]
    # Keep a venv interpreter path intact instead of resolving its symlink to
    # the base interpreter; the smoke tests reuse the venv's dependencies but
    # install KangEngine itself only into the temporary target directory.
    python_executable = os.path.abspath(args.python)
    smoke = source / "examples" / "smoke"
    if args.expect_no_usd:
        source_environment = os.environ.copy()
        source_environment["PYTHONPATH"] = str(source)
        run(
            python_executable,
            "-c",
            (
                "import kangengine as ke; "
                "assert not ke.scene.has_usd_support(), "
                "'native module unexpectedly contains USD support'"
            ),
            cwd=source.parent,
            env=source_environment,
        )

    with tempfile.TemporaryDirectory(prefix="kangengine-wheel-") as temp_name:
        temp = Path(temp_name)
        staged = temp / "project"
        wheelhouse = temp / "wheelhouse"
        target = temp / "site-packages"
        staged.mkdir()
        wheelhouse.mkdir()
        target.mkdir()
        stage_project(source, staged)

        run(
            args.uv,
            "build",
            "--wheel",
            "--out-dir",
            str(wheelhouse),
            str(staged),
        )
        wheels = list(wheelhouse.glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"expected one wheel, found {len(wheels)}")
        wheel = wheels[0]
        inspect_wheel(wheel)

        if args.output_dir is not None:
            output_dir = args.output_dir.absolute()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_wheel = output_dir / wheel.name
            shutil.copy2(wheel, output_wheel)
            print(f"Built wheel: {output_wheel}")

        if args.build_only:
            return

        run(
            args.uv,
            "pip",
            "install",
            "--python",
            python_executable,
            "--target",
            str(target),
            "--no-deps",
            str(wheel),
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(target)
        environment["PYTHONPYCACHEPREFIX"] = str(temp / "pycache")
        run(
            python_executable,
            str(smoke / "public_api_surface_smoke.py"),
            cwd=temp,
            env=environment,
        )
        run(
            python_executable,
            str(smoke / "public_stub_surface_smoke.py"),
            cwd=temp,
            env=environment,
        )

        print(f"PASS: isolated wheel {wheel.name}")


if __name__ == "__main__":
    main()
