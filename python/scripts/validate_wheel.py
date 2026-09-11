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


def resolve_usd_runtime(runtime: Path) -> Path:
    runtime = runtime.absolute()
    if (runtime / "lib" / "usd" / "plugInfo.json").is_file():
        return runtime
    candidates = sorted(
        child
        for child in runtime.iterdir()
        if (child / "lib" / "usd" / "plugInfo.json").is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one USD vcpkg runtime below {runtime}, found {len(candidates)}"
        )
    return candidates[0]


def bundle_usd_runtime(package: Path, runtime: Path) -> None:
    runtime = resolve_usd_runtime(runtime)
    lib_dir = runtime / "lib"
    if not (lib_dir / "usd" / "plugInfo.json").is_file():
        raise FileNotFoundError(f"USD plugin resources not found under {lib_dir}")

    destination = package / "_native" / "lib"
    destination.mkdir(parents=True)
    libraries = sorted(lib_dir.glob("libusd_*.dylib"))
    libraries += sorted(lib_dir.glob("libusd_*.so*"))
    if not libraries:
        raise FileNotFoundError(f"USD shared libraries not found under {lib_dir}")
    for library in libraries:
        if library.is_file():
            shutil.copy2(library, destination / library.name)
    shutil.copytree(lib_dir / "usd", destination / "usd")

    licenses = package / "licenses"
    licenses.mkdir()
    for package_name, output_name in (
        ("usd", "OpenUSD.txt"),
        ("tbb", "oneTBB.txt"),
        ("zlib", "zlib.txt"),
    ):
        copyright_file = runtime / "share" / package_name / "copyright"
        if copyright_file.is_file():
            shutil.copy2(copyright_file, licenses / output_name)

    extension = package / "_kangengine.so"
    if sys.platform == "darwin":
        run("install_name_tool", "-delete_rpath", str(lib_dir), str(extension))
        run(
            "install_name_tool",
            "-add_rpath",
            "@loader_path/_native/lib",
            str(extension),
        )


def stage_project(
    source: Path,
    destination: Path,
    usd_runtime: Path | None = None,
    extension: Path | None = None,
) -> None:
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
    if extension is not None:
        shutil.copy2(extension, destination / "kangengine" / "_kangengine.so")
    if usd_runtime is not None:
        bundle_usd_runtime(destination / "kangengine", usd_runtime)


def inspect_wheel(wheel: Path, expect_usd: bool) -> None:
    if wheel.name.endswith("-none-any.whl"):
        raise AssertionError(f"native wheel has a pure-Python tag: {wheel.name}")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        if "kangengine/_kangengine.so" not in names:
            raise AssertionError("wheel does not contain kangengine/_kangengine.so")
        bundled_usd = any(
            name.startswith("kangengine/_native/lib/libusd_") for name in names
        )
        if expect_usd and not bundled_usd:
            raise AssertionError("USD wheel does not contain bundled libraries")
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
    parser.add_argument("--expect-usd", action="store_true")
    parser.add_argument("--usd-runtime", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.expect_usd and args.expect_no_usd:
        parser.error("--expect-usd and --expect-no-usd are mutually exclusive")
    if args.expect_usd and args.usd_runtime is None:
        parser.error("--expect-usd requires --usd-runtime")
    if args.extension is not None and not args.extension.is_file():
        parser.error(f"--extension does not exist: {args.extension}")

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
        stage_project(source, staged, args.usd_runtime, args.extension)

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
        inspect_wheel(wheel, args.expect_usd)

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
        if args.expect_usd:
            run(
                python_executable,
                "-c",
                (
                    "import kangengine as ke; "
                    "scene = ke.scene.USDScene(); "
                    "scene.create_cube('/Root/Cube', 2.0); "
                    "assert scene.save_scene('scene.usda'); "
                    "loaded = ke.scene.USDScene(); "
                    "assert loaded.load_scene('scene.usda'); "
                    "assert loaded.save_scene('scene.usdc')"
                ),
                cwd=temp,
                env=environment,
            )
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
