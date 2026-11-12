"""Package an existing native build into release wheels without rebuilding it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tomllib
import zipfile

from validate_wheel import inspect_wheel, run
from wheel_licenses import inspect_license_bundle, write_license_bundle


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
    destination.mkdir(parents=True, exist_ok=True)
    libraries = sorted(lib_dir.glob("libusd_*.dylib"))
    libraries += sorted(lib_dir.glob("libusd_*.so*"))
    if not libraries:
        raise FileNotFoundError(f"USD shared libraries not found under {lib_dir}")
    for library in libraries:
        if library.is_file():
            shutil.copy2(library, destination / library.name)
    shutil.copytree(lib_dir / "usd", destination / "usd")

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
    physx_runtime: Path | None = None,
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
            "_native",
            "licenses",
            "*.so",
            "*.dylib",
        ),
    )
    compiled = extension if extension is not None else source / "kangengine/_kangengine.so"
    shutil.copy2(compiled, destination / "kangengine" / "_kangengine.so")
    # Use the selected build's notices, never stale files from the source package.
    licenses = compiled.parent / "licenses"
    manifest = licenses / "manifest.txt"
    staged_licenses = destination / "kangengine/licenses"
    write_license_bundle(
        staged_licenses,
        {name: licenses / name for name in manifest.read_text().splitlines()},
        json.loads((licenses / "versions.json").read_text()),
    )
    if usd_runtime is not None:
        bundle_usd_runtime(destination / "kangengine", usd_runtime)
    if sys.platform == "linux":
        from wheel_linux import bundle_linux_runtime

        bundle_linux_runtime(
            destination / "kangengine",
            physx_runtime,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--expect-usd", action="store_true")
    parser.add_argument("--usd-runtime", type=Path)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("python/dist/release"))
    parser.add_argument("--physx-runtime", type=Path)
    parser.add_argument("--physx-license", type=Path)
    args = parser.parse_args()
    if not args.extension.is_file():
        parser.error(f"Native extension not found: {args.extension}; run make build_wheel first")
    if args.expect_usd and args.usd_runtime is None:
        parser.error("--expect-usd requires --usd-runtime")
    if sys.platform != "linux" and args.physx_runtime:
        parser.error("CUDA/PhysX GPU runtime staging is supported only on Linux")

    source = Path(__file__).resolve().parents[2]
    cuda_runtime_version = None
    if sys.platform == "linux":
        project = tomllib.loads((source / "pyproject.toml").read_text())
        requirements = [
            r
            for r in project["project"]["dependencies"]
            if r.startswith("nvidia-cuda-runtime==")
        ]
        if len(requirements) != 1:
            raise ValueError(
                "Expected one pinned CUDA runtime dependency in pyproject.toml"
            )
        match = re.fullmatch(
            r"nvidia-cuda-runtime==(13\.\d+\.\d+);\s*sys_platform == ['\"]linux['\"]",
            requirements[0],
        )
        if match is None:
            raise ValueError("Expected a Linux-only CUDA 13 runtime dependency")
        cuda_runtime_version = match.group(1)
    # Use this interpreter for building; validation gets its own environment.
    python_executable = os.path.abspath(args.python)
    with tempfile.TemporaryDirectory(prefix="kangengine-wheel-") as temp_name:
        temp = Path(temp_name)
        staged = temp / "project"
        wheelhouse = temp / "wheelhouse"
        staged.mkdir()
        wheelhouse.mkdir()
        runtime_wheel = None
        runtime_version = None
        if sys.platform == "linux":
            from wheel_linux import stage_physx_runtime

            if args.physx_runtime is None or args.physx_license is None:
                parser.error("Linux wheels require --physx-runtime and --physx-license")
            runtime_project = temp / "physx-project"
            runtime_version = stage_physx_runtime(
                source / "physx_gpu_runtime",
                runtime_project,
                args.physx_runtime,
                args.physx_license,
            )
            run(
                args.uv,
                "build",
                "--python",
                python_executable,
                "--wheel",
                "--out-dir",
                str(wheelhouse),
                str(runtime_project),
            )
            (runtime_wheel,) = wheelhouse.glob("kangengine_physx_gpu-*.whl")
            with zipfile.ZipFile(runtime_wheel) as archive:
                required = {
                    "kangengine_physx_gpu/lib/libPhysXGpu_64.so",
                }
                if not required.issubset(archive.namelist()):
                    raise AssertionError(
                        "PhysX runtime wheel is missing its library"
                    )
                inspect_license_bundle(archive, "kangengine_physx_gpu")
                if "-py3-none-linux_" not in runtime_wheel.name:
                    raise AssertionError(
                        f"Unexpected runtime wheel tag: {runtime_wheel.name}"
                    )
        stage_project(
            source,
            staged,
            args.usd_runtime,
            args.extension,
            args.physx_runtime,
        )

        run(
            args.uv,
            "build",
            "--python",
            python_executable,
            "--wheel",
            "--out-dir",
            str(wheelhouse),
            str(staged),
        )
        wheels = list(wheelhouse.glob("kangengine-*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"expected one wheel, found {len(wheels)}")
        wheel = wheels[0]
        inspect_wheel(wheel, args.expect_usd, cuda_runtime_version, runtime_version)

        # Prepare final platform artifacts before recording what validation must use.
        artifacts = [wheel, *([runtime_wheel] if runtime_wheel else [])]
        prepared = temp / "release"
        if sys.platform == "linux":
            run(
                python_executable, str(Path(__file__).with_name("prepare_linux_release.py")),
                *map(str, artifacts), "--output-dir", str(prepared),
            )
        else:
            run(python_executable, "-m", "twine", "check", "--strict", *map(str, artifacts))
            prepared.mkdir()
            for artifact in artifacts:
                shutil.copy2(artifact, prepared / artifact.name)

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        final_wheels = sorted(prepared.glob("*.whl"))
        for artifact in final_wheels:
            if artifact.name.startswith("kangengine-"):
                inspect_wheel(
                    artifact, args.expect_usd, cuda_runtime_version, runtime_version
                )
            shutil.copy2(artifact, output_dir / artifact.name)
            print(f"Packed wheel: {output_dir / artifact.name}")
        manifest = {"platform": sys.platform, "wheels": [p.name for p in final_wheels]}
        (output_dir / "wheel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
