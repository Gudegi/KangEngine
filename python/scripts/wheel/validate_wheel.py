"""Validate packaged release wheels in a fresh virtual environment."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

from wheel_licenses import inspect_license_bundle


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


def inspect_wheel(
    wheel: Path,
    expect_usd: bool,
    cuda_runtime_version: str | None = None,
    physx_runtime_version: str | None = None,
) -> None:
    if wheel.name.endswith("-none-any.whl"):
        raise AssertionError(f"native wheel has a pure-Python tag: {wheel.name}")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        licensed_projects = inspect_license_bundle(archive, "kangengine")
        split_artifacts = [
            name for name in names
            if Path(name).name.startswith("libengine_")
            or name.startswith(("kangengine_viewer/", "kangengine_sim/", "kangengine_usd/"))
        ]
        if split_artifacts:
            raise AssertionError(f"Split-package artifacts in wheel: {split_artifacts}")
        if cuda_runtime_version:
            if "PhysX" not in licensed_projects:
                raise AssertionError("CUDA wheel is missing its PhysX license")
            metadata_name = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            metadata = archive.read(metadata_name).decode()
            if f"nvidia-cuda-runtime=={cuda_runtime_version}" not in metadata:
                raise AssertionError("CUDA runtime dependency is missing from METADATA")
            if f"kangengine-physx-gpu=={physx_runtime_version}" not in metadata:
                raise AssertionError(
                    "PhysX runtime dependency does not match the built runtime"
                )
            if any("libPhysXGpu" in name for name in names):
                raise AssertionError("PhysX GPU must come from the runtime dependency")
            if any("libcudart.so" in name for name in names):
                raise AssertionError(
                    "CUDA runtime must come from the NVIDIA dependency"
                )
        if "kangengine/_kangengine.so" not in names:
            raise AssertionError("wheel does not contain kangengine/_kangengine.so")
        bundled_usd = any(
            name.startswith("kangengine/_native/lib/libusd_") for name in names
        )
        if expect_usd and not bundled_usd:
            raise AssertionError("USD wheel does not contain bundled libraries")
        if bundled_usd and "OpenUSD" not in licensed_projects:
            raise AssertionError("USD wheel is missing its OpenUSD license")
        external = sorted(
            name for name in names if name.startswith("kangengine/assets/external/")
        )
        if external:
            raise AssertionError(f"wheel contains external assets: {external[0]}")
        stale = sorted(names & STALE_MODULES)
        if stale:
            raise AssertionError(f"wheel contains stale modules: {', '.join(stale)}")


def create_validation_environment(
    uv: str,
    python_executable: str,
    directory: Path,
    wheel: Path,
    runtime_wheel: Path | None,
) -> tuple[str, dict[str, str]]:
    environment = os.environ.copy()
    for name in (
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
        "KANGENGINE_ASSETS_ROOT",
    ):
        environment.pop(name, None)

    # Install only the wheel's declared dependencies into a fresh environment.
    run(
        uv, "--no-config", "venv", "--python", python_executable, str(directory),
        cwd=directory.parent, env=environment,
    )
    validation_python = str(directory / "bin" / "python")
    run(
        uv, "--no-config", "pip", "install", "--python", validation_python,
        str(wheel), *([str(runtime_wheel)] if runtime_wheel else []),
        cwd=directory.parent, env=environment,
    )
    run(
        uv, "--no-config", "pip", "check", "--python", validation_python,
        cwd=directory.parent, env=environment,
    )
    run(
        validation_python, "-I", "-c",
        (
            "import sys; from pathlib import Path; import kangengine, numpy, torch; "
            "root = Path(sys.prefix); "
            "assert sys.prefix != sys.base_prefix; "
            "assert all(Path(m.__file__).resolve().is_relative_to(root) "
            "for m in (kangengine, numpy, torch)); "
            "print('PASS: engine and dependencies imported from fresh venv', root)"
        ),
        cwd=directory.parent, env=environment,
    )
    return validation_python, environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wheel-dir", type=Path, default=Path("python/dist/release"))
    parser.add_argument("--expect-usd", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true",
                        help="Run Linux CUDA/PhysX checks without requiring a display")
    parser.add_argument("--render-smoke", action="store_true",
                        help="Run rendering checks; requires a working OpenGL display. "
                             "Combine with --gpu-smoke to check CUDA/OpenGL interop")
    args = parser.parse_args()
    if args.gpu_smoke and sys.platform != "linux":
        parser.error("--gpu-smoke requires Linux")

    # Select exactly the artifacts from the last successful packaging run.
    manifest = json.loads((args.wheel_dir / "wheel-manifest.json").read_text())
    if manifest["platform"] != sys.platform:
        raise ValueError("Packaged wheels belong to a different platform")
    wheels = [(args.wheel_dir / name).resolve() for name in manifest["wheels"]]
    (wheel,) = [path for path in wheels if path.name.startswith("kangengine-")]
    runtime_wheel = None
    runtime_version = None
    cuda_runtime_version = None
    if sys.platform == "linux":
        (runtime_wheel,) = [
            path for path in wheels if path.name.startswith("kangengine_physx_gpu-")
        ]
        runtime_version = runtime_wheel.name.split("-")[1]
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
        dependency = next(
            value for value in metadata.get_all("Requires-Dist", [])
            if value.startswith("nvidia-cuda-runtime==")
        )
        cuda_runtime_version = re.match(r"nvidia-cuda-runtime==([^; ]+)", dependency)[1]
    inspect_wheel(wheel, args.expect_usd, cuda_runtime_version, runtime_version)

    python_executable = os.path.abspath(args.python)
    smoke = Path(__file__).resolve().parents[2] / "examples" / "smoke"
    with tempfile.TemporaryDirectory(prefix="kangengine-wheel-test-") as temp_name:
        temp = Path(temp_name)
        python_executable, environment = create_validation_environment(
            args.uv, python_executable, temp / "venv", wheel, runtime_wheel,
        )
        if args.expect_usd:
            run(
                python_executable,
                "-I",
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
            "-I",
            str(smoke / "public_api_surface_smoke.py"),
            cwd=temp,
            env=environment,
        )
        run(
            python_executable,
            "-I",
            str(smoke / "public_stub_surface_smoke.py"),
            cwd=temp,
            env=environment,
        )

        # Keep CPU and GPU initialization in separate processes.
        run(
            python_executable,
            "-I",
            str(smoke / "physics_cpu_system_smoke.py"),
            cwd=temp,
            env=environment,
        )

        if args.gpu_smoke:
            run(
                python_executable,
                "-I",
                "-c",
                (
                    "import runpy, sysconfig; from pathlib import Path; "
                    f"runpy.run_path({str(smoke / 'physics_gpu_system_smoke.py')!r}, run_name='__main__'); "
                    "paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line}; "
                    "root = Path(sysconfig.get_path('purelib')); "
                    "required = ('libPhysXGpu_64.so', 'libcudart.so.13'); "
                    "selected = [p for p in paths if any(Path(p).name.startswith(n) for n in required)]; "
                    "assert all(any(Path(p).name.startswith(n) for p in selected) for n in required), selected; "
                    "assert all(Path(p).is_relative_to(root) for p in selected), selected; "
                    "assert all(Path(p).is_relative_to(root / 'kangengine_physx_gpu') "
                    "for p in selected if Path(p).name == 'libPhysXGpu_64.so'), selected; "
                    "print('PASS: PhysX GPU and CUDA runtime loaded from wheel installation', selected)"
                ),
                cwd=temp,
                env=environment,
            )

        for script in (
            "scene_data_smoke.py",
            "animation_sim_roundtrip_smoke.py",
            "scene_resource_manager_smoke.py",
            "render_component_lifecycle_smoke.py",
            "sim_visual_batch_smoke.py",
        ):
            run(
                python_executable, "-I", str(smoke / script), cwd=temp,
                env={k: v for k, v in environment.items()
                     if k not in ("DISPLAY", "WAYLAND_DISPLAY")},
            )
        if args.render_smoke:
            for script in (
                "scene_renderer_boundary_smoke.py",
                "scene_render_system_smoke.py",
            ):
                run(python_executable, "-I", str(smoke / script), cwd=temp, env=environment)
        else:
            print("SKIP: rendering checks (enable with --render-smoke)")
        if args.gpu_smoke:
            run(
                python_executable, "-I", str(smoke / "mimickit_engine_gpu_reset_smoke.py"),
                cwd=temp, env=environment,
            )
            if args.render_smoke:
                run(
                    python_executable, "-I", str(smoke / "cuda_gl_visual_smoke.py"),
                    cwd=temp, env=environment,
                )
        print(f"PASS: isolated wheel {wheel.name}")


if __name__ == "__main__":
    main()
