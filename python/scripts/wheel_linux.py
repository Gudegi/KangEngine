"""Relocate trusted, locally built ELF files for Linux wheel staging."""

from pathlib import Path
import filecmp
import os
import re
import shutil
import subprocess


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def needed(path: Path) -> list[str]:
    return output("patchelf", "--print-needed", str(path)).splitlines()


def is_elf(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def bundle_linux_runtime(
    package: Path,
    physx_runtime: Path | None,
    physx_license: Path | None,
) -> None:
    if not shutil.which("patchelf"):
        raise RuntimeError("Linux wheel staging requires patchelf on PATH")
    extension = package / "_kangengine.so"
    extension_dependencies = needed(extension)
    if any(name.startswith("libusd_") for name in extension_dependencies):
        if not (package / "_native/lib/usd/plugInfo.json").is_file():
            raise ValueError("USD-enabled Linux wheels require --usd-runtime")
    cuda = [name for name in extension_dependencies if name.startswith("libcudart.so")]
    if cuda != ["libcudart.so.13"]:
        raise RuntimeError("Linux distribution wheels require a CUDA 13 extension")
    if physx_runtime is None:
        raise ValueError("Linux wheels require --physx-runtime")

    lib = package / "_native" / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    if physx_runtime is not None:
        if physx_license is None or not physx_license.is_file():
            raise ValueError("--physx-license must name the SDK license file")
        shutil.copy2(physx_runtime / "libPhysXGpu_64.so", lib)
        licenses = package / "licenses"
        licenses.mkdir(exist_ok=True)
        shutil.copy2(physx_license, licenses / "PhysX.txt")

    # ldd covers transitive dependencies, but not PhysX's dlopen plugin above.
    # Collect before rewriting any search paths. Keep host OS/driver libraries
    # external; auditwheel determines the eventual manylinux policy separately.
    binaries = [extension, *lib.rglob("*.so*")]
    dependencies: dict[str, Path] = {}
    for binary in binaries:
        if not is_elf(binary):
            continue
        for line in output("ldd", str(binary)).splitlines():
            if "=> not found" in line:
                raise RuntimeError(f"Unresolved dependency for {binary}: {line}")
            match = re.match(r"\s*(\S+) => (/\S+)", line)
            if not match:
                continue
            name, resolved = match.groups()
            path = Path(resolved)
            if name.startswith("libcudart.so"):
                if name != "libcudart.so.13":
                    raise RuntimeError(f"Unexpected CUDA dependency: {name}")
                continue
            if name.startswith(("libcuda.so", "libnvidia-")):
                continue
            if any(
                path.resolve().is_relative_to(Path(root).resolve())
                for root in ("/usr/lib", "/usr/lib64", "/lib", "/lib64")
            ):
                continue
            previous = dependencies.setdefault(name, path)
            if previous.resolve() != path.resolve() and not filecmp.cmp(
                previous, path, shallow=False
            ):
                raise RuntimeError(f"Conflicting libraries for {name}")
    for name, path in dependencies.items():
        destination = lib / name
        if destination.exists():
            if not filecmp.cmp(destination, path, shallow=False):
                raise RuntimeError(f"Bundled library conflicts with dependency: {name}")
        else:
            shutil.copy2(path, destination)

    for binary in lib.rglob("*.so*"):
        if is_elf(binary):
            relative_lib = Path(os.path.relpath(lib, binary.parent))
            subprocess.run(
                ["patchelf", "--set-rpath", f"$ORIGIN/{relative_lib}", str(binary)],
                check=True,
            )
    paths = "$ORIGIN/_native/lib:$ORIGIN/../nvidia/cu13/lib"
    subprocess.run(["patchelf", "--set-rpath", paths, str(extension)], check=True)
