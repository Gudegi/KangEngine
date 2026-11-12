"""Bundle third-party notices and check native wheel coverage."""

import csv
from fnmatch import fnmatch
from io import BytesIO, StringIO
from pathlib import Path
import zipfile


INVENTORY = "THIRD-PARTY-LICENSES.txt"
BUNDLE = "THIRD-PARTY-LICENSES.zip"

# Include dependencies embedded in these binaries, not just separate .so files.
LIBRARY_PROJECTS = {
    "_kangengine.so": {
        "glm", "fmt", "eigen3", "tinyobjloader", "tinyxml2", "openfbx",
        "libdeflate", "mikktspace", "stb", "pybind11", "glad", "glfw3",
        "imgui", "imguizmo", "implot", "LBFGSpp", "IconFontCppHeaders",
        "ImGuiFileDialog", "ImViewGuizmo",
    },
    "libusd_*": {"OpenUSD", "tbb", "zlib"},
    "libtbb*": {"tbb"},
    "libz.so*": {"zlib"},
    "libz-*.so*": {"zlib"},
    "libz.*.dylib": {"zlib"},
    "libdeflate*": {"libdeflate"},
    "libPhysX*": {"PhysX"},
    "libGLX*": {"libglvnd"},
    "libGLdispatch*": {"libglvnd"},
    "libXt-*": {"libXt"},
    "libXt.so*": {"libXt"},
    "libMaterialX*": {"MaterialX"},
    "libosdCPU*": {"OpenSubdiv"},
    "libosdGPU*": {"OpenSubdiv"},
}


def write_license_bundle(
    destination: Path, notices: dict[str, Path], versions: dict[str, str]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    inventory = StringIO()
    writer = csv.writer(inventory, delimiter="\t", lineterminator="\n")
    writer.writerow(("Project", "Version", "Notice"))
    with zipfile.ZipFile(destination / BUNDLE, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(notices.items()):
            content = source.read_bytes()
            if not content.strip():
                raise ValueError(f"Empty third-party notice: {source}")
            project = name.split("/", 1)[0]
            writer.writerow((project, versions[project], name))
            archive.writestr(name, content)
    (destination / INVENTORY).write_text(inventory.getvalue(), encoding="utf-8")


def inspect_license_bundle(archive: zipfile.ZipFile, package: str) -> set[str]:
    prefix = f"{package}/licenses/"
    names = set(archive.namelist())
    if not {prefix + INVENTORY, prefix + BUNDLE}.issubset(names):
        raise AssertionError(f"{package} is missing its third-party license bundle")
    reader = csv.DictReader(
        StringIO(archive.read(prefix + INVENTORY).decode("utf-8")), delimiter="\t"
    )
    if reader.fieldnames != ["Project", "Version", "Notice"]:
        raise AssertionError("Invalid third-party license inventory header")
    entries = list(reader)
    if not entries:
        raise AssertionError("Empty third-party license inventory")
    with zipfile.ZipFile(BytesIO(archive.read(prefix + BUNDLE))) as notices:
        listed = [row["Notice"] for row in entries]
        if len(set(listed)) != len(listed) or set(listed) != set(notices.namelist()):
            raise AssertionError("Third-party license inventory and ZIP differ")
        for row in entries:
            name = row["Notice"]
            if (not row["Project"] or not row["Version"]
                    or not name.startswith(row["Project"] + "/")
                    or not notices.read(name).strip()):
                raise AssertionError(f"Invalid or empty third-party notice: {name}")
        projects = {row["Project"] for row in entries}
        for project in projects:
            if f"{project}/LICENSE" not in listed:
                raise AssertionError(f"Missing LICENSE for {project}")

    for path in names:
        name = Path(path).name
        if not (".so" in name or name.endswith(".dylib")):
            continue
        # USD plugin payloads are covered by its complete upstream notice.
        if "/_native/lib/usd/" in path:
            required = {"OpenUSD", "tbb", "zlib"}
        else:
            required = next(
                (owners for pattern, owners in LIBRARY_PROJECTS.items()
                 if fnmatch(name, pattern)), None
            )
        if required is None:
            raise AssertionError(f"No license mapping for bundled library: {path}")
        missing = required - projects
        if missing:
            raise AssertionError(f"Missing licenses for {path}: {sorted(missing)}")
    return projects
