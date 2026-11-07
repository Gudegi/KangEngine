"""Offline convex-decomposition cache contracts for MJCF collision meshes.

This planner records backend settings without running decomposition.
Derived MJCF export is not implemented; the intended output
is an MJCF plus convex part OBJ files for the normal mesh-to-PhysX-hull path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kangengine import asset


MANIFEST_SCHEMA = "kangengine.mjcf-convex-cache"
MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class CollisionMeshUse:
    body_index: int
    body_name: str
    geom_index: int
    geom_name: str
    local_position: tuple[float, float, float]
    local_rotation_xyzw: tuple[float, float, float, float]


@dataclass(slots=True)
class CollisionMeshAsset:
    source_file: str
    source_sha256: str
    cache_key: str
    part_directory: str
    uses: list[CollisionMeshUse] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MjcfConvexCacheManifest:
    source_mjcf: str
    source_sha256: str
    output_mjcf: str
    backend: str
    backend_options: dict[str, Any]
    meshes: list[CollisionMeshAsset]
    schema: str = MANIFEST_SCHEMA
    version: int = MANIFEST_VERSION
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def plan_mjcf_convex_cache(
    mjcf_path: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "coacd",
    backend_options: dict[str, Any] | None = None,
) -> MjcfConvexCacheManifest:
    """Inspect authored MJCF mesh collisions and create a non-converting plan."""

    source_mjcf = Path(mjcf_path).expanduser().resolve()
    cache_root = Path(output_dir).expanduser().resolve()
    data = asset.MJCFLoader.load(str(source_mjcf))
    body_names = list(data.skeleton_tree.node_names())

    meshes: dict[Path, CollisionMeshAsset] = {}
    for body_index, geoms in sorted(data.collision_geoms.items()):
        body_name = (
            body_names[body_index]
            if 0 <= body_index < len(body_names)
            else f"body_{body_index}"
        )
        for geom_index, geom in enumerate(geoms):
            if _enum_name(geom.type) != "CONVEX_MESH":
                continue
            source_file = (Path(data.asset_dir) / geom.mesh_file).resolve()
            mesh = meshes.get(source_file)
            if mesh is None:
                source_hash = _sha256(source_file)
                cache_key = f"{source_file.stem}-{source_hash[:12]}"
                mesh = CollisionMeshAsset(
                    source_file=str(source_file),
                    source_sha256=source_hash,
                    cache_key=cache_key,
                    part_directory=str(cache_root / "meshes" / cache_key),
                )
                meshes[source_file] = mesh
            pos = geom.pos
            quat = geom.quat
            mesh.uses.append(
                CollisionMeshUse(
                    body_index=int(body_index),
                    body_name=body_name,
                    geom_index=geom_index,
                    geom_name=str(geom.name),
                    local_position=(float(pos.x), float(pos.y), float(pos.z)),
                    local_rotation_xyzw=(
                        float(quat.x),
                        float(quat.y),
                        float(quat.z),
                        float(quat.w),
                    ),
                )
            )

    return MjcfConvexCacheManifest(
        source_mjcf=str(source_mjcf),
        source_sha256=_sha256(source_mjcf),
        output_mjcf=str(cache_root / f"{source_mjcf.stem}.convex.xml"),
        backend=backend.strip().lower(),
        backend_options=dict(backend_options or {}),
        meshes=list(meshes.values()),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", str(value).split(".")[-1])).upper()


__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "CollisionMeshUse",
    "CollisionMeshAsset",
    "MjcfConvexCacheManifest",
    "plan_mjcf_convex_cache",
]
