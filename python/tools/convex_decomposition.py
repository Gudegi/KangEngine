"""Offline decomposition and saved results, independent of the engine runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class ConvexPartData:
    """Triangles in the input mesh's local frame and units, with no extra pose.

    vertices has shape (N, 3); faces has shape (M, 3) with integer indices.
    """

    vertices: np.ndarray
    faces: np.ndarray


@runtime_checkable
class ConvexDecompositionBackend(Protocol):
    """Adapter returning parts in the unchanged input coordinate frame."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def decompose(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        options: dict[str, Any],
    ) -> list[ConvexPartData]: ...


class CoACDBackend:
    """Load the optional CoACD dependency only when decomposition is requested."""

    name = "coacd"

    @property
    def version(self) -> str:
        return version("coacd")

    def decompose(self, vertices, faces, options):
        import coacd

        return [
            ConvexPartData(points, triangles)
            for points, triangles in coacd.run_coacd(
                coacd.Mesh(vertices, faces), **options
            )
        ]


_BACKENDS: dict[str, ConvexDecompositionBackend] = {"coacd": CoACDBackend()}


def register_convex_decomposition_backend(
    backend: ConvexDecompositionBackend,
) -> None:
    if not isinstance(backend, ConvexDecompositionBackend):
        raise TypeError("backend does not implement ConvexDecompositionBackend")
    name = backend.name.strip().lower()
    if not name:
        raise ValueError("backend name must not be empty")
    _BACKENDS[name] = backend


def get_convex_decomposition_backend(name: str) -> ConvexDecompositionBackend:
    key = name.strip().lower()
    try:
        return _BACKENDS[key]
    except KeyError as error:
        available = ", ".join(sorted(_BACKENDS)) or "none"
        raise LookupError(
            f"convex decomposition backend {name!r} is not registered "
            f"(available: {available})"
        ) from error


def available_convex_decomposition_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def decompose_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    backend: str = "coacd",
    options: dict[str, Any] | None = None,
    result_path: str | Path | None = None,
    rebuild: bool = False,
) -> tuple[list[ConvexPartData], dict[str, Any]]:
    """Load an existing result, or decompose and optionally save its parts.

    Args:
        vertices: Input positions shaped (N, 3), including any mesh scale.
        faces: Integer triangle indices shaped (M, 3).
        backend: Registered algorithm name, used only on a miss or rebuild.
        options: Algorithm-specific keyword arguments; omitted settings use
            that backend version's defaults.
        result_path: NPZ file to reuse by existence alone. Input and setting
            changes require a new path or rebuild=True. None disables storage.
        rebuild: Recompute even if the result file exists.

    Returns:
        Parts in the input frame and a metadata dict containing backend,
        backend_version, options, decomposition_seconds, and cache_hit.
        On a hit, metadata describes the saved run, not the current request.
    """
    destination = Path(result_path).expanduser().resolve() if result_path else None
    if destination is not None and destination.is_file() and not rebuild:
        with np.load(destination, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            parts = [
                ConvexPartData(saved[f"vertices_{i}"], saved[f"faces_{i}"])
                for i in range(metadata["part_count"])
            ]
        return parts, {**metadata, "cache_hit": True}

    implementation = get_convex_decomposition_backend(backend)
    options = dict(options or {})
    metadata = {
        "backend": implementation.name,
        "backend_version": implementation.version,
        "options": options.copy(),
    }
    start = perf_counter()
    parts = implementation.decompose(vertices, faces, options)
    metadata["decomposition_seconds"] = perf_counter() - start
    if not parts:
        raise RuntimeError(f"{implementation.name} returned no convex parts")
    metadata["part_count"] = len(parts)

    if destination is not None:
        arrays = {"metadata": np.array(json.dumps(metadata))}
        for i, part in enumerate(parts):
            arrays[f"vertices_{i}"] = part.vertices
            arrays[f"faces_{i}"] = part.faces

        # Publish only complete results; a failed rebuild keeps the previous file.
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=destination.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    return parts, {**metadata, "cache_hit": False}
