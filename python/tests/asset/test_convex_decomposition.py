"""Saved decomposition results remain usable independently of their backend."""

from pathlib import Path
import sys

import numpy as np
import pytest

import kangengine as ke
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import convex_decomposition as decomposition


@pytest.fixture
def mesh():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


@pytest.fixture
def backend(monkeypatch):
    class Backend:
        name = "test"
        version = "1"

        def __init__(self):
            self.calls = []

        def decompose(self, vertices, faces, options):
            self.calls.append(options.copy())
            return [
                decomposition.ConvexPartData(vertices.copy(), faces.copy()),
                decomposition.ConvexPartData(
                    vertices + [options.get("offset", 2), 0, 0], faces.copy()
                ),
            ]

    monkeypatch.setattr(decomposition, "_BACKENDS", {})
    implementation = Backend()
    decomposition.register_convex_decomposition_backend(implementation)
    return implementation


def test_saved_parts_load_without_backend_and_keep_original_metadata(
    tmp_path, mesh, backend, monkeypatch
):
    path = tmp_path / "result.npz"
    parts, metadata = decomposition.decompose_mesh(
        *mesh, backend="test", options={"offset": 3}, result_path=path
    )
    monkeypatch.setattr(decomposition, "_BACKENDS", {})
    loaded, saved = decomposition.decompose_mesh(
        mesh[0] * 10, mesh[1], backend="unavailable",
        options={"offset": 99}, result_path=path,
    )
    assert len(backend.calls) == 1
    assert saved == {**metadata, "cache_hit": True}
    assert saved["options"] == {"offset": 3}
    assert not metadata["cache_hit"]
    for original, restored in zip(parts, loaded, strict=True):
        np.testing.assert_array_equal(restored.vertices, original.vertices)
        np.testing.assert_array_equal(restored.faces, original.faces)

    # The serialized arrays still feed the existing native compound path.
    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.y_up())
    actor = world.create_dynamic_convex_compound(
        [ke.physics.ConvexMeshPart(part.vertices, part.faces) for part in loaded],
        pos=(0, 2, 0),
    )
    assert actor.num_shapes() == len(parts)
    assert actor.get_mass() > 0


def test_rebuild_replaces_result_with_selected_backend(tmp_path, mesh, backend):
    path = tmp_path / "result.npz"
    decomposition.decompose_mesh(*mesh, backend="test", result_path=path)
    backend.name = "second"
    backend.version = "2"
    decomposition.register_convex_decomposition_backend(backend)
    parts, metadata = decomposition.decompose_mesh(
        *mesh, backend="second", options={"offset": 5},
        result_path=path, rebuild=True,
    )
    assert backend.calls == [{}, {"offset": 5}]
    assert metadata["backend"] == "second"
    assert metadata["backend_version"] == "2"
    loaded, saved = decomposition.decompose_mesh(*mesh, result_path=path)
    np.testing.assert_array_equal(loaded[1].vertices, parts[1].vertices)
    assert saved == {**metadata, "cache_hit": True}


def test_failed_write_preserves_previous_result(tmp_path, mesh, backend, monkeypatch):
    path = tmp_path / "result.npz"
    decomposition.decompose_mesh(*mesh, backend="test", result_path=path)
    previous = path.read_bytes()

    def interrupted_write(stream, **arrays):
        stream.write(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(decomposition.np, "savez_compressed", interrupted_write)
    with pytest.raises(OSError, match="interrupted write"):
        decomposition.decompose_mesh(
            *mesh, backend="test", result_path=path, rebuild=True
        )
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_existing_file_does_not_recompute(tmp_path, mesh, backend):
    path = tmp_path / "result.npz"
    path.write_bytes(b"partial")
    with pytest.raises(ValueError):
        decomposition.decompose_mesh(*mesh, backend="test", result_path=path)
    assert backend.calls == []


def test_empty_decomposition_is_not_saved(tmp_path, mesh, backend, monkeypatch):
    path = tmp_path / "result.npz"
    monkeypatch.setattr(backend, "decompose", lambda *args: [])
    with pytest.raises(RuntimeError, match="no convex parts"):
        decomposition.decompose_mesh(*mesh, backend="test", result_path=path)
    assert not path.exists()


def test_without_result_path_runs_each_time(mesh, backend):
    for _ in range(2):
        _, metadata = decomposition.decompose_mesh(*mesh, backend="test")
        assert not metadata["cache_hit"]
    assert len(backend.calls) == 2
