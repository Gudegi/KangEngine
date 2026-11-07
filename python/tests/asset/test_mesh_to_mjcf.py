"""Mesh export preserves collision parts and cooked mass properties on reload."""

from pathlib import Path
import sys

import numpy as np
import pytest
import kangengine as ke

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import convex_decomposition as decomposition
from mesh_to_mjcf import mesh_to_mjcf


@pytest.mark.parametrize("mass", [None, 0.75])
def test_mesh_export_round_trip(tmp_path, monkeypatch, mass):
    source = tmp_path / "offset.obj"
    source.write_text(
        "v 1 2 3\nv 2 2 3\nv 1 3 3\nv 1 2 4\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )

    class Backend:
        name = "test"
        version = "1"

        def decompose(self, vertices, faces, options):
            return [
                decomposition.ConvexPartData(vertices, faces),
                decomposition.ConvexPartData(vertices + [2, 0, 0], faces),
            ]

    monkeypatch.setattr(decomposition, "_BACKENDS", {"coacd": Backend()})
    output = mesh_to_mjcf(source, tmp_path / "output", mass=mass)
    imported = ke.asset.MJCFLoader.parse(str(output))
    data = imported.articulation
    assert not imported.diagnostics.warnings
    assert len(data.visual_geoms) == 1
    assert len(data.collision_geoms[0]) == 2
    assert data.skeleton_tree.node_names() == ["offset"]

    mesh = ke.asset.load_obj(str(source))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    np.testing.assert_allclose(data.collision_geoms[0][0].mesh_data.vertices, vertices)
    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.y_up())
    original = world.create_dynamic_convex_compound(
        [ke.physics.ConvexMeshPart(vertices, faces),
         ke.physics.ConvexMeshPart(vertices + [2, 0, 0], faces)], pos=(0, 0, 0),
    )
    restored = world.create_dynamic_rigid(
        ke.asset.RigidDesc.from_articulation(data), pos=(0, 0, 0),
    )
    assert restored.num_shapes() == 2
    expected_mass = original.get_mass() if mass is None else mass
    np.testing.assert_allclose(restored.get_mass(), expected_mass, rtol=1e-5)
    np.testing.assert_allclose(
        restored.get_local_com_pose(), original.get_local_com_pose(), atol=1e-6,
    )
    np.testing.assert_allclose(
        restored.get_mass_space_inertia(),
        original.get_mass_space_inertia() * (expected_mass / original.get_mass()),
        rtol=1e-5,
    )


@pytest.mark.parametrize("mass", [0.0, -1.0, float("nan"), float("inf")])
def test_mesh_export_rejects_invalid_mass(tmp_path, mass):
    with pytest.raises(ValueError, match="mass must be finite and greater than zero"):
        mesh_to_mjcf(tmp_path / "unused.obj", tmp_path / "output", mass=mass)
