"""Geometry and collision contracts for terrain, independent of G1 assets."""

import numpy as np
import pytest
import torch

from kangengine import UpAxis, physics, terrain


@pytest.mark.parametrize("axis,component", [(UpAxis.Y, 1), (UpAxis.Z, 2)])
@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_heightfield_winding_faces_up(axis, component, backend):
    mesh = terrain.height_field_to_mesh(np.zeros((3, 5)), up_axis=axis, backend=backend)
    points = np.array([[p.x, p.y, p.z] for p in mesh.vertices])
    faces = points[np.array(mesh.indices).reshape(-1, 3)]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    assert (normals[:, component] > 0).all()


def test_heightfield_rejects_nonfinite_and_handles_tiny_heights():
    world = physics.PhysicsWorld(physics.PhysicsConfig.z_up())
    for value in [np.nan, np.inf]:
        with pytest.raises(ValueError):
            world.add_heightfield(np.array([0, value, 0, 0], dtype=np.float32), 2, 2)
    with pytest.raises(ValueError):
        world.add_heightfield(
            np.zeros(4, dtype=np.float32), 2, 2, horizontal_scale=np.nan
        )
    assert world.add_heightfield(np.full(4, 2e-6, dtype=np.float32), 2, 2)


def test_triangle_interpolation_is_not_bilinear():
    field = terrain.SubTerrain(2, 2, horizontal_scale=1, vertical_scale=1)
    field.height_field_raw[:] = [[0, 0], [0, 1]]
    mesh = terrain.TerrainMesh.from_heightfield(field, center=False)
    positions = torch.tensor([[0.25, 0.25, 0], [0.75, 0.75, 0], [2, 2, 0]])
    heights = mesh.sample_height(positions)
    torch.testing.assert_close(heights[:2], torch.tensor([0.0, 0.5]))
    assert heights[2].isneginf()


def test_stair_raycast_preserves_vertical_risers():
    mesh = terrain.pyramid_stairs_mesh(step_height=0.1)
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [2.9, 0.0, 0.0], [2.6, 0.0, 0.0]]
    )
    heights = mesh.sample_height(points)
    torch.testing.assert_close(
        heights, torch.tensor([0.7, 0.0, 0.1, 0.2]), atol=1e-5, rtol=0
    )
    inverse = terrain.pyramid_stairs_mesh(step_height=0.1, inverted=True)
    torch.testing.assert_close(
        inverse.sample_height(points), -heights, atol=1e-5, rtol=0
    )


def test_chunked_instances_share_cooking_and_remove_ground_registration():
    field = terrain.SubTerrain(5, 7, horizontal_scale=1)
    source = terrain.TerrainMesh.from_heightfield(field)
    mesh = terrain.TerrainMesh(
        source.vertices, source.indices, max_triangles_per_chunk=8
    )
    world = physics.PhysicsWorld(physics.PhysicsConfig.z_up())
    first = mesh.create(world)
    count = world.num_cached_triangle_meshes()
    second = mesh.create(world, position=(10, 0, 2))
    assert count > 1 and world.num_cached_triangle_meshes() == count
    assert world.num_ground_actors() == 2 * count
    torch.testing.assert_close(
        second.sample_height(torch.tensor([[10.0, 0.0, 0.0]])), torch.tensor([2.0])
    )
    first.remove()
    first.remove()
    assert world.num_ground_actors() == count
    second.remove()
    assert world.num_ground_actors() == 0


@pytest.mark.parametrize(
    "cooking", [{}, {"num_prims_per_leaf": 15}, {"weld_tolerance": 1e-5}]
)
def test_sphere_settles_on_triangle_collision(cooking):
    world = physics.PhysicsWorld(physics.PhysicsConfig.z_up())
    field = terrain.SubTerrain(3, 5, horizontal_scale=1)
    mesh = terrain.TerrainMesh.from_heightfield(field)
    instance = mesh.create(world, **cooking)
    sphere = world.create_dynamic_sphere(0.15, (0, 0, 1))
    for _ in range(150):
        world.step()
    assert abs(sphere.get_root_position()[2] - 0.15) < 0.02
    instance.remove()


def test_cooking_options_partition_cache_and_validate_before_registration():
    world = physics.PhysicsWorld(physics.PhysicsConfig.z_up())
    mesh = terrain.TerrainMesh.from_heightfield(
        terrain.SubTerrain(3, 3, horizontal_scale=1)
    )
    instances = [
        mesh.create(world),
        mesh.create(world, num_prims_per_leaf=15),
        mesh.create(world, weld_tolerance=1e-5),
        mesh.create(world, num_prims_per_leaf=15),
    ]
    assert world.num_cached_triangle_meshes() == 3
    assert world.num_ground_actors() == 4
    for options in [
        {"num_prims_per_leaf": 1},
        {"num_prims_per_leaf": 16},
        {"weld_tolerance": -1},
        {"weld_tolerance": float("nan")},
        {"weld_tolerance": float("inf")},
    ]:
        with pytest.raises(ValueError):
            mesh.create(world, **options)
        assert world.num_cached_triangle_meshes() == 3
        assert world.num_ground_actors() == 4
    for instance in instances:
        instance.remove()
