"""Analytic mesh intersections for directional terrain queries."""

import numpy as np
import pytest
import torch

from kangengine import physics, terrain


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    pytest.importorskip("warp")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


@pytest.fixture
def plane():
    vertices = np.array(
        [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]],
        dtype=np.float32,
    )
    return terrain.TerrainMesh(vertices, [[0, 1, 2], [0, 2, 3]])


def test_diagonal_plane_hits_broadcast_and_measure_metres(plane, device):
    starts = torch.tensor([[[0.0, 0.0, 2.0]], [[1.0, -1.0, 4.0]]], device=device)
    directions = torch.tensor(
        [[[3.0, 0.0, -4.0], [0.0, 0.0, -7.0], [0.0, 3.0, -4.0]]],
        device=device,
    )

    hits, distances = plane.raycast(starts, directions)

    expected_hits = torch.tensor(
        [
            [[1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.5, 0.0]],
            [[4.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 2.0, 0.0]],
        ],
        device=device,
    )
    expected_distances = torch.tensor([[2.5, 2.0, 2.5], [5.0, 4.0, 5.0]], device=device)
    torch.testing.assert_close(hits, expected_hits)
    torch.testing.assert_close(distances, expected_distances)


def test_distance_limit_and_misses(plane, device):
    start = torch.tensor([0.0, 0.0, 2.0], device=device)
    directions = torch.tensor(
        [[3.0, 0.0, -4.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [10.0, 0.0, -1.0]],
        device=device,
    )

    hits, distances = plane.raycast(start, directions)
    torch.testing.assert_close(hits[0], torch.tensor([1.5, 0.0, 0.0], device=device))
    torch.testing.assert_close(distances[0], torch.tensor(2.5, device=device))
    assert hits[1:].isposinf().all()
    assert distances[1:].isposinf().all()

    capped_hits, capped_distances = plane.raycast(start, directions, max_distance=2.4)
    assert capped_hits.isposinf().all()
    assert capped_distances.isposinf().all()


def test_diagonal_ray_hits_riser_before_floor(plane, device):
    wall = np.array([[1, -2, 0], [1, -2, 2], [1, 2, 2], [1, 2, 0]], np.float32)
    mesh = terrain.TerrainMesh(
        np.concatenate([plane.vertices, wall]),
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
    )
    start = torch.tensor([0.0, 0.0, 1.5], device=device)
    direction = torch.tensor([2.0, 0.0, -1.0], device=device)

    hit, distance = mesh.raycast(start, direction)

    torch.testing.assert_close(hit, torch.tensor([1.0, 0.0, 1.0], device=device))
    torch.testing.assert_close(distance, torch.tensor(1.25**0.5, device=device))
    torch.testing.assert_close(
        mesh.sample_height(start), torch.tensor(0.0, device=device)
    )


def test_instance_raycast_returns_world_coordinates(plane, device):
    world = physics.PhysicsWorld(physics.PhysicsConfig.z_up())
    instance = plane.create(world, position=(7.0, -3.0, 2.0))
    start = torch.tensor([7.0, -3.0, 6.0], device=device)
    direction = torch.tensor([3.0, 0.0, -4.0], device=device)

    try:
        hit, distance = instance.raycast(start, direction)
        torch.testing.assert_close(hit, torch.tensor([10.0, -3.0, 2.0], device=device))
        torch.testing.assert_close(distance, torch.tensor(5.0, device=device))

        hit, distance = instance.raycast(start, direction, max_distance=4.9)
        assert hit.isposinf().all()
        assert distance.isposinf()
    finally:
        instance.remove()


def test_empty_batch_preserves_shape_and_device(plane, device):
    starts = torch.empty((2, 0, 3), device=device)
    direction = torch.tensor([0.0, 0.0, -1.0], device=device)

    hits, distances = plane.raycast(starts, direction)

    assert hits.shape == (2, 0, 3)
    assert distances.shape == (2, 0)
    assert hits.device == distances.device == starts.device
    assert hits.dtype == distances.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_queries_follow_current_stream_and_share_mesh(plane):
    pytest.importorskip("warp")
    build_stream = torch.cuda.Stream()
    query_stream = torch.cuda.Stream()

    # Build the cached mesh on one stream, then use it from another.
    with torch.cuda.stream(build_stream):
        plane.raycast(
            torch.tensor([0.0, 0.0, 1.0], device="cuda"),
            torch.tensor([0.0, 0.0, -1.0], device="cuda"),
        )

    try:
        with torch.cuda.stream(query_stream):
            heights = torch.linspace(1.0, 3.0, 256, device="cuda")
            starts = torch.zeros((256, 3), device="cuda")
            starts[:, 2] = heights
            direction = torch.tensor([3.0, 0.0, -4.0], device="cuda")

            hits, distances = plane.raycast(starts, direction)
            expected_hits = torch.zeros_like(starts)
            expected_hits[:, 0] = 0.75 * heights
            torch.testing.assert_close(hits, expected_hits)
            torch.testing.assert_close(distances, 1.25 * heights)
    finally:
        build_stream.synchronize()
        query_stream.synchronize()
