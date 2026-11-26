import numpy as np
import pytest

from kangengine.geometry.signed_distance import SignedDistanceMesh


def box():
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


def test_signed_distance_inside_outside_and_gradient():
    vertices, faces = box()
    vertices += [3, 0, 0]
    sdf = SignedDistanceMesh(vertices, faces)
    points = np.array([[4.5, .1, .2], [3.8, .1, .2], [100, 0, 0]])
    distance, nearest, gradient = sdf.query(points, chunk_size=1)
    np.testing.assert_allclose(distance, [.5, -.2, 96])
    np.testing.assert_allclose(gradient, [[1, 0, 0]] * 3, atol=1e-10)
    np.testing.assert_allclose(nearest[:, 0], 4)
    for axis in range(3):
        offset = np.eye(3)[axis] * 1e-5
        numerical = (sdf.query(points + offset)[0] - sdf.query(points - offset)[0]) / 2e-5
        np.testing.assert_allclose(numerical, gradient[:, axis], atol=1e-7)
    assert sdf.query(np.empty((0, 3)))[0].shape == (0,)
    vertices += [1, 0, 0]
    assert sdf.content_hash != SignedDistanceMesh(vertices, faces).content_hash


def test_rejects_open_mesh():
    vertices, faces = box()
    with pytest.raises(ValueError, match="closed"):
        SignedDistanceMesh(vertices, faces[:-1])


def test_box_distance_random_points_near_surface_and_snapshot():
    vertices, faces = box()
    sdf = SignedDistanceMesh(vertices, faces)
    points = np.random.default_rng(7).uniform(-2, 2, (2000, 3))
    q = np.abs(points) - 1
    expected = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)
    np.testing.assert_allclose(sdf.query(points)[0], expected, atol=1e-10)
    near = np.array([[1 + 1e-8, .2, .3], [1 - 1e-8, .2, .3], [1, 0, 0]])
    np.testing.assert_allclose(sdf.query(near)[0], [1e-8, -1e-8, 0], atol=1e-12)
    vertices[:] = 100
    np.testing.assert_allclose(sdf.query([[0, 0, 0]])[0], [-1])
    with pytest.raises(ValueError, match="finite"):
        sdf.query([[np.nan, 0, 0]])


def test_rejects_inconsistent_faces():
    vertices, faces = box()
    faces[0] = faces[0, ::-1]
    with pytest.raises(ValueError, match="oriented"):
        SignedDistanceMesh(vertices, faces)
    with pytest.raises(ValueError, match="volume"):
        SignedDistanceMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2], [2, 1, 0]])


def test_single_batch_and_chunked_queries_match_for_strided_inputs():
    sdf = SignedDistanceMesh(*box())
    points = np.random.default_rng(14).uniform(-2, 2, (4, 3, 3)).transpose(1, 0, 2)
    assert not points.flags.c_contiguous
    direct = sdf.query(points)
    chunked = sdf.query(points, chunk_size=2)
    for actual, expected in zip(direct, chunked):
        np.testing.assert_allclose(actual, expected)
    assert direct[0].shape == (3, 4)
    assert direct[1].shape == direct[2].shape == (3, 4, 3)
    distance, closest, gradient = sdf.query([2, .2, .1])
    assert distance.shape == ()
    np.testing.assert_allclose(closest, [1, .2, .1])
    np.testing.assert_allclose(gradient, [1, 0, 0])
