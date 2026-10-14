"""Validate NumPy quaternion conversion conventions and input boundaries."""

import numpy as np
import pytest

from kangengine import utils


@pytest.mark.parametrize("magnitude", [1.0, -3.0, 100.0])
def test_quaternion_matrix_rotates_column_vectors(magnitude):
    matrix = utils.quat_wxyz_to_matrix_numpy(
        np.array([1.0, 0.0, 0.0, 1.0]) * magnitude
    )
    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix @ [1, 0, 0], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(matrix @ [0, 0, 1], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)


@pytest.mark.parametrize(
    "quat",
    [
        None,
        np.eye(3),
        [1, 0, 0],
        [0, 0, 0, 0],
        [1e-7, 0, 0, 0],
        [np.nan, 0, 0, 0],
        [np.inf, 0, 0, 0],
    ],
)
def test_quaternion_matrix_rejects_invalid_input(quat):
    with pytest.raises(ValueError):
        utils.quat_wxyz_to_matrix_numpy(quat)


@pytest.mark.parametrize("shape", [(4,), (1, 4), (5, 4), (2, 3, 4), (2, 0, 4)])
def test_quaternion_matrix_batches_match_vector_rotation(shape):
    rng = np.random.default_rng(42)
    # A strided view also checks that conversion does not depend on contiguous input.
    quat = rng.normal(size=shape[:-1] + (8,))[..., ::2]
    vector = rng.normal(size=shape[:-1] + (3,))
    matrix = utils.quat_wxyz_to_matrix_numpy(quat)
    assert matrix.shape == shape[:-1] + (3, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(
        (matrix @ vector[..., None])[..., 0],
        utils.quat_wxyz_rotate_numpy(quat, vector),
        atol=1e-6,
    )


@pytest.mark.parametrize("invalid", [[0, 0, 0, 0], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]])
def test_quaternion_matrix_rejects_invalid_batch_member(invalid):
    with pytest.raises(ValueError):
        utils.quat_wxyz_to_matrix_numpy([[1, 0, 0, 0], invalid])


def test_batched_quaternion_operations_preserve_conventions_and_broadcasting():
    rotation_vectors = np.array([[[0, 0, 0], [0, 0, np.pi / 2]]])
    quat = utils.quat_wxyz_from_rotation_vector_numpy(rotation_vectors)
    assert quat.shape == (1, 2, 4)
    np.testing.assert_allclose(
        utils.quat_wxyz_to_rotation_vector_numpy(quat), rotation_vectors, atol=1e-6
    )
    np.testing.assert_array_equal(
        utils.quat_xyzw_to_wxyz(utils.quat_wxyz_to_xyzw(quat)), quat
    )
    np.testing.assert_allclose(
        utils.quat_wxyz_multiply_numpy([1, 0, 0, 0], quat), quat, atol=1e-6
    )
    blended = utils.quat_wxyz_slerp_numpy(
        [1, 0, 0, 0], quat, np.array([[0.0, 1.0]])
    )
    np.testing.assert_allclose(blended, quat, atol=1e-6)
    rotated = utils.quat_wxyz_rotate_numpy(quat, [1, 0, 0])
    np.testing.assert_allclose(rotated, [[[1, 0, 0], [0, 1, 0]]], atol=1e-6)


@pytest.mark.parametrize(
    "function, args, name",
    [
        (utils.quat_wxyz_to_xyzw, ([1, 0, 0],), "quat"),
        (utils.quat_xyzw_to_wxyz, (0,), "quat"),
        (utils.quat_wxyz_multiply_numpy, ([1, 0, 0], [1, 0, 0, 0]), "first"),
        (utils.quat_wxyz_multiply_numpy, ([1, 0, 0, 0], [1, 0, 0]), "second"),
        (utils.quat_wxyz_to_rotation_vector_numpy, ([1, 0, 0],), "quat"),
        (utils.quat_wxyz_from_rotation_vector_numpy, ([1, 0, 0, 0],), "rotation_vector"),
        (utils.quat_wxyz_rotate_numpy, ([1, 0, 0], [1, 0, 0]), "quat"),
        (utils.quat_wxyz_rotate_numpy, ([1, 0, 0, 0], [1, 0]), "vector"),
        (utils.quat_wxyz_slerp_numpy, ([1, 0, 0], [1, 0, 0, 0], 0.5), "first"),
        (utils.quat_wxyz_slerp_numpy, ([1, 0, 0, 0], [1, 0, 0], 0.5), "second"),
    ],
)
def test_batch_shape_errors_identify_argument(function, args, name):
    with pytest.raises(ValueError, match=f"{name} must have a final dimension"):
        function(*args)
