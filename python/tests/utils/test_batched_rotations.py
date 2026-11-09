"""Batched rotation conversion on CPU and CUDA."""

import pytest
import torch

from kangengine.utils.batched_rotations import matrix_to_quat_wxyz, quat_wxyz_to_matrix


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_quaternion_batch_preserves_rotation_and_branch_sign(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator().manual_seed(29)
    quaternions = torch.cat(
        (
            torch.eye(4, dtype=dtype),
            torch.tensor([[0.5, 0.5, 0.5, 0.5], [1e-7, -1, 0, 0]], dtype=dtype),
            torch.randn(250, 4, generator=generator, dtype=dtype),
        )
    ).to(device)
    quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
    # Transposition also checks noncontiguous batches and inverse rotations.
    matrices = quat_wxyz_to_matrix(quaternions).reshape(16, 16, 3, 3).transpose(-2, -1)
    expected = quaternions.reshape(16, 16, 4).clone()
    expected[..., 1:] *= -1

    if device.startswith("cuda"):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            actual = matrix_to_quat_wxyz(matrices)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    else:
        actual = matrix_to_quat_wxyz(matrices)

    diagonal = matrices.diagonal(dim1=-2, dim2=-1)
    branch = torch.where(diagonal.sum(-1) > 0, 0, diagonal.argmax(-1) + 1)
    sign = expected.gather(-1, branch.unsqueeze(-1)).sign()
    expected = expected * sign
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        quat_wxyz_to_matrix(actual), matrices, atol=1e-6, rtol=1e-6
    )
    assert actual.dtype == dtype and actual.device == matrices.device
    torch.testing.assert_close(matrix_to_quat_wxyz(matrices[0, 0]), actual[0, 0])
    assert matrix_to_quat_wxyz(matrices[:0]).shape == (0, 16, 4)


def test_matrix_quaternion_batch_gradients():
    quaternions = torch.tensor(
        [
            [0.9, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.2, 0.0],
            [0.1, 0.2, 0.9, 0.0],
            [0.1, 0.2, 0.0, 0.9],
        ],
        dtype=torch.float64,
    )
    matrices = quat_wxyz_to_matrix(quaternions).requires_grad_()
    assert torch.autograd.gradcheck(matrix_to_quat_wxyz, (matrices,), fast_mode=True)
