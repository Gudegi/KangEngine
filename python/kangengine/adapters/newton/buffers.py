"""CPU and CUDA buffer conversion helpers for the Newton adapter."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np


def to_numpy(value, *, name: str, allow_cuda_readback: bool = False) -> np.ndarray:
    """Convert a NumPy or Warp-like array to a host NumPy array.

    CUDA readback must be explicitly enabled. This prevents the initial CPU
    viewer implementation from silently turning a CUDA simulation into a
    device-to-host copy every frame.
    """

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value

    device = getattr(value, "device", None)
    is_cuda = bool(getattr(device, "is_cuda", False)) or str(device).startswith(
        "cuda"
    )
    if is_cuda and not allow_cuda_readback:
        raise RuntimeError(
            f"{name} is stored on CUDA. This NewtonViewer feature refuses "
            "implicit readback; pass allow_cuda_readback=True for debugging "
            "or use a device-resident adapter path."
        )

    numpy_method = getattr(value, "numpy", None)
    if numpy_method is None:
        return np.asarray(value)
    return np.asarray(numpy_method())


def rgba_array(colors, count: int, *, allow_cuda_readback: bool) -> np.ndarray:
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)
    if colors is None:
        return np.ones((count, 4), dtype=np.float32)
    values = np.asarray(
        to_numpy(
            colors,
            name="colors",
            allow_cuda_readback=allow_cuda_readback,
        ),
        dtype=np.float32,
    )
    if values.ndim == 1:
        if values.size in (3, 4):
            channels = values.size
        elif values.size == count * 3:
            channels = 3
        elif values.size == count * 4:
            channels = 4
        else:
            raise ValueError(
                "colors must contain RGB or RGBA entries; "
                f"received {values.size} scalar values for {count} colors"
            )
    elif values.shape[-1] in (3, 4):
        channels = values.shape[-1]
    else:
        raise ValueError(
            "colors must have a final dimension of 3 (RGB) or 4 (RGBA); "
            f"received shape {values.shape}"
        )
    values = values.reshape(-1, channels)
    if len(values) == 1 and count > 1:
        values = np.repeat(values, count, axis=0)
    if len(values) != count:
        raise ValueError(f"colors has {len(values)} entries; expected 1 or {count}")
    if channels == 4:
        return values
    alpha = np.ones((count, 1), dtype=np.float32)
    return np.concatenate((values, alpha), axis=1)


def transform_array_to_torch_matrices(xforms, scales=None, out=None):
    """Convert Warp transforms to GLM-storage matrices without a host copy.

    ``wp.to_torch`` aliases the Warp allocation on both CPU and CUDA. When the
    input is CUDA, conversion runs on Warp's active stream and returns that
    Torch stream so the renderer descriptor can retain producer ordering.
    """

    import torch
    import warp as wp

    transforms = wp.to_torch(xforms).reshape(-1, 7)
    count = int(transforms.shape[0])
    if scales is None:
        scale_values = None
    else:
        scale_values = wp.to_torch(scales).reshape(-1, 3)
        if len(scale_values) not in (1, count):
            raise ValueError(
                f"scales has {len(scale_values)} entries; expected 1 or {count}"
            )

    if (
        out is None
        or tuple(out.shape) != (count, 4, 4)
        or out.device != transforms.device
    ):
        out = torch.empty(
            (count, 4, 4), dtype=torch.float32, device=transforms.device
        )

    is_cuda = transforms.device.type == "cuda"
    torch_stream = (
        wp.stream_to_torch(wp.get_stream(xforms.device)) if is_cuda else None
    )
    context = torch.cuda.stream(torch_stream) if is_cuda else nullcontext()
    with context:
        q = transforms[:, 3:7]
        q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True).clamp_min(1.0e-12)
        x, y, z, w = q.unbind(dim=1)
        if scale_values is None:
            sx = sy = sz = torch.ones_like(x)
        else:
            if len(scale_values) == 1 and count > 1:
                scale_values = scale_values.expand(count, 3)
            sx, sy, sz = scale_values.unbind(dim=1)

        # GLM column-major storage: each first matrix index is one basis column.
        out.zero_()
        out[:, 0, 0] = (1.0 - 2.0 * (y * y + z * z)) * sx
        out[:, 0, 1] = (2.0 * (x * y + z * w)) * sx
        out[:, 0, 2] = (2.0 * (x * z - y * w)) * sx
        out[:, 1, 0] = (2.0 * (x * y - z * w)) * sy
        out[:, 1, 1] = (1.0 - 2.0 * (x * x + z * z)) * sy
        out[:, 1, 2] = (2.0 * (y * z + x * w)) * sy
        out[:, 2, 0] = (2.0 * (x * z + y * w)) * sz
        out[:, 2, 1] = (2.0 * (y * z - x * w)) * sz
        out[:, 2, 2] = (1.0 - 2.0 * (x * x + y * y)) * sz
        out[:, 3, :3] = transforms[:, :3]
        out[:, 3, 3] = 1.0

    return out, torch_stream
