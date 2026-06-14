"""Torch-first tensor helpers around the current CPU-native engine boundary."""

import numpy as np
import torch


def resolve_device(device=None) -> torch.device:
    """Return the public tensor device, defaulting to CPU."""
    return torch.device("cpu" if device is None else device)


def as_tensor(value, *, shape=None, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a tensor on the requested public API device."""
    tensor = torch.as_tensor(value, dtype=dtype, device=resolve_device(device))
    return tensor if shape is None else tensor.reshape(shape)


def as_cpu_numpy(value, *, shape=None, dtype=np.float32) -> np.ndarray:
    """Copy or view a value as a CPU NumPy array for CPU-native bindings."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    return array if shape is None else array.reshape(shape)


def as_sim_buffer(value, *, shape=None, sim_device="cpu", dtype=np.float32):
    """Return a buffer suitable for the selected simulation backend.

    CPU simulation currently uses NumPy because the native PhysX bindings are
    CPU-facing. CUDA simulation is intentionally left as a future zero-copy
    interop boundary instead of silently copying through CPU memory.
    """
    if sim_device is None:
        sim_device = "cpu"
    device = str(sim_device).strip().lower()
    if device == "cpu":
        return as_cpu_numpy(value, shape=shape, dtype=dtype)
    if device == "cuda" or device.startswith("cuda:"):
        raise NotImplementedError(
            "CUDA simulation buffers require Torch/C++ CUDA interop and are not "
            "implemented yet"
        )
    raise ValueError(
        "sim_device must be 'cpu', 'cuda', or 'cuda:<ordinal>' "
        f"(got {sim_device!r})"
    )
