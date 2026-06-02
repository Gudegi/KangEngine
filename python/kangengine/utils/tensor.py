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


def as_native_numpy(value, *, shape=None, dtype=np.float32) -> np.ndarray:
    """Copy or view a value as a CPU NumPy array for the current native backend."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    return array if shape is None else array.reshape(shape)
