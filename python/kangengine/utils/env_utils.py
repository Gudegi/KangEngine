"""Environment id helpers shared by simulation APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from .tensor import as_cpu_numpy

if TYPE_CHECKING:
    import torch

    EnvIdLike: TypeAlias = (
        int | np.integer | Sequence[int] | np.ndarray | torch.Tensor | None
    )
else:
    EnvIdLike: TypeAlias = int | np.integer | Sequence[int] | np.ndarray | None


def env_id_list(env_id: EnvIdLike, num_envs: int) -> list[int]:
    """Normalize scalar, sequence, tensor, or None env ids into a list."""
    if env_id is None:
        return list(range(int(num_envs)))
    if isinstance(env_id, (str, bytes)):
        raise TypeError("env_id must be an int, a sequence of ints, or None")
    if hasattr(env_id, "detach"):
        env_id = env_id.detach().cpu().numpy()
    arr = np.asarray(env_id)
    if arr.ndim == 0:
        env_ids = [int(arr.item())]
    else:
        env_ids = [int(v) for v in arr.reshape(-1)]
    for eid in env_ids:
        if eid < 0 or eid >= int(num_envs):
            raise ValueError(f"env_id {eid} out of range for num_envs={num_envs}")
    return env_ids


def select_env_value(value, env_index: int, env_count: int, shape):
    """Select one row from batched env values, or reshape a shared value."""
    arr = as_cpu_numpy(value)
    if arr.ndim > len(shape) and arr.shape[0] == env_count:
        arr = arr[env_index]
    return arr.reshape(shape)


def select_optional_env_value(value, env_index: int, env_count: int, shape):
    """Select an env value unless the optional value is absent."""
    if value is None:
        return None
    return select_env_value(value, env_index, env_count, shape)
