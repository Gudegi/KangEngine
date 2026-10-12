"""Internal array validation/interpolation shared by feature builders."""

import numpy as np
import numpy.typing as npt
from ...utils.math import quat_wxyz_slerp_numpy

_EPSILON = 1e-6


def _readonly(value: npt.ArrayLike, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_quaternions(value: npt.ArrayLike, name: str) -> np.ndarray:
    """Check unit quaternions and return norms for optional normalization."""
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite WXYZ quaternions")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms < _EPSILON):
        raise ValueError(f"{name} must not contain zero quaternions")
    if not np.allclose(norms, 1.0, atol=1.0e-3, rtol=0.0):
        raise ValueError(f"{name} must contain normalized WXYZ quaternions")
    return norms


def _unit_quaternions(value: npt.ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    return result / _validate_quaternions(result, name)


def _interpolate(values: np.ndarray, frame: npt.ArrayLike) -> np.ndarray:
    frame = np.clip(np.asarray(frame, dtype=np.float64), 0, len(values) - 1)
    before = np.floor(frame).astype(np.int64)
    after = np.minimum(before + 1, len(values) - 1)
    blend = (frame - before).reshape(frame.shape + (1,) * (values.ndim - 1))
    return values[before] * (1.0 - blend) + values[after] * blend


def _interpolate_rotation(values: np.ndarray, frame: npt.ArrayLike) -> np.ndarray:
    frame = np.clip(np.asarray(frame, dtype=np.float64), 0, len(values) - 1)
    before = np.floor(frame).astype(np.int64)
    after = np.minimum(before + 1, len(values) - 1)
    return quat_wxyz_slerp_numpy(values[before], values[after], frame - before)
