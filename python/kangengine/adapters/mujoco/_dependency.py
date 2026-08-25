"""Optional MuJoCo dependency loading."""

from __future__ import annotations

from importlib import import_module


class MuJoCoUnavailableError(ImportError):
    """Raised when the optional MuJoCo package is unavailable."""


def load_mujoco():
    """Return the imported MuJoCo module."""

    try:
        return import_module("mujoco")
    except ImportError as error:
        raise MuJoCoUnavailableError(
            "MuJoCo motion conversion requires the optional 'mujoco' package."
        ) from error
