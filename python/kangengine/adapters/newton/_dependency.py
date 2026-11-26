"""Lazy Newton/Warp dependency discovery."""

from __future__ import annotations

from importlib import import_module


class NewtonUnavailableError(ImportError):
    """Raised when the optional Newton adapter is used without Newton."""


def load_newton():
    """Return the imported ``newton`` and ``warp`` modules.

    Dependency loading stays inside the adapter so importing :mod:`kangengine`
    never initializes Warp or a CUDA context.
    """

    try:
        newton = import_module("newton")
        warp = import_module("warp")
    except ImportError as exc:
        raise NewtonUnavailableError(
            "The KangEngine Newton adapter requires Newton and Warp. "
            "Install 'kangengine[newton]' for Newton, MuJoCo, and USD import support. Importing "
            "kangengine itself does not require them."
        ) from exc
    return newton, warp


def load_viewer_base():
    """Return Newton's renderer-agnostic base viewer class.

    Newton does not currently re-export ``ViewerBase`` from ``newton.viewer``.
    Keep this version-sensitive import isolated in one adapter module.
    """

    load_newton()
    try:
        module = import_module("newton._src.viewer.viewer")
        return module.ViewerBase
    except (ImportError, AttributeError) as exc:
        raise NewtonUnavailableError(
            "This Newton version does not provide the ViewerBase contract "
            "expected by KangEngine."
        ) from exc


def load_picking():
    """Return Newton's viewer picking helper from the isolated private API."""

    load_newton()
    try:
        module = import_module("newton._src.viewer.picking")
        return module.Picking
    except (ImportError, AttributeError) as exc:
        raise NewtonUnavailableError(
            "This Newton version does not provide the Picking contract "
            "expected by KangEngine."
        ) from exc


def is_newton_available() -> bool:
    try:
        load_newton()
    except NewtonUnavailableError:
        return False
    return True


def load_raycast_kernel():
    """Reuse Newton's shape raycasts without starting its force-drag interaction."""
    return import_module("newton._src.geometry.raycast").raycast_kernel
