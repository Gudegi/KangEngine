"""Optional PyRoki dependency loading."""

from __future__ import annotations

from importlib import import_module


class PyrokiUnavailableError(ImportError):
    """Raised when optional PyRoki adapter dependencies are unavailable."""


def load_pyroki():
    """Return PyRoki and yourdfpy without affecting base package imports."""

    try:
        pyroki = import_module("pyroki")
        yourdfpy = import_module("yourdfpy")
    except ImportError as error:
        raise PyrokiUnavailableError(
            "The KangEngine PyRoki adapter requires the optional 'pyroki' "
            "and 'yourdfpy' packages. Use the project-pinned PyRoki/JAXLS "
            "revisions; importing kangengine itself does not require them."
        ) from error
    return pyroki, yourdfpy
