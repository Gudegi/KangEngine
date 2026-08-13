"""Optional adapters for simulation and learning frameworks."""

from importlib import import_module

__all__ = ["mimickit", "newton"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value
