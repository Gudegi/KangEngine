"""Helpers for presenting pybind11 objects as public Python API."""
from __future__ import annotations

from types import ModuleType


def export_public_module(source: ModuleType, target_globals: dict[str, object]) -> list[str]:
    """Copy public names from a pybind11 submodule into a Python wrapper module."""
    module_name = str(target_globals["__name__"])
    public_names = [name for name in dir(source) if not name.startswith("_")]

    for name in public_names:
        value = getattr(source, name)
        target_globals[name] = value
        try:
            value.__module__ = module_name
        except (AttributeError, TypeError):
            pass

    target_globals["_native"] = source
    target_globals["__all__"] = public_names
    return public_names


def set_public_module(value: object, module_name: str) -> object:
    """Best-effort helper for making pybind11 objects document as public names."""
    try:
        value.__module__ = module_name
    except (AttributeError, TypeError):
        pass
    return value
