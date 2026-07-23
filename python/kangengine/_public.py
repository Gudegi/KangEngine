"""Helpers for presenting pybind11 objects as public Python API."""
from __future__ import annotations

from types import ModuleType
from typing import Any


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


def unwrap_native(obj: Any) -> Any:
    """Return ``obj.native``/``obj._native`` for thin KangEngine wrappers."""
    if hasattr(obj, "native"):
        return getattr(obj, "native")
    return getattr(obj, "_native", obj)


class NativeWrapper:
    """Small forwarding base for Python facades around pybind11 objects."""

    _native: Any

    def __init__(self, native: Any):
        object.__setattr__(self, "_native", native)

    @property
    def native(self) -> Any:
        """Native pybind11 object for advanced interop and hot paths."""
        return self._native

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_native":
            object.__setattr__(self, name, value)
            return
        descriptor = getattr(type(self), name, None)
        if hasattr(descriptor, "__set__"):
            descriptor.__set__(self, value)
            return
        setattr(self._native, name, unwrap_native(value))

    def __repr__(self) -> str:
        return repr(self._native)
