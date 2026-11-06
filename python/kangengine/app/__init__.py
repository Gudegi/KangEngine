"""Python-friendly application and scene workflow facades."""

from .._public import set_public_module
from .application import (
    App,
    DebugContext,
    DebugGeometry,
    DebugOverlay,
    ObjImportView,
    RenderablePrimView,
    ScreenText,
    SceneContext,
    WorldText,
)

__all__ = [
    "App",
    "DebugContext",
    "DebugGeometry",
    "DebugOverlay",
    "ObjImportView",
    "RenderablePrimView",
    "ScreenText",
    "SceneContext",
    "WorldText",
]

for _type in (
    App,
    DebugGeometry,
    DebugOverlay,
    ObjImportView,
    RenderablePrimView,
    ScreenText,
    SceneContext,
    WorldText,
):
    set_public_module(_type, "kangengine")

set_public_module(DebugContext, "kangengine.app")

del _type
