"""Simulation visual sync API.

Objects in this submodule mirror ``KangSimWorld`` state into scene/render
visuals. They are separated from static visual asset wrappers to keep the
ownership model explicit.
"""

from __future__ import annotations

from .batch import VisualBatch
from .world_visualizer import SimWorldVisualizer
from .records import (
    VisualArticulationSceneGraph,
    VisualBodyPick,
    VisualRigidSceneGraph,
)

__all__ = [
    "SimWorldVisualizer",
    "VisualArticulationSceneGraph",
    "VisualBatch",
    "VisualBodyPick",
    "VisualRigidSceneGraph",
]

for _name in __all__:
    _value = globals()[_name]
    try:
        _value.__module__ = __name__
    except (AttributeError, TypeError):
        pass

del _name, _value
