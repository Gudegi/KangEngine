# ke.visual

Python viewer-side bridges for scene objects, animated characters, and simulation
state visualization.

## Scene and asset visuals

```{eval-rst}
.. currentmodule:: kangengine.visual

.. autosummary::
   :nosignatures:

   ArticulationVisual
   ArticulationVisualAsset
   ArticulatedSurface
   ArticulatedSurfaceAsset
   DeformableSurface
   SkinVisual
   SkinnedSurface
   SkinnedSurfaceAsset
   SkeletalVisual
   SkeletalVisualConfig
```

```{eval-rst}
.. currentmodule:: kangengine.visual

.. autoclass:: ArticulationVisual

.. autoclass:: ArticulationVisualAsset

.. autoclass:: ArticulatedSurfaceAsset

.. autoclass:: ArticulatedSurface

.. autoclass:: SkinnedSurfaceAsset

.. autoclass:: SkinnedSurface

.. autoclass:: DeformableSurface

.. autoclass:: SkinVisual

.. autoclass:: SkeletalVisualConfig
   :special-members: __init__

.. autoclass:: SkeletalVisual
```

## Simulation visual sync

`ke.visual.sim` mirrors `KangSimWorld` objects into scene and render visuals.
These objects own visualization state only; simulation state remains owned by
the world and its simulation handles.

```{eval-rst}
.. currentmodule:: kangengine.visual.sim

.. autosummary::
   :nosignatures:

   SimWorldVisualizer
   VisualBatch
   VisualBodyPick
   VisualRigidSceneGraph
   VisualArticulationSceneGraph
```

```{eval-rst}
.. currentmodule:: kangengine.visual.sim

.. autoclass:: SimWorldVisualizer

.. autoclass:: VisualBatch

.. autoclass:: VisualBodyPick

.. autoclass:: VisualRigidSceneGraph

.. autoclass:: VisualArticulationSceneGraph
```
