# API Reference

This section contains KangEngine's Python API reference, split between the
native pybind11 surface and higher-level pure Python workflow helpers.

```{toctree}
:hidden:

python
scene
geometry
asset
animation
character
motion
material
rendering
physics
simulation
terrain
visual
utils
```

The reference follows the public `kangengine` and `ke.*` package roots directly.

## Native Runtime

```{eval-rst}
.. currentmodule:: kangengine.render

.. autosummary::
    :nosignatures:

    GraphicsDevice
    Shader
    Texture

```

Materials are documented separately under the `ke.material` domain.

## Scene And Assets

```{eval-rst}
.. currentmodule:: kangengine.scene

.. autosummary::
    :nosignatures:

    Prim
    MeshData
    SkinnedMeshData
    SceneBackend

.. currentmodule:: kangengine.asset

.. autosummary::
    :nosignatures:

    MJCFLoader
    BVHLoader
    FBXLoader
    USDLoader
```

## Animation And Physics

```{eval-rst}
.. currentmodule:: kangengine.animation

.. autosummary::
    :nosignatures:

    SkeletonTree
    SkeletonMotion
    SkeletonState

.. currentmodule:: kangengine.visual

.. autosummary::
    :nosignatures:

    ArticulationVisual
    SkeletalVisual

.. currentmodule:: kangengine.physics

.. autosummary::
    :nosignatures:

    PhysicsWorld
    Articulation
    PhysicsBridge
    PhysicsGpuSystem
```

## Python Workflow Layer

```{eval-rst}
.. currentmodule:: kangengine

.. autosummary::
    :nosignatures:

    DebugGeometry
    DebugOverlay
    WorldText
    ScreenText

.. currentmodule:: kangengine.sim

.. autosummary::
    :nosignatures:

    KangSimWorld

.. currentmodule:: kangengine.motion_module

.. autosummary::
    :nosignatures:

    MotionEditor
    MotionPlayer
    MotionModule

.. currentmodule:: kangengine

.. autosummary::
    :nosignatures:

    JointMapper

.. currentmodule:: kangengine.visual.sim

.. autosummary::
    :nosignatures:

    SimWorldVisualizer
```
