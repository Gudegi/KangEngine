# API Reference

This section contains KangEngine's public Python API. Examples conventionally
use `import kangengine as ke`; `ke` is only a local alias for the `kangengine`
package, not a separate package or API layer.

The public API exposes native pybind11 types and pure Python helpers side by
side under the documented `ke.*` paths. Their implementation origin does not
affect how they are imported or called. Underscored implementation modules are
internal and may change without notice.

<span class="api-origin api-origin-native">[native]</span> identifies compiled
pybind11 APIs. <span class="api-origin api-origin-python">[python]</span>
identifies Python workflow helpers. Meaningful public inheritance is still shown;
implementation-only base classes are omitted.

Application-wide entry points and common app helpers live directly under `ke`,
while specialized APIs are grouped by domain under modules such as `ke.scene`,
`ke.physics`, and `ke.material`.

```{toctree}
:hidden:

python
scene
geometry
asset
animation
exports
motion
material
rendering
physics
simulation
terrain
visual
utils
```

## Application

```{eval-rst}
.. currentmodule:: kangengine

.. autosummary::
    :nosignatures:

    App
    DebugGeometry
    DebugOverlay
    WorldText
    ScreenText
```

## Rendering And Materials

### Rendering

```{eval-rst}
.. currentmodule:: kangengine.render

.. autosummary::
    :nosignatures:

    Renderer
    GraphicsDevice
    Texture
    Buffer
    GraphicsPipeline
    ShaderDesc
    SceneHookPipelineDesc
```

### Materials

```{eval-rst}
.. currentmodule:: kangengine.material

.. autosummary::
    :nosignatures:

    Material
    VertexColorMaterial
    PhongMaterial
    PBRMaterial
```

## Scene And Assets

### Scene

```{eval-rst}
.. currentmodule:: kangengine.scene

.. autosummary::
    :nosignatures:

    Prim
    MeshData
    SkinnedMeshData
    SceneBackend
```

### Assets

```{eval-rst}
.. currentmodule:: kangengine.asset

.. autosummary::
    :nosignatures:

    MJCFLoader
    BVHLoader
    FBXLoader
    USDLoader
```

## Animation And Visuals

### Animation

```{eval-rst}
.. currentmodule:: kangengine.animation

.. autosummary::
    :nosignatures:

    SkeletonTree
    SkeletonMotion
    SkeletonState
```

### Visual Bridges

```{eval-rst}
.. currentmodule:: kangengine.visual

.. autosummary::
    :nosignatures:

    ArticulationVisual
    ArticulatedSurface
    SkeletalVisual
```

### Deformable Surfaces

```{eval-rst}
.. currentmodule:: kangengine.visual

.. autosummary::
    :nosignatures:

    SkinnedSurfaceAsset
    SkinnedSurface
    ArticulatedSurfaceAsset
    DeformableSurface
```

### Exports

```{eval-rst}
.. currentmodule:: kangengine.exports

.. autosummary::
    :nosignatures:

    save_motion_bvh
```

## Physics And Simulation

### Physics

```{eval-rst}
.. currentmodule:: kangengine.physics

.. autosummary::
    :nosignatures:

    PhysicsWorld
    Articulation
    PhysicsBridge
    PhysicsGpuSystem
```

### Simulation

```{eval-rst}
.. currentmodule:: kangengine.sim

.. autosummary::
    :nosignatures:

    KangSimWorld
```

### Simulation Visualization

```{eval-rst}
.. currentmodule:: kangengine.visual.sim

.. autosummary::
    :nosignatures:

    SimWorldVisualizer
```

## Motion And Utilities

### Motion

```{eval-rst}
.. currentmodule:: kangengine.motion_module

.. autosummary::
    :nosignatures:

    MotionEditor
    MotionPlayer
    MotionModule
```

### Joint Mapping

```{eval-rst}
.. currentmodule:: kangengine.utils

.. autosummary::
    :nosignatures:

    JointMapper
```
