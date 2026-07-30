# Application

Application lifecycle, camera access, and viewport interaction.

```{eval-rst}
.. currentmodule:: kangengine

.. autoclass:: App

.. autoclass:: SimulationTimingConfig
   :members:

.. autoclass:: SceneContext

.. autoclass:: DebugGeometry

.. autoclass:: DebugOverlay

.. autoclass:: WorldText

.. autoclass:: ScreenText

.. autoclass:: RenderablePrimView

.. autoclass:: Camera

.. autoclass:: RayPickResult
```

## ke.imgui

Small Dear ImGui binding for application panels and controls. Access these
functions through `ke.imgui`.

```{eval-rst}
.. automodule:: kangengine.imgui
   :members:
   :imported-members:
   :member-order: alphabetical
   :undoc-members:
```

## ke.keys

Keyboard constants accepted by the `App` input helpers. Access them through
`ke.keys`, for example `ke.keys.ESCAPE`, `ke.keys.SPACE`, or `ke.keys.A`.

```{eval-rst}
.. automodule:: kangengine.keys
   :members:
   :imported-members:
   :member-order: alphabetical
   :undoc-members:
```
