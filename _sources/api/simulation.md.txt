# ke.sim


High-level Python simulation helpers built on the low-level PhysX bindings.

## API overview

```{eval-rst}
.. currentmodule:: kangengine.sim

.. autosummary::
   :nosignatures:

   KangSimWorld
   GridCloner
   SimulationRuntime
   SimulationTimingConfig
   SimulationRunConfig
   SimArticulation
   SimRigid
```
## State return and lifetime rules

Simulation getters return Torch tensors on the configured state device. They are
views into reusable state storage, not immutable snapshots; use `tensor.clone()`
when data must survive later `step()`, `refresh()`, reset, or `release()` calls.
Shape labels use `N` for environments, `B` for bodies, and `D` for DOFs.

| API | Return and lifetime contract |
| --- | --- |
| `SimArticulation`, `SimRigid`, and batch `get_*()` | Tensor views backed by the world's reusable state cache. |
| `KangSimWorld.get_gpu_*()` | Zero-copy CUDA views over PhysX GPU mirrors; fetching or stepping updates their contents. |
| `ContactSensor.data` / `refresh()` | Views over reused sensor output buffers; clone fields needed as snapshots. |
| `KangSimWorld.get_articulation()` / `get_rigid()` | Lightweight handles tied to the world; do not use them after `world.release()`. |

Invalid object IDs or names raise `KeyError`; incompatible shapes or configuration
values raise `ValueError`; unavailable devices, uninitialized GPU state, and use
after release raise `RuntimeError`.


```{eval-rst}
.. currentmodule:: kangengine.sim

.. autoclass:: KangSimWorld

.. autoclass:: SimArticulation

.. autoclass:: SimArticulationBatch

.. autoclass:: SimRigid

.. autoclass:: SimRigidBatch

.. autoclass:: SimulationRuntime

.. autoclass:: ArticulationStateView

.. autoclass:: SimulationTimingConfig
   :members:

.. autoclass:: SimulationRunMode
   :members:

.. autoclass:: SimulationRunConfig
   :members:

.. autoclass:: SimulationPacer
   :members:

.. autoclass:: ControlMode

.. autoclass:: SimDevice

.. currentmodule:: kangengine.state

.. autoclass:: ArticulationState

.. autoclass:: ArticulationStateCache

.. autoclass:: RigidStateCache

.. autoclass:: ArticulationRecord
```

```{include} sensor.md
:heading-offset: 1
```
