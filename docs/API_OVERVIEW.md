# API Overview

## Quaternion ordering

`ke.quat` objects consistently use `wxyz` ordering, including construction
from Python or NumPy values and conversion with `np.asarray(q)`. Use
`ke.quat.from_xyzw(...)` and `q.to_xyzw()` at APIs that explicitly use
`rot_xyzw` arrays. Physics and simulation state arrays retain their documented
`xyzw` ordering.

KangEngine's Python API is organized by task and domain. Start with the user
guide for runnable workflows; use the API Reference for exact signatures.

## Common entry points

| Goal | Start here |
|---|---|
| Create a window or tool | `ke.App` |
| Add authored scene objects | `app.scene` and `ke.geometry` |
| Load FBX, BVH, MJCF, OBJ, or optional USD | `ke.asset` |
| Configure appearance | `ke.material` |
| Run Python simulation | `ke.sim.KangSimWorld` |
| Configure render, physics, and fixed-update rates | `ke.SimulationTimingConfig` |
| Display simulation state | `ke.visual.sim.SimWorldVisualizer` |
| Inspect or edit motion | `ke.motion_module` |
| Use low-level renderer objects | `ke.render` |
| Use low-level PhysX objects | `ke.physics` |

## Recommended reading

1. [Installation](getting_started/INSTALLATION.md)
2. [Hello App](getting_started/HELLO_APP.md)
3. [First Scene](getting_started/FIRST_SCENE.md)
4. [First Simulation](getting_started/FIRST_SIMULATION.md)
5. [Scene and Rendering](scene_rendering/INDEX.md) or
   [Simulation](simulation/INDEX.md)

Architecture and backend details are intentionally kept under
[Advanced Topics](advanced/INDEX.md).
