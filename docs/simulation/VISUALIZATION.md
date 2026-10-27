# Simulation Visualization

Display simulated objects and keep their visuals aligned with the latest
physics state. `ke.visual.sim.SimWorldVisualizer` connects a `KangSimWorld`
to an App and synchronizes poses before rendering.

For objects driven by supplied poses without physics, use
[`ke.visual.RigidVisual`](RIGID_BODIES.md#render-without-physics) and call
`apply_pose()` directly.

## Batched rendering

Use `add()` for normal simulation rendering, including batched environments.
The snippets extend an `App` subclass with `import kangengine as ke`, as in
[First Simulation](../getting_started/FIRST_SIMULATION.md). In `setup()`, create
`self.world` and register an articulation as shown in
[Articulations](ARTICULATION.md); keep the returned `SimArticulation` as
`record`. The following code also belongs in `setup()` and uses that record's
registered visual geometry and materials:

```python
self.visual: ke.visual.sim.SimWorldVisualizer = (
    ke.visual.sim.SimWorldVisualizer(app=self, world=self.world)
)
self.robot_visual: ke.visual.sim.VisualBatch = self.visual.add(
    sim_handle=record,
    path="/robot",
)
```

Add these methods to the same App class:

```python
def fixed_update(self, fixed_dt):
    self.world.advance(duration=fixed_dt)

def pre_render(self):
    self.visual.sync()
```

`add()` returns a `VisualBatch`, not an `ArticulationVisual`. It is the normal
rendering handle for both rigid bodies and articulations and uses an
ExternalBuffer-backed path.

Rigid and articulation batches choose their CPU or CUDA rendering path from the
simulation device. CUDA batches update render transforms directly from GPU
simulation state. Static rigid bodies use the CPU path in either kind of world.
Both paths preserve each visual part's local position, rotation, and scale.

Rigid bodies registered from a `RigidDesc` already carry their visual geometry
and material subsets. Use `visualizer.add(sim_handle=rigids, path="/props")`
without an MJCF path. The same call renders rigid descriptions using
their collision shapes when no visual parts are authored. Articulations can also
use the `VisualGeomDesc` parts in their registered `ArticulationDesc` without
reloading an MJCF. See [Rigid Bodies](RIGID_BODIES.md#prepare-a-rigid-description)
for textured objects whose visual surface differs from their colliders.

`VisualGeomDesc.material` accepts the same PBR and Phong materials used by the
renderer. A source material's `texture_paths` can defer texture loading until
the visual is attached to an App. Rigid and articulation visuals retain the
source appearance while creating their own App-owned runtime materials.

Simulation state and scene/render state are separate. Advance physics in
`fixed_update()` and synchronize visuals in `pre_render()` so rendering speed
does not alter the physics rate.

## Inspectable SceneGraph visuals

Use `add_scene_graph()` when individual prims must be visible and selectable in
the editor. In `setup()`, replace the `self.visual.add(...)` call above with:

```python
robot_visual: ke.visual.sim.VisualArticulationSceneGraph = self.visual.add_scene_graph(
    sim_handle=record,
    env_id=0,
)
robot_visual.set_color(color=(0.8, 0.8, 0.9, 1.0))
robot_visual.set_collision_visible(visible=True)
```

For an articulation, `robot_visual` is a `VisualArticulationSceneGraph` and
`robot_visual.articulation_visual` is the underlying `ArticulationVisual`
synchronized from the simulation. Most applications should use the record
rather than call the underlying bridge directly.

`VisualBatch.release()` removes one batch. `SimWorldVisualizer.release()`
releases every visual registered through that visualizer.

Use one representation deliberately:

- `add()` / `VisualBatch` for throughput and batched environments.
- `add_scene_graph()` for inspectable prims, selection, and collision display.

## Visual types

| Type | Role |
|---|---|
| `SimWorldVisualizer` | Registers simulation objects and synchronizes their visual state |
| `VisualBatch` | CPU/GPU external-buffer visual returned by `add()` |
| `VisualArticulationSceneGraph` | Inspectable per-link articulation record returned by `add_scene_graph()` |
| `VisualRigidSceneGraph` | Inspectable rigid-body record returned by `add_scene_graph()` |
| `VisualBodyPick` | Maps a renderer selection to environment, object, and body IDs |

## Examples

- `python/examples/sim_world_minimal.py`
- `python/examples/sim_world_multi_env.py`
- `python/examples/mjcf_dof_control.py`
