# Simulation Visualization

`ke.visual.sim` synchronizes `KangSimWorld` objects to renderer and scene
visuals.

| Type | Role |
|---|---|
| `SimWorldVisualizer` | Registers simulation objects and synchronizes their visual state |
| `VisualBatch` | CPU/GPU external-buffer visual returned by `add()` |
| `VisualArticulationSceneGraph` | Inspectable per-link articulation record returned by `add_scene_graph()` |
| `VisualRigidSceneGraph` | Inspectable rigid-body record returned by `add_scene_graph()` |
| `VisualBodyPick` | Maps a renderer selection to environment, object, and body IDs |

## Batched rendering

Use `add()` for normal simulation rendering, including batched environments:

```python
self.visual: ke.visual.sim.SimWorldVisualizer = (
    ke.visual.sim.SimWorldVisualizer(app=self, world=self.world)
)
self.robot_visual: ke.visual.sim.VisualBatch = self.visual.add(
    sim_handle=robot,
    mjcf_path=mjcf_path,
    path="/robot",
    material=material,
)

def fixed_update(self, fixed_dt):
    self.world.advance(duration=fixed_dt)

def pre_render(self):
    self.visual.sync()
```

`add()` returns a `VisualBatch`, not an `ArticulationVisual`. It is the normal
rendering handle for both rigid bodies and articulations and uses an
ExternalBuffer-backed path.

Simulation state and scene/render state are separate. Advance physics in
`fixed_update()` and synchronize visuals in `pre_render()` so rendering speed
does not alter the physics rate.

## Inspectable SceneGraph visuals

Use `add_scene_graph()` when individual prims must be visible and selectable in
the editor:

```python
record: ke.visual.sim.VisualArticulationSceneGraph = self.visual.add_scene_graph(
    sim_handle=robot,
    mjcf_path=mjcf_path,
    env_id=0,
)
record.set_color(color=(0.8, 0.8, 0.9, 1.0))
record.set_collision_visible(visible=True)
```

For an articulation, `record` is a `VisualArticulationSceneGraph` and
`record.articulation_visual` is the underlying `ArticulationVisual`
synchronized from the simulation. Most applications should use the record
rather than call the underlying bridge directly.

`VisualBatch.release()` removes one batch. `SimWorldVisualizer.release()`
releases every visual registered through that visualizer.

Use one representation deliberately:

- `add()` / `VisualBatch` for throughput and batched environments.
- `add_scene_graph()` for inspectable prims, selection, and collision display.

Examples:

- `python/examples/sim_world_minimal.py`
- `python/examples/sim_world_multi_env.py`
- `python/examples/mjcf_dof_control.py`
