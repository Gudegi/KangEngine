# Viewer Synchronization

Simulation state and scene state are separate. `SimWorldVisualizer` creates the
render representation and synchronizes it after simulation steps.

```python
visual = ke.visual.sim.SimWorldVisualizer(app, world)
visual_batch = visual.add(
    rigid_batch,
    rigid_xml,
    prim_base_path="/rigids",
    material=material,
)

world.step(substeps=2)
visual.sync()
```

`visual.add(...)` is the normal path for many simulated bodies and uses
ExternalBuffer-backed batches. For editor interaction, collision inspection,
or a small authored robot, use the explicit scene-graph helpers:

```python
visual.add_articulation_scene_graph(...)
visual.add_rigid_scene_graph(...)
```

Use one representation deliberately: ExternalBuffer for throughput,
SceneGraph for inspectable prims and interaction.

Examples:

- `python/examples/sim_world_minimal.py`
- `python/examples/sim_world_multi_env.py`
- `python/examples/mjcf_dof_control.py`
