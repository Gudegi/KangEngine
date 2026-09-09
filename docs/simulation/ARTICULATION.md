# Articulations

MJCF, URDF, and USD loaders produce a common `ArticulationDesc`.
Load a robot, then pass its data to `world.add_articulation()`.

## Load a robot

Choose the loader for your file:

```python
# world is an existing KangSimWorld.
data: ke.asset.ArticulationDesc = world.load_mjcf("robot.xml", order="DFS")
# Or:
data = world.load_urdf("robot.urdf", order="DFS")
data = world.load_usd("robot.usd", order="DFS")
```

USD requires a native build with `USE_USD` enabled. It selects the default
prim; pass `prim_path="/Robot"` to select a different robot subtree.

| Format | Supported joints | 3-axis rotation |
| --- | --- | --- |
| MJCF | Fixed, Revolute (hinge) | 3 orthogonal Revolute joints on one body |
| URDF | Fixed, Revolute, Prismatic | 3 Revolute joints with intermediate links |
| USD | Fixed, Revolute, Prismatic | 3 Revolute joints with intermediate links |

Native MJCF `ball`/`slide` and USD Spherical joint import are not supported.

## Create the articulation

```python
config = ke.physics.ArticulationConfig.free_base()

record = world.add_articulation(
    data,
    env_id=0,
    obj_id=0,
    name="robot",
    config=config,
)
robot = record.articulation
```

For all three formats, choose `ArticulationConfig.free_base()` for a floating
root or `ArticulationConfig.fixed_base()` for a fixed root.

The following visualization example uses an MJCF asset:

Use the same `order` for loading and visualization.

```python
robot_visual = visual.add_articulation_scene_graph(
    0,
    0,
    mjcf_path,
    path="/robot",
    order="DFS",
    material=robot_material,
)
```

## Control example

Run the complete control example with an MJCF file:

```bash
python ./python/examples/mjcf_dof_control.py /path/to/robot.xml
```

<details>
<summary>Complete source: <code>mjcf_dof_control.py</code></summary>

```{literalinclude} ../../../../python/examples/mjcf_dof_control.py
:language: python
:linenos:
```

</details>

| MJCF articulation | Collision debug |
|---|---|
| ![Loaded MJCF articulation](../images/simulation/mjcf_articulation.png) | ![MJCF collision debug geometry](../images/simulation/mjcf_articulation_collision.png) |

## USD import requirements

Prepare a USD robot with:

- Z-up, `metersPerUnit = 1`, and `kilogramsPerUnit = 1`.
- One connected body tree with explicit positive mass and inertia.
- `convexHull` collision meshes and no joint connecting the root to the world.

Configure initial states, drives, simulation settings, and materials after
loading. These scene settings are not fully reproduced by the importer.
Unsupported joint or collider configurations raise errors; inspect warnings
for omitted properties:

```python
result: ke.asset.USDArticulationImportResult = ke.asset.USDLoader.parse_articulation(
    usd_path="robot.usd", prim_path="/Robot"
)
print(result.diagnostics.warnings)
```
