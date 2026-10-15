# First Scene

This example creates geometry data, inserts it into the scene, and demonstrates
materials, hierarchy, local/world transforms, quaternion input, and debug axes.

```python
import kangengine as ke
import numpy as np


class MyApp(ke.App):
    def setup(self):
        self.standard_materials = self.create_standard_materials()
        self.scene.add_ground(path="/ground", scale=30.0)

        box: ke.RenderablePrimView = self.scene.add_mesh(
            path="/box",
            mesh_data=ke.geometry.create_cube_data(scale=1.0),
            material=self.standard_materials.common,
            color=ke.Vec4(0.8, 0.3, 0.02, 1.0),
        )
        box.set_local_translation(translation=ke.Vec3(0.0, 2.0, 0.0))
        box.set_local_rotation_axis_angle(
            axis=ke.Vec3(0.0, 1.0, 0.0),
            angle_radians=np.deg2rad(25.0),
        )
```

The important separation is:

- `ke.geometry.create_cube_data(...)` creates mesh data only.
- `app.scene.add_mesh(...)` creates a scene prim and renderable view.
- The returned `box` view controls common rendering and transform operations.
- `box.prim` remains available for lower-level scene graph operations.

## Parent and child transforms

An absolute prim path establishes hierarchy. `/box/box2` is a child of
`/box`, so its local transform is relative to the first box.

```python
box2: ke.RenderablePrimView = self.scene.add_mesh(
    path="/box/box2",
    mesh_data=ke.geometry.create_cube_data(scale=1.0),
    material=self.standard_materials.common,
)
box2.set_local_translation(translation=ke.Vec3(0.0, 1.5, 0.0))
box2.set_local_scale(scale=ke.Vec3(0.5, 0.5, 0.5))

local_position = box2.get_local_translation()
world_position = box2.get_world_translation()
```

Use `set_world_translation(...)` when a value is already expressed in scene
coordinates. Otherwise, prefer local transforms so children follow their
parent naturally.

## Quaternion ordering

`ke.Quat` and `set_local_quat_wxyz()` / `set_world_quat_wxyz()` inputs use WXYZ
ordering. Setters accept a four-element array or a `ke.Quat` object; getters
`get_local_quat()` / `get_world_quat()` return `ke.Quat` objects. Local means
relative to the parent; world means relative to the scene origin.

```python
# Approximately 45 degrees around Z, in wxyz order.
box2.set_local_quat_wxyz(
    quat_wxyz=np.array([0.924, 0.0, 0.0, 0.383], dtype=np.float32)
)

rotation: ke.Quat = box2.get_world_quat()
quat_wxyz = rotation.to_wxyz()
quat_xyzw = rotation.to_xyzw()
```

Physics and simulation state arrays use XYZW, for example
`get_root_quat_xyzw()` and `get_body_quat_xyzw()`. Convert at the scene boundary:

```python
box2.set_world_quat_wxyz(quat_wxyz=ke.Quat.from_xyzw(quat_xyzw))
```

For array conversion, use `ke.utils.quat_xyzw_to_wxyz()` and
`ke.utils.quat_wxyz_to_xyzw()` (NumPy), or the same functions in
`kangengine.utils.batched_rotations` (Torch, preserving device and dtype).

## Inspect transform axes

For a lightweight render overlay that does not create a scene prim:

```python
self.debug_overlay.axes(
    "/debug/world_axes",
    origin=np.array([0.0, 1.0, 0.0]),
    rotation=np.eye(3),
    length=1.0,
)
```

To create axes that appear in the SceneGraph, use the scene-backed helper:

```python
self.scene.debug_geometry.add_axes(
    "/debug/box2_axes",
    box2.get_world_translation(),
    box2.get_world_quat(),
    length=0.8,
    radius=0.01,
)
```

`debug_overlay` draws directly through the graphics debug renderer and does not
create scene prims. `scene.debug_geometry` creates mesh-based renderables that
appear in the SceneGraph and returns a `DebugPrimitiveView`.

To run the complete example below, clone the repository and run from its root
(the example scripts are not installed by pip):

```bash
python ./python/examples/render_prim_scene.py
```

<details>
<summary>Complete source: <code>render_prim_scene.py</code></summary>

```{literalinclude} ../../../../python/examples/render_prim_scene.py
:language: python
:linenos:
```

</details>

![Parent and child prims with transform axes](../images/getting_started/first_scene.png)

Next: [First Simulation](FIRST_SIMULATION.md) or
[Scene and Rendering](../scene_rendering/INDEX.md).
