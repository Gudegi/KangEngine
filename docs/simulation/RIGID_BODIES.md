# Rigid Bodies

A rigid body is a physical object that moves without deforming. This guide
shows how to define its geometry, add it to a simulation, and synchronize its
visual representation. You can also reuse the same description to create
bodies across multiple environments.

- To draw an object at supplied poses, [render without physics](#render-without-physics).
- To simulate it, [register one body](#register-a-body) or
  [place bodies in multiple environments](#place-a-body-in-each-environment).
- For a runnable viewer with surface queries, see the
  [interactive stair raycast example](PROCEDURAL_TERRAIN.md#try-the-interactive-stair-raycast-example).

The basic example below needs no asset files. Later snippets name the objects
they reuse; rendering snippets require an initialized `app`, as described in
[Hello App](../getting_started/HELLO_APP.md).

```{note}
Simulation snippets use CPU PhysX with Z-up unless stated otherwise.
When attaching simulated visuals, initialize
the App with `up_axis=ke.UpAxis.Z` instead of the Y-up setting in Hello App.
```

## Prepare a rigid description

Use `ke.asset.RigidDesc` to prepare visual geometry, collision geometry, and
optional mass properties. One description can supply multiple visual or
simulated instances with independent poses.

Create a 1.0 × 0.6 × 0.8 meter box with separate visual and collision geometry:

```python
import kangengine as ke

mesh: ke.scene.MeshData = ke.geometry.create_box_data(
    x_scale=1.0, y_scale=0.6, z_scale=0.8
)
material: ke.material.PBRMaterial = ke.material.PBRMaterial()
material.base_color = ke.Vec4(0.8, 0.4, 0.2, 1.0)
material.roughness = 0.45

box_collider: ke.asset.CollisionGeomDesc = ke.asset.CollisionGeomDesc(
    type=ke.asset.CollisionGeomDescType.BOX,
    size=[0.5, 0.3, 0.4],  # Half extents in meters.
    position=[0.0, 0.0, 0.0],
    quat_xyzw=[0.0, 0.0, 0.0, 1.0],
)
box_asset: ke.asset.RigidDesc = ke.asset.RigidDesc(
    visual_geoms=[ke.asset.VisualGeomDesc(mesh_data=mesh, material=material)],
    collision_geoms=[box_collider],
)
```

`visual_geoms` holds `VisualGeomDesc` parts, each with its own mesh, local
position, rotation, scale, and material. `collision_geoms` independently
defines the physics shapes in the same body-local frame. They need not have the
same detail or number of parts. Articulations use the same visual part and
material types.

Without an `inertial` description, the instance density and its colliders
determine mass properties. Supply `ke.asset.InertialDesc` for authored mass,
center of mass, and diagonal inertia.

```{warning}
Finish editing descriptions, meshes, and source materials before creating
instances: later changes do not automatically update existing visuals or
collision/query caches. For changed geometry, create new mesh data and a new
description, then recreate the instances.
```

For imported geometry, see [Load Assets](../scene_rendering/LOAD_ASSETS.md#rigid-body-sources);
for textures and appearance, see
[asset source materials](../scene_rendering/MATERIALS_TEXTURES.md#asset-source-materials).

## Render without physics

Using `box_asset` from the preparation example and an initialized `app`:

```python
import kangengine as ke

box_visual: ke.visual.RigidVisual = ke.visual.RigidVisual(
    app=app, path="/props/box", data=box_asset
)
box_visual.apply_pose(position=(1.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0))
```

This creates scene objects without a physics world. `apply_pose()` takes a
world position `[3]` in meters and a unit XYZW quaternion `[4]`, including when
the visual has a transformed parent. Parts retain their authored local poses,
scales, and materials. Multiple visuals can share `box_asset` and use different
poses. Remove them with `box_visual.remove()` when finished.

## Register a body

For physics, use the same `box_asset` prepared above:

```python
import kangengine as ke

world: ke.sim.KangSimWorld = ke.sim.KangSimWorld(
    num_envs=1, sim_dt=1.0 / 120.0, add_ground=True
)
box: ke.sim.SimRigid = world.add_rigid(
    box_asset,
    env_id=0,
    obj_id=0,
    name="box",
    pos=[0.0, 0.0, 1.8],
    density=600.0,
)
world.step()
```

This registers and steps the physics body. To display it, attach a visualizer
as shown in [Attach and synchronize simulated visuals](#attach-and-synchronize-simulated-visuals).

`env_id` selects an environment. `obj_id` is an object slot within it, shared
by rigid bodies and articulations; each `(env_id, obj_id)` must be unused before
registration. Across environments, the same `obj_id` groups instances into a
batch. It does not identify a mesh or asset type. See
[Object identifiers](INDEX.md#object-identifiers) for the shared terminology.

`world.add_rigid()` creates a dynamic body by default. Use `kinematic=True`
for a body unaffected by gravity and contact forces, or
`world.add_static_rigid()` for fixed geometry. Reposition kinematic bodies
through pose resets; see [State and Reset](STATE_AND_RESET.md) for the
distinction from continuous kinematic motion.

The returned `box` supports the same pose reset API as the batch below;
`env_ids=None` selects its single environment. For a complete simulation loop,
see [First Simulation](../getting_started/FIRST_SIMULATION.md).

## Attach and synchronize simulated visuals

Using the single body's `world` and `box`, plus an initialized Z-up `app`:

```python
import kangengine as ke

viewer: ke.visual.sim.SimWorldVisualizer = ke.visual.sim.SimWorldVisualizer(
    app=app, world=world
)
rendered: ke.visual.sim.VisualBatch = viewer.add(sim_handle=box, path="/scenery")
viewer.sync()
```

The registered description supplies the visual geometry and materials, so no
`mjcf_path` is needed. Synchronize after simulation or applied resets and before
drawing. For CUDA simulation, initialize the GPU runtime before `viewer.add()`.
See [Simulation Visualization](VISUALIZATION.md) for the shared viewer workflow.

## Place a body in each environment

Reuse `box_asset` to place the same box in three environments. This replaces
the single-body world above; release its viewer first if you attached one:

```python
import kangengine as ke

world.release()
world: ke.sim.KangSimWorld = ke.sim.KangSimWorld(num_envs=3, add_ground=True)
cloner: ke.sim.GridCloner = ke.sim.GridCloner(world, spacing=10.0)
rigids: ke.sim.SimRigidBatch = cloner.add_rigid(
    box_asset, obj_id=0, initial_root_pos=(0.0, 0.0, 1.8), density=600.0
)
```

The source geometry is shared; each environment owns its physical actor and
collision shapes. Initial positions are relative to `cloner.env_origins`,
and environment collision groups isolate bodies from other environments.
For different sources per environment and batch layout constraints, see
[Multi-Environment Simulation](MULTI_ENV.md#different-rigid-sources-per-environment).

To display this batch, create a new `SimWorldVisualizer` for the new `world`
using the setup above, and pass `sim_handle=rigids` to `viewer.add()` instead
of `sim_handle=box`.

## Reset selected poses

Reset positions use world coordinates. Add an environment origin to an
environment-local offset to move only that box:

```python
import torch

position = cloner.env_origins[1:2] + torch.tensor([[0.0, 0.0, 0.8]])
quat_xyzw = torch.tensor([[0.0, 0.0, 0.38268343, 0.92387953]])  # 45° yaw.
rigids.set_root_state(env_ids=[1], pos=position, quat_xyzw=quat_xyzw, immediate=True)
```

This CPU example changes environment 1 immediately. Positions are meters and
rotations are unit XYZW quaternions; `env_ids=None` selects every environment
in the view. With the default `immediate=False`, resets take effect when
`world.apply_resets()` or `world.step()` runs.

GPU resets use `immediate=False`; supported input devices, velocity changes,
and reset ordering are described in [State and Reset](STATE_AND_RESET.md).
To read collider poses rather than body poses, use
`rigids.get_collision_shape_pose()`:
[Collision Shapes](COLLISION_SHAPES.md).

## Clean up

| Objects created | Release when finished |
| --- | --- |
| Standalone `box_visual` | `box_visual.remove()` |
| Simulated visuals through `viewer` | `viewer.release()` |
| Physics `world` | `world.release()` |

Remove visuals and release the viewer before releasing the world and closing
the App. Put these calls in `App.cleanup()` when using `App.start()`; for a
manually stepped application, call `app.close()` last.

## Terminology

| Term | Meaning |
| --- | --- |
| Body | One physical actor for a rigid; one physical link for an articulation. |
| `collision_geom` | Authored collision geometry in `collision_geoms`. |
| Collision shape | A runtime collider attached to a body, with its own local pose and material. |
| `visual_geom` | Authored visual geometry and appearance in `visual_geoms`. |
| Visual instance | Scene prim/render component or an instance in a render batch. Its count need not match collision shapes. |
