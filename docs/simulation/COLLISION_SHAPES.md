# Collision Shapes

A body is a physical object or articulation link whose motion is simulated.
Collision shapes define the geometry used to detect its contacts. Multiple
shapes attached to one body move together, while each shape retains its own
local pose relative to that body.

For example, a cup is one rigid body. Its main cup section and handle can be
represented by separate collision shapes, with several shapes used for either
part when needed to approximate its geometry. These shapes move together as
one cup; they are not separate simulated bodies.

This page explains how to read shape world poses and map them back to the
authored geometry.

For an offset box collider, the body's position and the collider's center are
different. Use the shape pose when placing a collision-debug overlay.
For creating bodies and their geometry, see [Rigid Bodies](RIGID_BODIES.md) or
[Articulations](ARTICULATION.md).

## Read world poses

The snippet uses an existing `world` with rigid bodies registered under
`obj_id=0`, as in [Rigid Bodies](RIGID_BODIES.md#register-a-body).
Apply queued resets before querying.

```python
import kangengine as ke

rigids: ke.sim.SimRigidBatch = world.get_rigid_batch(obj_id=0)
shape_poses = rigids.get_collision_shape_pose()  # [N, S, 7]

rigid: ke.sim.SimRigid = rigids.first
one_pose = rigid.get_collision_shape_pose()  # [S, 7]
```

For an articulation, obtain the batch with
`world.get_articulation_batch(obj_id=0)`; the shape query interface is the same.

Each pose contains world position XYZ in meters followed by quaternion XYZW.
`N` is the number of selected environments and `S` is `num_collision_shapes`.

| Query | Return shape |
| --- | --- |
| Batch | `[N, S, 7]` |
| Single record, `env_ids` omitted | `[S, 7]` |
| Single record, explicit `env_ids` | `[N, S, 7]` |

By default, batch rows follow the view's environment order. Explicit CPU
`env_ids` select and reorder rows; an empty selection returns `[0, S, 7]`.

Rigid `num_bodies` is always 1. Its body pose, mass, force, and contact state
refer to the actor, regardless of the number of attached shapes. For example,
`rigids.data.body_pos` and `rigids.data.body_quat_xyzw` describe the body pose,
not the shape poses. Articulation body state refers to physical links.

## Map shapes to authored geometry

Using the rigid batch above:

```python
body_ids = rigids.collision_shape_body_ids  # CPU int64 [N, S]
geom_ids = rigids.collision_shape_geom_ids  # CPU int64 [N, S]
shape_names = rigids.collision_shape_names  # list[list[str]], one row per environment
```

These metadata arrays follow the batch view's environment order. Pose queries
with explicit `env_ids` follow the requested order instead; align metadata rows
before combining them with a reordered or subset pose query. For a single
record, mapping arrays have shape `[S]` and names are a `list[str]`.
Returned metadata can be changed without modifying the actor or source description.

Each shape maps to `(body_idx, geom_idx)`, where `geom_idx` indexes that body's
authored `collision_geoms`; `-1` means a generated collider with no authored
geom. Display names use `body_name:geom_name`, or `body_name:geom_<idx>` for an
unnamed geom. Read the mapping arrays instead of parsing names. The current
builders create one shape per geom; automatic convex decomposition is not
provided.

For articulations, `body_idx` indexes the runtime `body_names`, after fixed-frame
compaction. The articulation template's `body_source_indices` maps these bodies
back to the original description's skeleton nodes.
For rigid bodies, `body_idx` is 0 and `body_names` contains `RigidDesc.name`.

## Coordinate frames and devices

Queries combine current actor/link poses with the runtime shape-local frames.
CPU simulation returns tensors on the state device.
GPU dynamic rigid/articulation queries fetch current poses and return
CUDA tensors without downloading them; static-only rigid queries use the state
device. Apply queued resets before querying. Static actors support these pose
queries, but are not part of dynamic body state or contact sensor batches.

```{note}
Capsule poses use the PhysX shape frame, whose local X axis runs along the
capsule. The returned rotation includes the conversion from the authored
capsule axis; do not apply it directly to a Z-axis capsule mesh without
accounting for that axis difference.
```

[Contact sensors](CONTACT_SENSORS.md) select physical bodies; per-shape contact
filtering is not currently available. For older code that treated rigid body
slots as colliders, see the [migration note](../CHANGELOG.md#rigid-body-state).

See [Multi-Environment Simulation](MULTI_ENV.md#different-rigid-sources-per-environment)
for the fixed body/shape counts required within one object batch.
