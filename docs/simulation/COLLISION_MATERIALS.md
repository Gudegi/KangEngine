# Set Collision Materials

Collision materials control friction and restitution. They are independent of
render materials such as `ke.material.PBRMaterial`.

`ke.physics.PhysicsMaterialDesc` contains `static_friction`, `dynamic_friction`,
`restitution`, `friction_combine_mode`, and `restitution_combine_mode`.
Friction is non-negative; restitution is between zero and one. All three
coefficients must be finite. The defaults are `(1, 1, 0)`, with `MIN` friction
and `AVERAGE` restitution.

## Change selected shapes

This example changes one shape of one instance. The other instance keeps its
original material, and the shared description is unchanged.

```python
import kangengine as ke

source: ke.asset.RigidDesc = ke.asset.RigidDesc(
    name="box",
    collision_geoms=[ke.asset.CollisionGeomDesc(
        type=ke.asset.CollisionGeomDescType.BOX,
        size=(0.2, 0.2, 0.2), position=(0, 0, 0), quat_xyzw=(0, 0, 0, 1),
    )],
)
world: ke.sim.KangSimWorld = ke.sim.KangSimWorld(num_envs=2, state_device="cpu")
try:
    cloner: ke.sim.GridCloner = ke.sim.GridCloner(world, spacing=3.0)
    boxes: ke.sim.SimRigidBatch = cloner.add_rigid(source, obj_id=0)
    material: ke.physics.PhysicsMaterialDesc = ke.physics.PhysicsMaterialDesc(
        static_friction=0.8, dynamic_friction=0.6, restitution=0.1,
        friction_combine_mode=ke.physics.MaterialCombineMode.MULTIPLY,
        restitution_combine_mode=ke.physics.MaterialCombineMode.MULTIPLY,
    )
    override: ke.physics.CollisionMaterialOverride = ke.physics.CollisionMaterialOverride.for_indices(
        body_index=0, geom_index=0, material=material,
    )
    boxes.records[0].set_collision_material_overrides([override])
    print(boxes.records[0].rigid.get_collision_materials()[0])
finally:
    world.release()
```

For a rigid description, `body_index=0` selects its single actor and
`geom_index` selects an entry in `collision_geoms`. Dynamic, kinematic, and
static rigid instances support these material setters.

Batch setters support all three modes as well. To try static bodies, replace
the `boxes = cloner.add_rigid(...)` creation line above with
`boxes = cloner.add_rigid(source, obj_id=0, static=True)`.
Before the `finally` block, call
`boxes.set_collision_material(env_ids=[1], material=material)` or
`boxes.set_collision_material_overrides(env_ids=[1], material_overrides=[override])`.
`env_ids=None` selects every environment in that batch view. Both setters return
the total number of updated shapes across the selected environments.

Articulations use the same simulation interface:
`record.set_collision_material(material)` and
`record.set_collision_material_overrides(material_overrides=overrides)`.
`SimArticulationBatch` also accepts `env_ids`, just like `SimRigidBatch`.
For articulations, `body_index` selects a link, and `geom_index` selects a collision geometry
within that link. `CollisionMaterialOverride.for_body(body_name, material)`
selects all shapes on a named link; `for_geom(body_name, geom_name, material)`
selects a named shape.

Runtime overrides change **only matching shapes**. Unmatched shapes keep their
current materials; an empty list changes nothing. Later entries win when they
match the same shape. To replace every shape, use `set_collision_material`.
The setters return the number of updated shapes. At object creation, unmatched
shapes instead take their authored material from the description.

`rigid.get_collision_materials()` returns a list in shape order.
`articulation.get_collision_materials()` returns nested lists indexed by link,
then shape. These are independent descriptions: editing the returned values
does not change the simulation until a setter is called.

`PhysicsMaterialDesc.as_tuple()` returns only the three coefficients, without
the combine modes.

## Choose how contacting materials combine

Both contacting shapes contribute coefficients. The combine modes are:

| Mode | Combined coefficient for values `a`, `b` |
| --- | --- |
| `AVERAGE` | `(a + b) / 2` |
| `MIN` | `min(a, b)` |
| `MULTIPLY` | `a * b` |
| `MAX` | `max(a, b)` |

When the shapes specify different modes, PhysX uses the higher-priority mode:
`AVERAGE < MIN < MULTIPLY < MAX`. Friction and restitution resolve separately.
Set the intended mode on both the object and terrain when matching contact
behavior across scenes.

`PhysicsConfig.friction_combine_mode` and `restitution_combine_mode` configure
the world's default material, including `add_default_ground()`. Geometry APIs
that take an explicit `material` use that description's modes. Their default
`PhysicsMaterialDesc()` retains `MIN` / `AVERAGE`; it does not inherit a custom
world mode.

## Reuse materials for randomization

Worlds cache native materials by their three coefficients and both combine
modes. Coefficient cache keys have approximately `1e-4` resolution. A setter
reassigns the material of each exclusive shape instead of modifying a shared
material. Sample a finite candidate set once and repeatedly assign candidates
to keep the cache bounded. `world.physics.num_cached_materials()` reports cached
materials excluding the world's default material.

CPU and GPU material updates can run between completed simulation steps,
including after GPU initialization. GPU contact solving uses the new materials
from the next actual physics step; `step(substeps=0)` does not upload them.
These host setters do not advance simulation time or wake sleeping bodies.
