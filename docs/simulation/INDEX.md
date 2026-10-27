# Simulation

`ke.sim.KangSimWorld` is the recommended Python entry point for rigid bodies,
articulations, batched environments, commands, and public state.

- [Rigid Bodies](RIGID_BODIES.md)
- [State and Reset](STATE_AND_RESET.md)
- [Collision Shapes](COLLISION_SHAPES.md)
- [Fixed Timestep and Rendering](FIXED_TIMESTEP.md)
- [Articulations](ARTICULATION.md)
- [Procedural Articulation Builder](ARTICULATION_BUILDER.md)
- [D6 Joints](D6_JOINTS.md)
- [Control DOFs](DOF_CONTROL.md)
- [Multi-Environment Simulation](MULTI_ENV.md)
- [Mass, Inertia, and Center of Mass](MASS_PROPERTIES.md)
- [Collision Materials](COLLISION_MATERIALS.md)
- [Procedural Terrain and Collision](PROCEDURAL_TERRAIN.md)
- [Contact Sensors](CONTACT_SENSORS.md)
- [Simulation Visualization](VISUALIZATION.md)
- [Simulation Run Modes](RUN_MODES.md)

Start with [First Simulation](../getting_started/FIRST_SIMULATION.md) if you
have not run a KangEngine simulation yet.

## Object identifiers

`env_id` selects an environment within the world. `obj_id` selects an object
slot within that environment. Rigid bodies and articulations share these slots,
so each `(env_id, obj_id)` pair must be unique.

Use the same `obj_id` across environments to group objects into a batch.
It identifies the slot, not the asset type. `body_id` selects a body within that
object, such as an articulation link.
