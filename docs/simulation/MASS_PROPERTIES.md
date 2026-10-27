# Change Mass, Inertia, and Center of Mass

Use `KangSimWorld` to read or change the physical properties of selected
instances. Changes affect their PhysX actors and state metadata; shared asset
descriptions and articulation templates remain unchanged.

| Property | Getter / setter | CPU tensor shape | Units / frame |
| --- | --- | --- | --- |
| Mass | `get_body_masses` / `set_body_masses` | `(N, B)` | kg |
| Principal inertia | `get_body_mass_space_inertias` / `set_body_mass_space_inertias` | `(N, B, 3)` | kg·m², diagonal in the COM frame |
| COM pose | `get_body_local_com_poses` / `set_body_local_com_poses` | `(N, B, 7)` | actor/link-local position XYZ in meters, rotation XYZW |

`N` is the number of selected environments. Getters return all physical bodies;
setters use the selected body count for `B`. Physical bodies are articulation
links, or **one actor for a dynamic/kinematic
rigid**, including a compound rigid with multiple collision shapes. Static
actors do not have these properties.

Argument order is environment selection -> object selection -> values to set:
getters use `(env_ids=None, obj_id=0)`, and setters use
`(env_ids, obj_id, values, *, body_ids=None)`, with `masses`, `inertias`, or
`poses` as the value parameter. For example, `world.get_body_masses(1, 0)`
reads environment 1, object 0. Use `world.get_body_masses(obj_id=1)` to read
object 1 across all its registered environments.

Getters return independent CPU tensors in the requested environment order.
Omitting `env_ids` reads all registered instances for that object slot.
Setters accept CPU tensors, NumPy arrays, or lists with the shapes above.
They do not broadcast values across environments. `body_ids=None` writes all
physical bodies; otherwise each row follows the supplied body index order.

## Scale mass and inertia together

```{note}
For GPU simulation, property changes reach the GPU at the next actual physics
step. Advance one step before submitting forces that must use the new values;
`refresh()` and `step(substeps=0)` do not apply them on the GPU.
```

This example changes the density of one instance without changing its geometry.
Always retain nominal values when applying repeated randomized scales.

```python
import torch
import kangengine as ke

builder = ke.physics.ArticulationBuilder()
builder.add_link("base", mass=2.0, inertia_diagonal=(0.2, 0.2, 0.2))
data: ke.asset.ArticulationDesc = builder.build()

world: ke.sim.KangSimWorld = ke.sim.KangSimWorld(num_envs=2, state_device="cpu")
try:
    cloner: ke.sim.GridCloner = ke.sim.GridCloner(world=world, spacing=3.0)
    robots: ke.sim.SimArticulationBatch = cloner.add_articulation(
        data, obj_id=0, config=ke.physics.ArticulationConfig.free_base()
    )
    nominal_mass = world.get_body_masses(env_ids=None, obj_id=0)
    nominal_inertia = world.get_body_mass_space_inertias(env_ids=None, obj_id=0)
    nominal_com = world.get_body_local_com_poses(env_ids=None, obj_id=0)

    env_ids = [1]
    scale = torch.tensor([[1.2]])
    world.set_body_masses(
        env_ids=env_ids, obj_id=0, masses=nominal_mass[env_ids] * scale
    )
    world.set_body_mass_space_inertias(
        env_ids=env_ids, obj_id=0,
        inertias=nominal_inertia[env_ids] * scale[:, :, None],
    )

    poses = nominal_com[env_ids].clone()
    poses[:, :, 0] += 0.01  # Move COM 1 cm along the body's local X axis.
    world.set_body_local_com_poses(env_ids=env_ids, obj_id=0, poses=poses)

    print(world.get_body_masses(env_ids=None, obj_id=0))  # [[2.0], [2.4]]
    world.step()
finally:
    world.release()
```

For a particular articulation link, pass `body_ids=[robots.get_body_id("base")]`
and provide only that link's values. Rigid actors use `body_ids=[0]`, regardless
of how many collision shapes are attached to the actor.

## Apply changes at a supported boundary

CPU and GPU simulation support changes before stepping or between completed
steps, including after `init_gpu_system()`. Property values and indices must be
on CPU; these APIs do not silently download CUDA tensors.

On GPU, host getters report the new values immediately, while PhysX uploads them
at the next actual physics step. `refresh()` and `step(substeps=0)` do not upload
these changes. GPU dynamics queries and Direct GPU force/torque writes before
that step still use the previous GPU properties. In particular, direct rigid
force/torque writes convert to acceleration when submitted: advance one step
after a property change before submitting forces that must use the new values.
The setters themselves do not advance time or reset poses, velocities, or controls.

Changing mass does not change inertia automatically. For density scaling,
multiply both nominal mass and nominal inertia by the same scale. COM position
changes preserve the actor's world pose. Keep the existing COM quaternion when
only moving its position; changing that quaternion also changes the orientation
of the principal inertia axes. These setters do not wake sleeping bodies or
compensate velocities for a shifted COM frame.

Mass and all three inertia entries must be finite and positive. COM positions
must be finite and COM quaternions must be unit length. Validation rejects
invalid input rather than normalizing it. If an operation fails, earlier
environment writes are not rolled back.

The world setters refresh mass metadata automatically. The existing
`world.state.get_obj_body_masses(obj_id)` still reads environment 0 in the
physical body layout (one body per rigid, one per articulation link); use
`world.get_body_masses(...)` for per-environment physical masses.

For low-level, unregistered physics objects, `Articulation` provides
`get_link_masses`, `get_link_mass_space_inertias`, `get_link_local_com_poses`
and corresponding `set_link_*` methods. `RigidDynamic` provides `get_mass`,
`get_mass_space_inertia`, `get_local_com_pose` and corresponding setters.
Low-level setters do not refresh registered world metadata; prefer world APIs
for registered objects.
