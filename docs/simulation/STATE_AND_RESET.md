# State and Reset

Reset selected objects or change their velocities between simulation steps.
This guide explains coordinate frames, application order, and CPU/CUDA inputs.

Rigid and articulation records and batches share root reset and velocity APIs.
A world call also selects `obj_id`; a record or batch already identifies its
object. `env_ids=None` selects the environments in that view.

## Reset selected poses

The examples use an existing CPU `world` with `obj_id=0` registered in multiple
environments, as in [Rigid Bodies](RIGID_BODIES.md#place-a-body-in-each-environment).
Select its batch and reset environment 1 to a world-space pose:

```python
import kangengine as ke
import torch

rigids: ke.sim.SimRigidBatch = world.get_rigid_batch(obj_id=0)
position = torch.tensor([[0.0, 0.0, 0.8]])
quat_xyzw = torch.tensor([[0.0, 0.0, 0.38268343, 0.92387953]])  # 45° yaw.
rigids.set_root_state(env_ids=[1], pos=position, quat_xyzw=quat_xyzw, immediate=True)
```

This changes environment 1 only. `env_ids=None` selects every environment in
the view. Position and optional `linear_velocity`/`angular_velocity` inputs
have shape `[N, 3]`; rotations have shape `[N, 4]`. Linear velocities use m/s
and angular velocities use rad/s.

A queued reset (`immediate=False`, the default) takes effect when the world
applies resets or steps. Physics and surface queries then use the applied pose;
call `viewer.sync()` before rendering it. See
[Simulation Visualization](VISUALIZATION.md) for viewer setup.

### CPU and CUDA inputs

Here, CPU IDs means a Python integer/list or a CPU integer tensor.
`None` selects all environments in the view. Reset rows apply to dynamic and
kinematic bodies; static bodies cannot be reset through this API.

| Simulation | Reset position, rotation, and supplied velocities | `env_ids` | `immediate` |
| --- | --- | --- | --- |
| CPU | Lists, NumPy arrays, or CPU tensors | CPU IDs or `None` | `False` or `True` |
| CUDA, after `init_gpu_system()` | CPU inputs, uploaded when applied | CPU IDs or `None` | `False` |
| CUDA, after `init_gpu_system()` | CUDA float32 tensors on the simulation device | CPU IDs, CUDA int32/int64 tensor, or `None` | `False` |

Keep all supplied reset tensors on the same device. For CUDA tensor resets,
register that `obj_id` in every world environment before initializing the GPU
runtime; `env_ids` may then select a subset belonging to the view.

Pose resets reposition kinematic scenery between episodes. Continuous moving
platforms need kinematic-target control, which this public API does not
currently expose; repeated pose resets do not provide that motion behavior.

## Change velocity without resetting

Use `set_root_velocity()` between simulation steps to change motion while
preserving pose, joint state, forces, and control commands. The same method is
available on rigid and articulation records and batches, and on the world:

```python
import torch

world.set_root_velocity(
    env_ids=[1], obj_id=0,
    linear_velocity=torch.tensor([[0.5, 0.0, 0.0]], device=world.sim_device),
    angular_velocity=torch.zeros((1, 3), device=world.sim_device),
)
```

Both velocities are required: world linear velocity in m/s and world angular
velocity in rad/s. Each accepts `[N, 3]`, or `[3]` shared across selected
environments. Values replace the current velocities; add a sampled delta to the
current velocity first to produce a velocity disturbance. This controls dynamic
rigids and floating-base articulations, not static/kinematic actors or fixed bases.

CPU simulation accepts lists, NumPy arrays, or CPU tensors. CUDA simulation
requires float32 tensors on the simulation device after `init_gpu_system()`.
Environment selection accepts CPU IDs or `None`; CUDA simulation also accepts
one-dimensional CUDA int32/int64 IDs when the object is registered in every
environment. CUDA inputs stay on device, independently of `state_device`.

The write applies immediately without advancing time. It does not flush queued
resets: call `world.apply_resets()` first to apply a velocity after a reset.
Use `world.refresh()` before reading updated GPU snapshots, or fetch canonical
GPU state directly. On a batch, call
`rigids.set_root_velocity(env_ids, linear_velocity, angular_velocity)`; omit the
object selector because the batch already identifies it.

## Read current rigid poses

For the rigid batch above, `positions, quat_xyzw = rigids.get_root_pose()`
returns world positions `[N, 3]` and XYZW rotations `[N, 4]`, including static
actors. `env_ids` accepts CPU IDs or `None`, in the requested row order.
Apply queued resets before querying.

With CPU physics, poses use `state_device` (CPU by default).
CUDA dynamic/kinematic queries return poses on the simulation GPU, regardless
of `state_device`; static-only queries use `state_device`.
Ordinary `world.state` snapshots are separate: refresh them explicitly when
needed, or fetch canonical GPU state directly as described in
[PhysX GPU Simulation](../advanced/PHYSX_GPU.md#reading-simulation-state).

For collider poses use [Collision Shapes](COLLISION_SHAPES.md); for rays and
heights use [rigid surface queries](PROCEDURAL_TERRAIN.md#query-the-authored-rigid-surface).
