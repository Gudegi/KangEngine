# PhysX GPU Simulation

This guide describes the Linux/NVIDIA GPU contract for
`ke.sim.KangSimWorld`. Start with [Simulation](../simulation/INDEX.md) for the
normal CPU/GPU workflow. Use this page when working with Torch CUDA tensors,
GPU contact sensors, or CUDA/OpenGL visualization.

Build requirements, including PhysX 5.8 and CUDA 13 compatibility, are in
[Build from Source](BUILD_FROM_SOURCE.md).

## At a glance

The important differences from CPU simulation are:

| Topic | CPU simulation | PhysX GPU simulation |
|---|---|---|
| Canonical state | `world.state` | `world.state.gpu` / `PhysicsGpuSystem` |
| CPU-visible state | Updated normally | Explicit snapshot via `world.refresh()` |
| Tensor device | CPU | CUDA device used by PhysX |
| Adding objects | Before or during setup | Before `world.init_gpu_system()` |
| Cross-device copies | Caller controlled | Rejected |
| Renderer sync | SceneGraph or CPU `ExternalBuffer` | CUDA/OpenGL `ExternalBuffer` interop |

Typical high-throughput frame:

```python
world.step(refresh=False)
world.state.gpu.refresh_frame_cache()

q = world.state.gpu.get_dof_pos(obj_id, fetch=False)
body_pos = world.state.gpu.get_body_pos(obj_id, fetch=False)
```

Use `world.step(refresh=True)` when convenient CPU snapshots are more important
than avoiding GPU readback.

## Initialization and device rules

Call `world.init_gpu_system()` after registering every rigid object and
articulation, and before using `torch.cuda.*`:

```python
world = ke.sim.KangSimWorld(..., sim_device="cuda:0")

# Add all environments and simulation objects first.
world.init_gpu_system()
```

Required rules:

- `world.init_gpu_system()` establishes the PhysX CUDA context and caches GPU
  row mappings.
- Adding simulation objects after initialization is rejected because it would
  invalidate those mappings.
- `init_gpu_system(cuda_device_id=None)` uses the ordinal in `sim_device`;
  `"cuda"` selects device 0.
- PhysX buffers, Torch tensors, and renderer interop buffers must use the same
  CUDA device.
- CUDA tensors accepted by setters must be contiguous `torch.float32` tensors.
- A CUDA tensor passed to a CPU world, or before GPU initialization, is rejected
  instead of being silently downloaded.

## Reading simulation state

### CPU snapshot

`world.state` remains the common CPU/GPU API, but in GPU simulation it is an
explicit CPU/Torch snapshot:

```python
state = world.step(refresh=True)
root_pos = world.state.get_root_pos(obj_id)
```

After a GPU step or reset, the snapshot is stale until `world.refresh()` or
`world.step(refresh=True)` runs. Enable strict checking while debugging:

```python
world.state.set_strict_snapshot_reads(True)
```

### GPU frame cache

For control and training loops, fetch the required mirrors once per frame:

```python
world.step(refresh=False)
world.state.gpu.refresh_frame_cache()

q = world.state.gpu.get_dof_pos(obj_id, fetch=False)
dq = world.state.gpu.get_dof_vel(obj_id, fetch=False)
```

`refresh_frame_cache()` caches zero-copy Torch CUDA views and logical
object-level slices. Calls with `fetch=False` reuse those views. Steps, resets,
and world release invalidate the cache.

<details>
<summary>Low-level GpuArrayView ownership and synchronization</summary>

`GpuArrayView` is a non-owning descriptor containing a pointer, device,
shape, strides, stream, ready event, version, and optional owner token.

- Views from `PhysicsGpuSystem.views()` become invalid after
  `PhysicsGpuSystem.invalidate()`, world destruction, or reinitialization.
- Later fetch, apply, refresh, or synchronization operations may update the
  underlying allocation. Copy explicitly when a stable snapshot is required.
- `GpuArrayView.torch()` creates a zero-copy tensor through
  `__cuda_array_interface__`; it does not extend the allocation lifetime.
- `PhysicsGpuSystem.set_cuda_stream(stream)` selects the stream for PhysX
  fetch/apply kernels. Stream 0 is used by default.
- Consumers must wait for `ready_event_handle` before reading producer data.
- Low-level contact results describe the latest synchronization point and must
  not be assumed stable across `step()`, `sync()`, or `clear_contacts()`.

</details>

## Writing root and DOF state

CPU and GPU worlds use the same public setters:

```python
world.set_root_state(
    env_ids,
    obj_id,
    pos,
    rot_xyzw,
    linear_velocity,
    angular_velocity,
)

rigid.set_root_state(
    env_ids,
    pos,
    rot_xyzw,
    linear_velocity,
    angular_velocity,
)
```

`env_ids=None` selects every environment row for the object. A selected list or
tensor is converted to a cached CUDA logical-row index view.

Useful row helpers:

- `world.rigid_gpu_row(env_id, obj_id)`
- `world.rigid_gpu_index_view(env_ids, obj_id)`
- `world.articulation_gpu_row(env_id, obj_id)`
- `world.articulation_gpu_index_view(env_ids, obj_id)`

Root state layout is:

```text
[position xyz, rotation xyzw, linear velocity xyz, angular velocity xyz]
```

<details>
<summary>How high-level state maps to PhysX GPU buffers</summary>

Rigid root state is lowered as:

```text
env_ids / obj_id
  -> logical rigid rows
  -> PhysicsGpuSystem.rigidData [N, 13]
  -> applyRigidData(indices)
```

`PhysicsGpuSystem.apply_rigid_data(indices)` accepts logical rigid rows, not
PhysX internal GPU indices. `PhysicsGpuSystem.rigid_row(rigid)` resolves the
logical row for a registered low-level actor.

Full GPU mirrors are available through `world.get_gpu_rigid_data()` and
`world.get_gpu_articulation_*()`. Use row/index helpers for environment
selection; CUDA advanced indexing may allocate a copy.

Direct GPU apply updates simulation state only. It does not independently
update SceneGraph prim transforms or viewer state.

</details>

## Articulation control

After GPU initialization, `set_cmd()` accepts contiguous CUDA tensors shaped
either `[num_dofs]` or `[selected_envs, num_dofs]`.

Supported command paths:

- `ControlMode.POS`: target joint-position buffer
- `ControlMode.VEL`: target joint-velocity buffer
- `ControlMode.TORQUE`: joint-force buffer
- `ControlMode.PD_EXPLICIT`: explicit clipped PD forces on the GPU

`PD_EXPLICIT` requires a target tensor and per-DOF `kp` and `kd` arrays or
tensors. Scalar gains are intentionally rejected.

`set_kps()`, `set_kds()`, and `set_effort_limits()` synchronize the stored
per-DOF metadata with the native PhysX articulation drive parameters.

There are two lowerings:

- Full-batch fast path: `env_ids=None`, one articulation object across all
  environments, and `[num_envs, num_dofs]` CUDA commands.
- Sparse path: single-environment, partial-environment, editor, and debugging
  commands.

See `python/examples/sim_gpu_mixed_batch.py` for batched POS and explicit PD
control.

<details>
<summary>Articulation buffer layout and reset behavior</summary>

The link mirror is `[articulation_count, max_links, 13]`. Rows are padded to
the scene-wide `max_links`; call `articulation_link_count(row)` before consuming
one articulation. Each link uses the same position, XYZW rotation, linear
velocity, and angular velocity layout as rigid state.

Fetch and apply support includes link pose/velocity, joint position, velocity,
acceleration, force, target position/velocity, and incoming joint force.
Root pose/velocity apply uses the root link row followed by
`update_articulation_kinematics()`.

GPU reset clears rigid force/torque commands and articulation command state.
Articulations return to `ControlMode.NONE`; target position becomes the current
reset position, while target velocity and force become zero.

Reset also clears contact mirrors and packed contact-sensor outputs. A
reset-only frame (`world.step(substeps=0, ...)`) reports zero contacts. To read
new impact impulses, apply the reset first and then simulate one step.

</details>

## Contacts and sensors

Create sensors from rigid or articulation views:

```python
sensor = rigid.add_contact_sensor(body_ids=body_ids, name="feet")
```

After each step, `ContactSensor` provides CUDA tensors:

- `contact_count[env, body]`
- `in_contact[env, body]`
- `net_impulse[env, body, xyz]`
- `net_force`, computed as `net_impulse / world.sim_dt`

`add_force_sensor(...)` returns a `ForceSensor` with `force` and `impulse`
aliases. These are normal-force sensors, not six-axis wrench sensors.

All sensors share one raw PhysX contact fetch per step and use world-owned
packed CUDA storage. Aggregation does not create per-sensor Torch temporaries
or synchronize through the CPU.

<details>
<summary>Raw contact buffers, layouts, and capacity</summary>

Raw access is available through:

- `world.state.gpu.contact_pairs()`
- `world.state.gpu.contact_pair_count()`
- `world.state.gpu.contact_pair_headers()`
- `world.state.gpu.contact_pair_body_refs()`
- `world.state.gpu.contact_points()`
- `world.state.gpu.contact_point_count()`
- `world.state.gpu.contact_point_pair_indices()`

`contact_pairs()` is a raw `uint8` view of PhysX `PxGpuContactPair` records.
Their contact stream pointers are valid only until the next simulation step.

`contact_pair_headers()` is `uint64[max_contact_pairs, 6]`:

```text
node0, node1, actor0_ptr, actor1_ptr, transform_ref0, transform_ref1
```

`contact_pair_body_refs()` is `int32[max_contact_pairs, 6]`, containing two
`[kind, row, body]` groups. `kind` is `0` for rigid, `1` for articulation, and
`-1` for unknown or static bodies.

`contact_points()` is `float32[max_contact_points, 10]`:

```text
position.xyz, normal.xyz, impulse.xyz, separation
```

PhysX's `contactForces` scalar is treated as normal impulse for the completed
step. Net impulse uses `+normal * impulse` for actor 0 and the opposite sign for
actor 1. Tangential friction impulse is not included, so this is not a complete
contact wrench.

GPU contact mirrors have fixed capacity. Defaults are 65,536 pairs and 262,144
points, controlled by `GpuPhysicsConfig.max_contact_pairs` and
`max_contact_points`. Increase them before initialization for dense workloads.

</details>

## GPU visualization

Enable the GPU visualization backend through `SimWorldVisualizer.add(...)`.
For each visual batch it:

1. Fetches the required PhysX GPU mirror.
2. Maps the renderer transform buffers.
3. Runs CUDA gather/conversion kernels.
4. Unmaps once before drawing.

The path does not read transforms through the CPU, recreate
`ExternalBufferDesc` every frame, or allocate an intermediate Torch Mat4
tensor.

<details>
<summary>CUDA/OpenGL and articulation link-order details</summary>

The OpenGL backend maps all link VBOs for an articulation batch in one CUDA
interop transaction. It runs a fused link-major transform kernel and unmaps the
resources before draw; mapped CUDA pointers are never retained afterward.

PhysX Direct GPU link blocks use breadth-first indices from
`PxArticulationLink::getLinkIndex()`. These may differ from MJCF or
`ArticulationVisual` body order. The visual backend therefore applies an
explicit visual-link to PhysX-link permutation. Directly indexing the GPU block
with a visual body id is incorrect for branched articulations.

</details>

## Examples and validation

- `python/examples/sim_gpu_root_state_batch.py`: high-level batched root state
- `python/examples/sim_gpu_mixed_batch.py`: Torch CUDA articulation control
- `python/examples/smoke/physics_gpu_system_smoke.py`: low-level buffer contract

Run the GPU validation suites:

```bash
make validate_physx_gpu
make validate_physx_gpu_cpp
```
