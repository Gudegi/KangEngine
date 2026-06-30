"""Validate the PhysX GPU rigid mirror against expected warm-up state.

Run this on Linux with an NVIDIA GPU after building the Python extension. A
successful run verifies CUDA context compatibility, PhysX Direct GPU API rigid
fetch, the zero-copy Torch view, and the engine's ``[N, 13]`` rigid-state
layout.
"""

from __future__ import annotations
import gc

from kangengine import physics
from kangengine.utils import to_gpu_array_view
import math
import platform

import numpy as np
import torch



RIGID_COUNT = 8
VALIDATION_STEPS = 4
RIGID_LAYOUT = {
    "position": slice(0, 3),
    "rotation_xyzw": slice(3, 7),
    "linear_velocity": slice(7, 10),
    "angular_velocity": slice(10, 13),
}


def _print_python_environment():
    print("Environment")
    print(f"  platform     : {platform.platform()}")
    print("  torch        : import deferred until after PhysX GPU init")
    print()


def _print_cuda_environment():
    print("CUDA")
    print(f"  cuda devices : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  device 0     : {torch.cuda.get_device_name(0)}")
    print()


def _z_quat(angle: float) -> list[float]:
    half = angle * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _make_rigid_state(i: int):
    x = float(i - RIGID_COUNT // 2) * 1.25
    pos = [x, 0.35 * float(i % 3), 2.0 + 0.15 * float(i)]
    rot = _z_quat(0.07 * float(i))
    linear_velocity = [0.04 * float(i + 1), -0.03 * float(i % 2), 0.02]
    angular_velocity = [0.0, 0.0, 0.0]
    return pos, rot, linear_velocity, angular_velocity


def _expected_after_steps(pos, rot, linear_velocity, angular_velocity, dt, steps):
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    step_count = np.float32(steps)
    velocity = np.asarray(linear_velocity, dtype=np.float32) + gravity * dt * step_count
    position = (
        np.asarray(pos, dtype=np.float32)
        + np.asarray(linear_velocity, dtype=np.float32) * dt * step_count
        + gravity * (dt * dt) * step_count * (step_count + np.float32(1.0)) * np.float32(0.5)
    )
    return np.array(
        [
            *position,
            *rot,
            *velocity,
            *angular_velocity,
        ],
        dtype=np.float32,
    )


def _assert_cuda_float_view(view, *, shape, name, stream_handle):
    if view.name != name:
        raise AssertionError(f"expected view name {name!r}, got {view.name!r}")
    if not view.is_cuda:
        raise AssertionError(f"{name} is not a CUDA view")
    if view.is_cpu:
        raise AssertionError(f"{name} unexpectedly reports CPU memory")
    if view.ptr == 0:
        raise AssertionError(f"{name} has a null pointer")
    if tuple(view.shape) != tuple(shape):
        raise AssertionError(f"expected {name} shape {shape}, got {view.shape}")
    if tuple(view.strides) != (shape[1], 1):
        raise AssertionError(f"expected {name} strides {(shape[1], 1)}, got {view.strides}")
    if view.device_id != 0:
        raise AssertionError(f"expected {name} device_id 0, got {view.device_id}")
    if view.stream_handle != stream_handle:
        raise AssertionError(
            f"expected {name} stream 0x{stream_handle:x}, got 0x{view.stream_handle:x}"
        )
    if view.ready_event_handle == 0:
        raise AssertionError(f"{name} does not expose a ready event")
    if view.numel != shape[0] * shape[1]:
        raise AssertionError(f"unexpected {name} numel {view.numel}")
    if view.byte_size != shape[0] * shape[1] * 4:
        raise AssertionError(f"unexpected {name} byte size {view.byte_size}")

    cuda_iface = view.__cuda_array_interface__
    if cuda_iface["version"] != 3:
        raise AssertionError(f"{name} exposes CUDA array interface version mismatch")
    if tuple(cuda_iface["shape"]) != tuple(shape):
        raise AssertionError(f"{name} CUDA interface shape mismatch")
    if cuda_iface["typestr"] != "<f4":
        raise AssertionError(f"{name} CUDA interface dtype mismatch")
    if int(cuda_iface["data"][0]) != view.ptr:
        raise AssertionError(f"{name} CUDA interface pointer mismatch")
    if tuple(cuda_iface["strides"]) != (shape[1] * 4, 4):
        raise AssertionError(f"{name} CUDA interface byte stride mismatch")
    if int(cuda_iface.get("stream", 0)) != stream_handle:
        raise AssertionError(f"{name} CUDA interface stream mismatch")

    tensor = torch.as_tensor(view, device="cuda:0")
    if tensor.device.type != "cuda":
        raise AssertionError(f"{name} did not produce a CUDA tensor")
    if tensor.data_ptr() != view.ptr:
        raise AssertionError(f"Torch did not create a zero-copy tensor for {name}")
    if tuple(tensor.shape) != tuple(shape):
        raise AssertionError(f"{name} tensor shape mismatch")
    if tuple(tensor.stride()) != (shape[1], 1):
        raise AssertionError(f"{name} tensor stride mismatch")
    return tensor


def _sorted_by_x(state: np.ndarray) -> np.ndarray:
    return state[np.argsort(state[:, 0])]


def _validate_state(gpu_state: np.ndarray, expected_state: np.ndarray, step_index: int):
    gpu_state = _sorted_by_x(gpu_state)
    expected_state = _sorted_by_x(expected_state)

    for label, field_slice in RIGID_LAYOUT.items():
        np.testing.assert_allclose(
            gpu_state[:, field_slice],
            expected_state[:, field_slice],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"step {step_index} rigid {label} mismatch",
        )

    np.testing.assert_allclose(
        gpu_state,
        expected_state,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"step {step_index} rigid state mismatch",
    )
    return gpu_state


def _print_step_sample(step_index: int, state: np.ndarray):
    first = state[0]
    last = state[-1]
    print(
        f"  step {step_index:02d} first pos="
        f"[{first[0]: .5f}, {first[1]: .5f}, {first[2]: .5f}] "
        f"vel=[{first[7]: .5f}, {first[8]: .5f}, {first[9]: .5f}]"
    )
    print(
        f"          last  pos="
        f"[{last[0]: .5f}, {last[1]: .5f}, {last[2]: .5f}] "
        f"vel=[{last[7]: .5f}, {last[8]: .5f}, {last[9]: .5f}]"
    )


def _assert_sparse_root_state_apply(world, gpu_system, rigid_state, rigid_view, config):
    target_row = 3
    target_pos = torch.tensor([20.0, -7.0, 10.0], device="cuda:0")
    target_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0")
    target_lin_vel = torch.zeros(3, device="cuda:0")
    target_ang_vel = torch.zeros(3, device="cuda:0")

    rigid_state[target_row, RIGID_LAYOUT["position"]] = target_pos
    rigid_state[target_row, RIGID_LAYOUT["rotation_xyzw"]] = target_rot
    rigid_state[target_row, RIGID_LAYOUT["linear_velocity"]] = target_lin_vel
    rigid_state[target_row, RIGID_LAYOUT["angular_velocity"]] = target_ang_vel

    index_tensor = torch.tensor([target_row], device="cuda:0", dtype=torch.int32)
    sparse_data_indices = to_gpu_array_view(
        index_tensor,
        dtype=torch.int32,
        name="physics_rigid_sparse_data_indices",
    )
    previous_version = rigid_view.version
    gpu_system.apply_rigid_data(sparse_data_indices)
    torch.cuda.synchronize(0)
    if rigid_view.version != previous_version + 1:
        raise AssertionError(
            "expected rigid data version to advance after sparse apply, "
            f"got {rigid_view.version}"
        )

    world.step()
    gpu_system.fetch_rigid_data()
    torch.cuda.synchronize(0)

    gravity_z = -9.81
    expected_z = target_pos[2] + gravity_z * config.dt * config.dt
    selected = rigid_state[target_row].detach().cpu().numpy()
    np.testing.assert_allclose(
        selected[RIGID_LAYOUT["position"]],
        np.array([target_pos[0].item(), target_pos[1].item(), expected_z.item()]),
        rtol=1e-5,
        atol=1e-5,
        err_msg="sparse rigid root-state apply selected row mismatch",
    )
    for row in range(RIGID_COUNT):
        if row == target_row:
            continue
        if torch.allclose(rigid_state[row, :3], target_pos, atol=1.0):
            raise AssertionError(
                f"sparse rigid root-state apply unexpectedly moved row {row}"
            )


def main():
    _print_python_environment()

    config = physics.PhysicsConfig.z_up()
    config.enable_gpu = True
    config.enable_contact_reports = False
    world = physics.PhysicsWorld(config)

    initial_rows = []
    actors = []
    for i in range(RIGID_COUNT):
        pos, rot, linear_velocity, angular_velocity = _make_rigid_state(i)
        actor = world.create_dynamic_box([0.18, 0.16, 0.14], pos, rot)
        actor.set_root_state(pos, rot, linear_velocity, angular_velocity)
        initial_rows.append((pos, rot, linear_velocity, angular_velocity))
        actors.append(actor)

    gpu_config = physics.GpuPhysicsConfig()
    gpu_config.cuda_device_id = 0
    gpu_system = physics.PhysicsGpuSystem(world, gpu_config)
    gpu_system.init()
    rigid_rows = [gpu_system.rigid_row(actor) for actor in actors]
    if sorted(rigid_rows) != list(range(RIGID_COUNT)):
        raise AssertionError(f"unexpected rigid row mapping {rigid_rows}")

    stream = torch.cuda.current_stream(device=0)
    stream_handle = int(stream.cuda_stream)
    gpu_system.set_cuda_stream(stream_handle)

    rigid_view = gpu_system.rigid_data()
    rigid_state = _assert_cuda_float_view(
        rigid_view,
        shape=(RIGID_COUNT, 13),
        name="physics_rigid_data",
        stream_handle=stream_handle,
    )

    force_view = gpu_system.rigid_force()
    torque_view = gpu_system.rigid_torque()
    rigid_force = _assert_cuda_float_view(
        force_view,
        shape=(RIGID_COUNT, 3),
        name="physics_rigid_force",
        stream_handle=stream_handle,
    )
    rigid_torque = _assert_cuda_float_view(
        torque_view,
        shape=(RIGID_COUNT, 3),
        name="physics_rigid_torque",
        stream_handle=stream_handle,
    )

    torch.cuda.synchronize(0)
    if not torch.allclose(rigid_force, torch.zeros_like(rigid_force)):
        raise AssertionError("rigid force mirror should start at zero")
    if not torch.allclose(rigid_torque, torch.zeros_like(rigid_torque)):
        raise AssertionError("rigid torque mirror should start at zero")
    rigid_force.zero_()
    rigid_torque.zero_()
    gpu_system.apply_rigid_force()
    gpu_system.apply_rigid_torque()
    torch.cuda.synchronize(0)
    if force_view.version != 1:
        raise AssertionError(
            f"expected rigid force version 1 after apply, got {force_view.version}"
        )
    if torque_view.version != 1:
        raise AssertionError(
            f"expected rigid torque version 1 after apply, got {torque_view.version}"
        )
    sparse_indices = torch.tensor(
        [0, RIGID_COUNT - 1], device="cuda:0", dtype=torch.int32
    )
    sparse_index_view = to_gpu_array_view(
        sparse_indices,
        dtype=torch.int32,
        name="physics_rigid_sparse_indices",
    )
    gpu_system.apply_rigid_force(sparse_index_view)
    gpu_system.apply_rigid_torque(sparse_index_view)
    torch.cuda.synchronize(0)
    if force_view.version != 2:
        raise AssertionError(
            f"expected rigid force version 2 after sparse apply, got {force_view.version}"
        )
    if torque_view.version != 2:
        raise AssertionError(
            f"expected rigid torque version 2 after sparse apply, got {torque_view.version}"
        )

    print("Rigid mirror")
    print(f"  count        : {RIGID_COUNT}")
    print(f"  pointer      : 0x{rigid_view.ptr:x}")
    print(f"  shape        : {rigid_view.shape}")
    print(f"  strides      : {rigid_view.strides}")
    print(f"  version      : {rigid_view.version}")
    print(f"  stream       : 0x{rigid_view.stream_handle:x}")
    print(f"  ready event  : 0x{rigid_view.ready_event_handle:x}")
    print("Rigid command mirrors")
    print(f"  force ptr    : 0x{force_view.ptr:x}")
    print(f"  force ver    : {force_view.version}")
    print(f"  torque ptr   : 0x{torque_view.ptr:x}")
    print(f"  torque ver   : {torque_view.version}")
    print(f"  sparse idx   : {sparse_index_view.shape}")
    print()
    print("Step samples")

    for step_index in range(1, VALIDATION_STEPS + 1):
        if step_index > 1:
            world.step()
        gpu_system.fetch_rigid_data()
        if rigid_view.version != step_index:
            raise AssertionError(
                f"expected rigid data version {step_index}, got {rigid_view.version}"
            )
        torch.cuda.synchronize(0)

        gpu_state = rigid_state.detach().cpu().numpy()
        expected_state = np.stack(
            [
                _expected_after_steps(
                    pos,
                    rot,
                    linear_velocity,
                    angular_velocity,
                    config.dt,
                    step_index,
                )
                for pos, rot, linear_velocity, angular_velocity in initial_rows
            ]
        )
        sorted_state = _validate_state(gpu_state, expected_state, step_index)
        _print_step_sample(step_index, sorted_state)

    _assert_sparse_root_state_apply(
        world, gpu_system, rigid_state, rigid_view, config
    )

    del rigid_state
    del rigid_force
    del rigid_torque
    del sparse_index_view
    gpu_system.invalidate()
    for actor in actors:
        actor.release()
    del actors
    del gpu_system
    del world
    gc.collect()

    print()
    print(
        f"PASS: PhysX GPU rigid state matches expected state for "
        f"{VALIDATION_STEPS} steps"
    )


if __name__ == "__main__":
    main()
