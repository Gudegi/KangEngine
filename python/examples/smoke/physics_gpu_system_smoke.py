"""Validate the PhysX GPU rigid/contact mirrors against expected state.

Run this on Linux with an NVIDIA GPU after building the Python extension. A
successful run verifies CUDA context compatibility, PhysX Direct GPU API rigid
fetch, contact fetch/flattening, zero-copy Torch views, and the engine's
``[N, 13]`` rigid-state layout.
"""

from __future__ import annotations
import gc
from pathlib import Path

import kangengine
from kangengine import asset, physics
from kangengine.utils import to_gpu_array_view
import math
import platform

import numpy as np
import torch


RIGID_COUNT = 8
VALIDATION_STEPS = 4
CONTACT_RIGID_ROWS = (0, 1)
RIGID_LAYOUT = {
    "position": slice(0, 3),
    "rotation_xyzw": slice(3, 7),
    "linear_velocity": slice(7, 10),
    "angular_velocity": slice(10, 13),
}


def _print_python_environment():
    print("Environment")
    print(f"  platform     : {platform.platform()}")
    print("  torch CUDA   : initialization deferred until after PhysX GPU init")
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
    if i in CONTACT_RIGID_ROWS:
        x = -6.0 + 0.41 * float(i)
        pos = [x, 0.0, 1.0]
        rot = _z_quat(0.0)
        linear_velocity = [0.5 if i == 0 else -0.5, 0.0, 0.0]
        angular_velocity = [0.0, 0.0, 0.0]
        return pos, rot, linear_velocity, angular_velocity

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
        + gravity
        * (dt * dt)
        * step_count
        * (step_count + np.float32(1.0))
        * np.float32(0.5)
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
    expected_strides = []
    stride = 1
    for extent in reversed(shape):
        expected_strides.append(stride)
        stride *= extent
    expected_strides = tuple(reversed(expected_strides))
    if tuple(view.strides) != expected_strides:
        raise AssertionError(
            f"expected {name} strides {expected_strides}, got {view.strides}"
        )
    if view.device_id != 0:
        raise AssertionError(f"expected {name} device_id 0, got {view.device_id}")
    if view.stream_handle != stream_handle:
        raise AssertionError(
            f"expected {name} stream 0x{stream_handle:x}, got 0x{view.stream_handle:x}"
        )
    if view.ready_event_handle == 0:
        raise AssertionError(f"{name} does not expose a ready event")
    expected_numel = math.prod(shape)
    if view.numel != expected_numel:
        raise AssertionError(f"unexpected {name} numel {view.numel}")
    if view.byte_size != expected_numel * 4:
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
    expected_byte_strides = tuple(value * 4 for value in expected_strides)
    if tuple(cuda_iface["strides"]) != expected_byte_strides:
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
    if tuple(tensor.stride()) != expected_strides:
        raise AssertionError(f"{name} tensor stride mismatch")
    return tensor


def _assert_cuda_uint_view(view, *, shape, name, stream_handle, dtype, typestr):
    if view.name != name:
        raise AssertionError(f"expected view name {name!r}, got {view.name!r}")
    if not view.is_cuda:
        raise AssertionError(f"{name} is not a CUDA view")
    if view.ptr == 0:
        raise AssertionError(f"{name} has a null pointer")
    if tuple(view.shape) != tuple(shape):
        raise AssertionError(f"expected {name} shape {shape}, got {view.shape}")
    expected_strides = []
    stride = 1
    for extent in reversed(shape):
        expected_strides.append(stride)
        stride *= extent
    expected_strides = tuple(reversed(expected_strides))
    if tuple(view.strides) != expected_strides:
        raise AssertionError(
            f"expected {name} strides {expected_strides}, got {view.strides}"
        )
    if view.device_id != 0:
        raise AssertionError(f"expected {name} device_id 0, got {view.device_id}")
    if view.stream_handle != stream_handle:
        raise AssertionError(
            f"expected {name} stream 0x{stream_handle:x}, got 0x{view.stream_handle:x}"
        )
    if view.ready_event_handle == 0:
        raise AssertionError(f"{name} does not expose a ready event")
    expected_numel = math.prod(shape)
    if view.numel != expected_numel:
        raise AssertionError(f"unexpected {name} numel {view.numel}")

    cuda_iface = view.__cuda_array_interface__
    if cuda_iface["version"] != 3:
        raise AssertionError(f"{name} exposes CUDA array interface version mismatch")
    if tuple(cuda_iface["shape"]) != tuple(shape):
        raise AssertionError(f"{name} CUDA interface shape mismatch")
    if cuda_iface["typestr"] != typestr:
        raise AssertionError(f"{name} CUDA interface dtype mismatch")
    if int(cuda_iface["data"][0]) != view.ptr:
        raise AssertionError(f"{name} CUDA interface pointer mismatch")

    tensor = view.torch()
    if tensor.device.type != "cuda":
        raise AssertionError(f"{name} did not produce a CUDA tensor")
    if tensor.data_ptr() != view.ptr:
        raise AssertionError(f"Torch did not create a zero-copy tensor for {name}")
    if tuple(tensor.shape) != tuple(shape):
        raise AssertionError(f"{name} tensor shape mismatch")
    if tensor.dtype != dtype:
        raise AssertionError(f"{name} tensor dtype mismatch")
    return tensor


def _sorted_by_x(state: np.ndarray) -> np.ndarray:
    return state[np.argsort(state[:, 0])]


def _validate_state(
    gpu_state: np.ndarray,
    expected_state: np.ndarray,
    step_index: int,
    *,
    skip_sorted_rows: tuple[int, ...] = (),
):
    gpu_state = _sorted_by_x(gpu_state)
    expected_state = _sorted_by_x(expected_state)
    if skip_sorted_rows:
        mask = np.ones(gpu_state.shape[0], dtype=bool)
        mask[list(skip_sorted_rows)] = False
        gpu_state = gpu_state[mask]
        expected_state = expected_state[mask]

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


def _assert_nonzero_contact_points(contact_points: torch.Tensor, point_count: int):
    if point_count <= 0:
        raise AssertionError("expected nonzero flattened GPU contact points")
    active = contact_points[:point_count].detach().cpu().numpy()
    if not np.isfinite(active).all():
        raise AssertionError("flattened GPU contact points contain non-finite values")

    normal_norms = np.linalg.norm(active[:, 3:6], axis=1)
    impulse_norms = np.linalg.norm(active[:, 6:9], axis=1)
    if float(np.max(normal_norms)) < 0.5:
        raise AssertionError("flattened GPU contact normals look invalid")
    if not np.isfinite(impulse_norms).all():
        raise AssertionError("flattened GPU contact impulses contain non-finite values")

    return active, float(np.max(impulse_norms))


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
            f"expected rigid data version to advance after sparse apply, got {rigid_view.version}"
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

    articulation_data = asset.MJCFLoader.load(
        str(
            Path(kangengine.__file__).resolve().parent
            / "assets"
            / "characters"
            / "kw"
            / "kw5.xml"
        )
    )
    articulation = physics.Articulation.build(
        world, articulation_data, physics.ArticulationConfig.fixed_base()
    )
    articulation_root_position = np.asarray(
        articulation.get_root_position(), dtype=np.float32
    )
    articulation_root_rotation = np.asarray(
        articulation.get_root_rotation(), dtype=np.float32
    )

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

    articulation_row = gpu_system.articulation_row(articulation)
    articulation_link_count = gpu_system.articulation_link_count(articulation_row)
    if articulation_link_count != articulation.num_links():
        raise AssertionError(
            "articulation GPU link count does not match native articulation"
        )
    gpu_system.fetch_articulation_link_pose()
    articulation_link_view = gpu_system.articulation_link_data()
    articulation_link_state = _assert_cuda_float_view(
        articulation_link_view,
        shape=(
            gpu_system.articulation_count(),
            gpu_system.articulation_max_links(),
            13,
        ),
        name="physics_articulation_link_data",
        stream_handle=stream_handle,
    )
    torch.cuda.synchronize(0)
    np.testing.assert_allclose(
        articulation_link_state[articulation_row, 0, 0:3].detach().cpu().numpy(),
        articulation_root_position,
        rtol=1e-5,
        atol=1e-5,
        err_msg="articulation GPU root position mismatch",
    )
    np.testing.assert_allclose(
        articulation_link_state[articulation_row, 0, 3:7].detach().cpu().numpy(),
        articulation_root_rotation,
        rtol=1e-5,
        atol=1e-5,
        err_msg="articulation GPU root rotation mismatch",
    )
    gpu_system.fetch_articulation_link_vel()
    torch.cuda.synchronize(0)
    if articulation_link_view.version != 2:
        raise AssertionError(
            f"expected articulation link data version 2 after pose/velocity fetch, got {articulation_link_view.version}"
        )
    np.testing.assert_allclose(
        articulation_link_state[articulation_row, 0, 7:13].detach().cpu().numpy(),
        np.zeros(6, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
        err_msg="fixed articulation GPU root velocity mismatch",
    )

    articulation_dof_count = gpu_system.articulation_dof_count(articulation_row)
    if articulation_dof_count != articulation.num_dofs():
        raise AssertionError(
            "articulation GPU DOF count does not match native articulation"
        )
    gpu_system.fetch_articulation_joint_positions()
    gpu_system.fetch_articulation_joint_velocities()
    gpu_system.fetch_articulation_joint_accelerations()
    gpu_system.fetch_articulation_joint_forces()
    gpu_system.fetch_articulation_target_joint_positions()
    gpu_system.fetch_articulation_target_joint_velocities()
    gpu_system.fetch_articulation_link_incoming_joint_force()
    articulation_qpos_view = gpu_system.articulation_joint_positions()
    articulation_qvel_view = gpu_system.articulation_joint_velocities()
    articulation_qacc_view = gpu_system.articulation_joint_accelerations()
    articulation_qf_view = gpu_system.articulation_joint_forces()
    articulation_target_qpos_view = gpu_system.articulation_target_joint_positions()
    articulation_target_qvel_view = gpu_system.articulation_target_joint_velocities()
    articulation_incoming_joint_force_view = (
        gpu_system.articulation_link_incoming_joint_forces()
    )
    joint_shape = (
        gpu_system.articulation_count(),
        gpu_system.articulation_max_dofs(),
    )
    link_force_shape = (
        gpu_system.articulation_count(),
        gpu_system.articulation_max_links(),
        6,
    )
    articulation_qpos = _assert_cuda_float_view(
        articulation_qpos_view,
        shape=joint_shape,
        name="physics_articulation_joint_positions",
        stream_handle=stream_handle,
    )
    articulation_qvel = _assert_cuda_float_view(
        articulation_qvel_view,
        shape=joint_shape,
        name="physics_articulation_joint_velocities",
        stream_handle=stream_handle,
    )
    articulation_qacc = _assert_cuda_float_view(
        articulation_qacc_view,
        shape=joint_shape,
        name="physics_articulation_joint_accelerations",
        stream_handle=stream_handle,
    )
    articulation_qf = _assert_cuda_float_view(
        articulation_qf_view,
        shape=joint_shape,
        name="physics_articulation_joint_forces",
        stream_handle=stream_handle,
    )
    articulation_target_qpos = _assert_cuda_float_view(
        articulation_target_qpos_view,
        shape=joint_shape,
        name="physics_articulation_target_joint_positions",
        stream_handle=stream_handle,
    )
    articulation_target_qvel = _assert_cuda_float_view(
        articulation_target_qvel_view,
        shape=joint_shape,
        name="physics_articulation_target_joint_velocities",
        stream_handle=stream_handle,
    )
    articulation_incoming_joint_force = _assert_cuda_float_view(
        articulation_incoming_joint_force_view,
        shape=link_force_shape,
        name="physics_articulation_link_incoming_joint_forces",
        stream_handle=stream_handle,
    )
    torch.cuda.synchronize(0)
    if (
        articulation_qpos_view.version != 1
        or articulation_qvel_view.version != 1
        or articulation_qacc_view.version != 1
        or articulation_qf_view.version != 1
        or articulation_target_qpos_view.version != 1
        or articulation_target_qvel_view.version != 1
        or articulation_incoming_joint_force_view.version != 1
    ):
        raise AssertionError("articulation joint fetch did not advance view versions")
    for label, tensor in (
        ("qpos", articulation_qpos),
        ("qvel", articulation_qvel),
        ("qacc", articulation_qacc),
        ("qf", articulation_qf),
        ("target qpos", articulation_target_qpos),
        ("target qvel", articulation_target_qvel),
    ):
        if not torch.isfinite(tensor[articulation_row, :articulation_dof_count]).all():
            raise AssertionError(f"articulation GPU {label} contains non-finite values")
    if not torch.isfinite(
        articulation_incoming_joint_force[articulation_row, :articulation_link_count, :]
    ).all():
        raise AssertionError(
            "articulation GPU incoming joint force contains non-finite values"
        )
    command_qf = torch.tensor(1.5, device="cuda:0", dtype=torch.float32)
    command_target_qpos = torch.tensor(0.125, device="cuda:0", dtype=torch.float32)
    command_target_qvel = torch.tensor(-0.25, device="cuda:0", dtype=torch.float32)
    articulation_qf[articulation_row, 0] = command_qf
    articulation_target_qpos[articulation_row, 0] = command_target_qpos
    articulation_target_qvel[articulation_row, 0] = command_target_qvel
    gpu_system.apply_articulation_joint_forces()
    gpu_system.apply_articulation_target_joint_positions()
    gpu_system.apply_articulation_target_joint_velocities()
    gpu_system.fetch_articulation_joint_forces()
    gpu_system.fetch_articulation_target_joint_positions()
    gpu_system.fetch_articulation_target_joint_velocities()
    torch.cuda.synchronize(0)
    if (
        articulation_qf_view.version != 3
        or articulation_target_qpos_view.version != 3
        or articulation_target_qvel_view.version != 3
    ):
        raise AssertionError(
            "articulation joint command apply/fetch did not advance versions"
        )
    if not torch.allclose(
        articulation_qf[articulation_row, 0], command_qf, rtol=1e-5, atol=1e-5
    ):
        raise AssertionError("articulation GPU qf command echo mismatch")
    if not torch.allclose(
        articulation_target_qpos[articulation_row, 0],
        command_target_qpos,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError("articulation GPU target qpos command echo mismatch")
    if not torch.allclose(
        articulation_target_qvel[articulation_row, 0],
        command_target_qvel,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError("articulation GPU target qvel command echo mismatch")
    sparse_articulation_indices = torch.tensor(
        [articulation_row], device="cuda:0", dtype=torch.int32
    )
    sparse_articulation_index_view = to_gpu_array_view(
        sparse_articulation_indices,
        dtype=torch.int32,
        name="physics_articulation_sparse_indices",
    )
    command_qf = torch.tensor(2.5, device="cuda:0", dtype=torch.float32)
    command_target_qpos = torch.tensor(0.375, device="cuda:0", dtype=torch.float32)
    command_target_qvel = torch.tensor(-0.5, device="cuda:0", dtype=torch.float32)
    articulation_qf[articulation_row, 0] = command_qf
    articulation_target_qpos[articulation_row, 0] = command_target_qpos
    articulation_target_qvel[articulation_row, 0] = command_target_qvel
    gpu_system.apply_articulation_joint_forces(sparse_articulation_index_view)
    gpu_system.apply_articulation_target_joint_positions(sparse_articulation_index_view)
    gpu_system.apply_articulation_target_joint_velocities(
        sparse_articulation_index_view
    )
    gpu_system.fetch_articulation_joint_forces()
    gpu_system.fetch_articulation_target_joint_positions()
    gpu_system.fetch_articulation_target_joint_velocities()
    torch.cuda.synchronize(0)
    if (
        articulation_qf_view.version != 5
        or articulation_target_qpos_view.version != 5
        or articulation_target_qvel_view.version != 5
    ):
        raise AssertionError(
            "sparse articulation command apply/fetch did not advance versions"
        )
    if not torch.allclose(
        articulation_qf[articulation_row, 0], command_qf, rtol=1e-5, atol=1e-5
    ):
        raise AssertionError("sparse articulation GPU qf command echo mismatch")
    if not torch.allclose(
        articulation_target_qpos[articulation_row, 0],
        command_target_qpos,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError(
            "sparse articulation GPU target qpos command echo mismatch"
        )
    if not torch.allclose(
        articulation_target_qvel[articulation_row, 0],
        command_target_qvel,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError(
            "sparse articulation GPU target qvel command echo mismatch"
        )
    gpu_system.clear_articulation_commands(sparse_articulation_index_view)
    gpu_system.fetch_articulation_joint_forces()
    gpu_system.fetch_articulation_target_joint_positions()
    gpu_system.fetch_articulation_target_joint_velocities()
    torch.cuda.synchronize(0)
    active_dofs = slice(0, articulation_dof_count)
    if not torch.allclose(
        articulation_qf[articulation_row, active_dofs],
        torch.zeros_like(articulation_qf[articulation_row, active_dofs]),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError("clear_articulation_commands left stale joint forces")
    if not torch.allclose(
        articulation_target_qvel[articulation_row, active_dofs],
        torch.zeros_like(articulation_target_qvel[articulation_row, active_dofs]),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError(
            "clear_articulation_commands left stale target joint velocities"
        )
    if not torch.allclose(
        articulation_target_qpos[articulation_row, active_dofs],
        articulation_qpos[articulation_row, active_dofs],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError(
            "clear_articulation_commands did not restore target qpos to qpos"
        )
    root_command = torch.tensor(
        [0.25, 4.5, 0.75, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, -0.1, 0.0, 0.3, 0.0],
        device="cuda:0",
        dtype=torch.float32,
    )
    articulation_link_state[articulation_row, 0, :] = root_command
    gpu_system.apply_articulation_root_pose()
    gpu_system.apply_articulation_root_vel()
    gpu_system.update_articulation_kinematics()
    gpu_system.fetch_articulation_link_pose()
    gpu_system.fetch_articulation_link_vel()
    torch.cuda.synchronize(0)
    if not torch.allclose(
        articulation_link_state[articulation_row, 0, :],
        root_command,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError("articulation GPU root apply echo mismatch")

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
    rigid_force[0] = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
    rigid_torque[RIGID_COUNT - 1] = torch.tensor([-3.0, -2.0, -1.0], device="cuda:0")
    gpu_system.clear_rigid_commands(sparse_index_view)
    torch.cuda.synchronize(0)
    if force_view.version != 3 or torque_view.version != 3:
        raise AssertionError("clear_rigid_commands did not advance command versions")
    if not torch.allclose(
        rigid_force[sparse_indices], torch.zeros_like(rigid_force[sparse_indices])
    ):
        raise AssertionError("clear_rigid_commands left stale rigid force")
    if not torch.allclose(
        rigid_torque[sparse_indices], torch.zeros_like(rigid_torque[sparse_indices])
    ):
        raise AssertionError("clear_rigid_commands left stale rigid torque")

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
    print("Articulation link mirror")
    print(f"  shape        : {articulation_link_view.shape}")
    print(f"  row          : {articulation_row}")
    print(f"  links        : {articulation_link_count}")
    print(f"  version      : {articulation_link_view.version}")
    print("Articulation joint mirrors")
    print(f"  shape        : {articulation_qpos_view.shape}")
    print(f"  dofs         : {articulation_dof_count}")
    print(
        f"  qpos/qvel/qacc/qf ver: {articulation_qpos_view.version} / "
        f"{articulation_qvel_view.version} / {articulation_qacc_view.version} / "
        f"{articulation_qf_view.version}"
    )
    print(
        f"  target ver   : {articulation_target_qpos_view.version} / {articulation_target_qvel_view.version}"
    )
    print(
        f"  incoming f   : {articulation_incoming_joint_force_view.shape} "
        f"ver {articulation_incoming_joint_force_view.version}"
    )
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
        sorted_state = _validate_state(
            gpu_state,
            expected_state,
            step_index,
            skip_sorted_rows=tuple(range(len(CONTACT_RIGID_ROWS))),
        )
        _print_step_sample(step_index, sorted_state)

    gpu_system.fetch_contact_pairs()
    contact_pairs_view = gpu_system.contact_pairs()
    contact_pair_count_view = gpu_system.contact_pair_count()
    contact_pair_headers_view = gpu_system.contact_pair_headers()
    contact_pair_body_refs_view = gpu_system.contact_pair_body_refs()
    contact_points_view = gpu_system.contact_points()
    contact_point_count_view = gpu_system.contact_point_count()
    contact_point_pair_indices_view = gpu_system.contact_point_pair_indices()
    contact_pairs_raw = _assert_cuda_uint_view(
        contact_pairs_view,
        shape=tuple(contact_pairs_view.shape),
        name="physics_contact_pairs_raw",
        stream_handle=stream_handle,
        dtype=torch.uint8,
        typestr="|u1",
    )
    contact_pair_count = _assert_cuda_uint_view(
        contact_pair_count_view,
        shape=(1,),
        name="physics_contact_pair_count",
        stream_handle=stream_handle,
        dtype=torch.int32,
        typestr="<u4",
    )
    contact_pair_headers = _assert_cuda_uint_view(
        contact_pair_headers_view,
        shape=tuple(contact_pair_headers_view.shape),
        name="physics_contact_pair_headers",
        stream_handle=stream_handle,
        dtype=torch.int64,
        typestr="<u8",
    )
    contact_pair_body_refs = _assert_cuda_uint_view(
        contact_pair_body_refs_view,
        shape=tuple(contact_pair_body_refs_view.shape),
        name="physics_contact_pair_body_refs",
        stream_handle=stream_handle,
        dtype=torch.int32,
        typestr="<i4",
    )
    contact_points = _assert_cuda_float_view(
        contact_points_view,
        shape=tuple(contact_points_view.shape),
        name="physics_contact_points",
        stream_handle=stream_handle,
    )
    contact_point_count = _assert_cuda_uint_view(
        contact_point_count_view,
        shape=(1,),
        name="physics_contact_point_count",
        stream_handle=stream_handle,
        dtype=torch.int32,
        typestr="<u4",
    )
    contact_point_pair_indices = _assert_cuda_uint_view(
        contact_point_pair_indices_view,
        shape=tuple(contact_point_pair_indices_view.shape),
        name="physics_contact_point_pair_indices",
        stream_handle=stream_handle,
        dtype=torch.int32,
        typestr="<u4",
    )
    torch.cuda.synchronize(0)
    pair_count = int(contact_pair_count.detach().cpu().item())
    point_count = int(contact_point_count.detach().cpu().item())
    if pair_count <= 0:
        raise AssertionError("expected nonzero GPU contact pairs")
    if point_count > contact_points_view.shape[0]:
        raise AssertionError("contact point count exceeds flattened contact capacity")
    if (
        contact_pairs_view.version != 1
        or contact_pair_count_view.version != 1
        or contact_pair_headers_view.version != 1
        or contact_pair_body_refs_view.version != 1
        or contact_points_view.version != 1
        or contact_point_count_view.version != 1
        or contact_point_pair_indices_view.version != 1
    ):
        raise AssertionError("contact pair fetch did not advance view versions")
    active_contact_points, max_contact_impulse = _assert_nonzero_contact_points(
        contact_points, point_count
    )
    if max_contact_impulse <= 0.0:
        raise AssertionError("PhysX GPU impact contacts did not report solver impulse")
    active_pair_indices = contact_point_pair_indices[:point_count]
    if not torch.all((active_pair_indices >= 0) & (active_pair_indices < pair_count)):
        raise AssertionError("flattened contact point pair indices are invalid")
    active_headers = contact_pair_headers[:pair_count].detach().cpu().numpy()
    if active_headers.shape[1] != 6:
        raise AssertionError("contact pair header layout mismatch")
    if not np.any(active_headers[:, 0:2]):
        raise AssertionError("contact pair headers did not capture node indices")
    if not np.any(active_headers[:, 2:4]):
        raise AssertionError("contact pair headers did not capture actor pointers")
    active_body_refs = contact_pair_body_refs[:pair_count].detach().cpu().numpy()
    if active_body_refs.shape[1] != 6:
        raise AssertionError("contact pair body-ref layout mismatch")
    expected_contact_rows = sorted(rigid_rows[i] for i in CONTACT_RIGID_ROWS)
    found_expected_rigid_pair = False
    for row in active_body_refs:
        if row[0] != 0 or row[3] != 0:
            continue
        if row[2] != 0 or row[5] != 0:
            continue
        if sorted((int(row[1]), int(row[4]))) == expected_contact_rows:
            found_expected_rigid_pair = True
            break
    if not found_expected_rigid_pair:
        raise AssertionError(
            "contact pair body refs did not map the overlapping rigid actors"
        )
    print()
    print("Contact pair mirror")
    print(f"  raw bytes    : {contact_pairs_raw.numel()}")
    print(f"  pair count   : {pair_count}")
    print(f"  headers shape: {contact_pair_headers_view.shape}")
    print(f"  body refs    : {contact_pair_body_refs_view.shape}")
    print(f"  points shape : {contact_points_view.shape}")
    print(f"  point count  : {point_count}")
    print(f"  point pairs  : {contact_point_pair_indices_view.shape}")
    print(f"  max impulse  : {max_contact_impulse:.6f}")
    print(
        "  first point  : "
        f"pos=[{active_contact_points[0, 0]: .5f}, "
        f"{active_contact_points[0, 1]: .5f}, "
        f"{active_contact_points[0, 2]: .5f}] "
        f"normal=[{active_contact_points[0, 3]: .5f}, "
        f"{active_contact_points[0, 4]: .5f}, "
        f"{active_contact_points[0, 5]: .5f}]"
    )
    print(f"  version      : {contact_pairs_view.version}")

    _assert_sparse_root_state_apply(world, gpu_system, rigid_state, rigid_view, config)

    del rigid_state
    del rigid_force
    del rigid_torque
    del sparse_index_view
    del articulation_link_state
    del articulation_qpos
    del articulation_qvel
    del articulation_qacc
    del articulation_qf
    del articulation_target_qpos
    del articulation_target_qvel
    del articulation_incoming_joint_force
    del contact_pairs_raw
    del contact_pair_count
    del contact_pair_headers
    del contact_pair_body_refs
    del contact_points
    del contact_point_count
    del contact_point_pair_indices
    del root_command
    del sparse_articulation_index_view
    del sparse_articulation_indices
    gpu_system.invalidate()
    articulation.release()
    for actor in actors:
        actor.release()
    del actors
    del gpu_system
    del world
    gc.collect()

    print()
    print(
        "PASS: PhysX GPU rigid state/apply and articulation link/joint "
        f"state/command validation completed ({VALIDATION_STEPS} steps)"
    )


if __name__ == "__main__":
    main()
