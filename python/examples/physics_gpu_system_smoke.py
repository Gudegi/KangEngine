"""Validate the PhysX GPU rigid mirror against CPU-visible actor state.

Run this on Linux with an NVIDIA GPU after building the Python extension. A
successful run verifies CUDA context compatibility, PhysX ``copyBodyData``,
the zero-copy Torch view, and the engine's ``[N, 13]`` rigid-state layout.
"""

from __future__ import annotations

import platform

import numpy as np
import torch

from kangengine import physics


def _print_environment():
    print("Environment")
    print(f"  platform     : {platform.platform()}")
    print(f"  torch        : {torch.__version__}")
    print(f"  torch cuda   : {torch.version.cuda}")
    print(f"  cuda devices : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  device 0     : {torch.cuda.get_device_name(0)}")
    print()


def _cpu_state(actor) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(actor.get_root_position(), dtype=np.float32),
            np.asarray(actor.get_root_rotation(), dtype=np.float32),
            np.asarray(actor.get_root_linear_velocity(), dtype=np.float32),
            np.asarray(actor.get_root_angular_velocity(), dtype=np.float32),
        ]
    )


def main():
    _print_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this test requires an NVIDIA GPU")

    config = physics.PhysicsConfig.z_up()
    config.enable_gpu = True
    config.enable_contact_reports = False
    world = physics.PhysicsWorld(config)

    actors = [
        world.create_dynamic_box([0.2, 0.2, 0.2], [-1.0, 0.0, 2.0]),
        world.create_dynamic_box([0.2, 0.2, 0.2], [1.0, 0.0, 3.0]),
    ]
    actors[0].set_root_state(
        [-1.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.25, 0.0, 0.0],
        [0.0, 0.0, 0.5],
    )
    actors[1].set_root_state(
        [1.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
        [-0.25, 0.0, 0.0],
        [0.0, 0.0, -0.5],
    )

    gpu_config = physics.GpuPhysicsConfig()
    gpu_config.cuda_device_id = 0
    gpu_system = physics.PhysicsGpuSystem(world, gpu_config)
    stream = torch.cuda.current_stream(device=0)
    gpu_system.set_cuda_stream(int(stream.cuda_stream))
    gpu_system.init()
    gpu_system.fetch_rigid_data()

    view = gpu_system.rigid_data()
    state = torch.as_tensor(view, device="cuda:0")
    torch.cuda.synchronize(0)

    if tuple(state.shape) != (2, 13):
        raise AssertionError(f"expected rigid state shape (2, 13), got {state.shape}")
    if state.data_ptr() != view.ptr:
        raise AssertionError("Torch did not create a zero-copy GpuArrayView")

    gpu_state = state.detach().cpu().numpy()
    cpu_state = np.stack([_cpu_state(actor) for actor in actors])

    # Actor enumeration order is not a public PhysX contract. Sort both arrays
    # by world-space x before comparing their state rows.
    gpu_state = gpu_state[np.argsort(gpu_state[:, 0])]
    cpu_state = cpu_state[np.argsort(cpu_state[:, 0])]
    np.testing.assert_allclose(gpu_state, cpu_state, rtol=1e-5, atol=1e-5)

    print("Rigid mirror")
    print(f"  pointer      : 0x{view.ptr:x}")
    print(f"  shape        : {view.shape}")
    print(f"  strides      : {view.strides}")
    print(f"  version      : {view.version}")
    print(f"  stream       : 0x{view.stream_handle:x}")
    print(f"  ready event  : 0x{view.ready_event_handle:x}")
    print()
    print("PASS: PhysX GPU rigid state matches CPU-visible actor state")


if __name__ == "__main__":
    main()
