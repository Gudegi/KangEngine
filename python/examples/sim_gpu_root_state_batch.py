"""High-level KangSimWorld PhysX GPU root-state batch example.

This example stays headless: it creates one rigid body per environment, builds
the explicit PhysX GPU mirror, applies sparse root-state resets through the
high-level ``set_root_state(env_ids, ...)`` API, then reads the GPU rigid state
as a Torch CUDA tensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
import torch


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def spawn_position(env_id: int):
    return [float(env_id) * 0.75, 0.0, 2.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()

    world = ke.sim.KangSimWorld(
        num_envs=args.num_envs,
        sim_device=f"cuda:{args.cuda_device}",
        sim_dt=1.0 / 120.0,
        add_ground=True,
    )
    try:
        ball_xml = package_asset_path("objects", "ball.xml")
        ball_data = world.load_mjcf(ball_xml)
        for env_id in range(args.num_envs):
            world.add_rigid(
                ball_data,
                env_id=env_id,
                obj_id=0,
                name=f"ball_{env_id}",
                pos=spawn_position(env_id),
                density=10.0,
            )

        gpu_system = world.init_gpu_system(cuda_device_id=args.cuda_device)
        print("KangSimWorld PhysX GPU root-state batch")
        print(f"  envs       : {args.num_envs}")
        print(
            f"  rigid rows : {[world.rigid_gpu_row(i, 0) for i in range(args.num_envs)]}"
        )

        selected_envs = list(range(1, args.num_envs, 2))
        index_view = world.rigid_gpu_index_view(selected_envs, obj_id=0)
        cached_index_view = world.rigid_gpu_index_view(selected_envs, obj_id=0)
        print(f"  selected   : {selected_envs}")
        print(f"  index view : ptr=0x{index_view.ptr:x}, shape={index_view.shape}")
        print(f"  cached ptr : {index_view.ptr == cached_index_view.ptr}")

        positions = np.asarray(
            [[5.0 + env_id, -2.0, 8.0] for env_id in selected_envs],
            dtype=np.float32,
        )
        rotations = np.zeros((len(selected_envs), 4), dtype=np.float32)
        rotations[:, 3] = 1.0
        linear_velocity = np.zeros((len(selected_envs), 3), dtype=np.float32)
        linear_velocity[:, 0] = 1.5

        world.set_root_state(
            selected_envs,
            0,
            positions,
            rotations,
            linear_velocity=linear_velocity,
            angular_velocity=[0.0, 0.0, 0.0],
        )

        for _ in range(args.steps):
            world.step(refresh=False)

        gpu_system.fetch_rigid_data()
        torch.cuda.synchronize(args.cuda_device)
        rigid_state = torch.as_tensor(
            gpu_system.rigid_data(), device=f"cuda:{args.cuda_device}"
        )

        print("  gpu state samples:")
        for env_id in range(args.num_envs):
            row = world.rigid_gpu_row(env_id, 0)
            pos = rigid_state[row, 0:3].detach().cpu().numpy()
            vel = rigid_state[row, 7:10].detach().cpu().numpy()
            print(
                f"    env {env_id:02d} row {row:02d} "
                f"pos=[{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}] "
                f"vel=[{vel[0]: .3f}, {vel[1]: .3f}, {vel[2]: .3f}]"
            )
    finally:
        world.release()


if __name__ == "__main__":
    main()
