"""Validate KangSimWorld root-state resets through PhysX GPU rigid buffers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kangengine as ke


def main():
    world = ke.KangSimWorld(num_envs=2, sim_device="cuda", add_ground=False)
    try:
        ball_xml = Path(ke.__file__).resolve().parent / "assets" / "objects" / "ball.xml"
        data = world.load_mjcf(str(ball_xml))
        world.add_rigid(data, env_id=0, obj_id=0, pos=[0.0, 0.0, 2.0], density=10.0)
        world.add_rigid(data, env_id=1, obj_id=0, pos=[1.0, 0.0, 2.0], density=10.0)

        gpu_system = world.init_gpu_system(cuda_device_id=0)
        rows = [world.rigid_gpu_row(0, 0), world.rigid_gpu_row(1, 0)]
        if sorted(rows) != [0, 1]:
            raise AssertionError(f"unexpected KangSimWorld rigid GPU rows {rows}")
        index_view = world.rigid_gpu_index_view([1], 0)
        cached_index_view = world.rigid_gpu_index_view([1], 0)
        if index_view.ptr != cached_index_view.ptr or index_view.shape != [1]:
            raise AssertionError("KangSimWorld rigid GPU index buffer was not cached")

        target_pos = [30.0, -3.0, 9.0]
        world.set_root_state(
            [1],
            0,
            target_pos,
            [0.0, 0.0, 0.0, 1.0],
            linear_velocity=[0.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )
        world.step(refresh=False)

        import torch

        gpu_system.fetch_rigid_data()
        torch.cuda.synchronize(0)
        rigid_state = torch.as_tensor(gpu_system.rigid_data(), device="cuda:0")
        selected = rigid_state[world.rigid_gpu_row(1, 0), :3].detach().cpu().numpy()
        other = rigid_state[world.rigid_gpu_row(0, 0), :3].detach().cpu().numpy()

        np.testing.assert_allclose(
            selected[:2],
            np.asarray(target_pos[:2], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
            err_msg="KangSimWorld GPU root-state selected row mismatch",
        )
        if abs(float(other[0]) - target_pos[0]) < 1.0:
            raise AssertionError("KangSimWorld GPU root-state moved the wrong row")

        print("PASS: KangSimWorld GPU root-state sparse apply path")
    finally:
        world.release()


if __name__ == "__main__":
    main()
