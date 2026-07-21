"""Validate KangSimWorld root-state resets through PhysX GPU rigid buffers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kangengine as ke


def main():
    world = ke.sim.KangSimWorld(num_envs=2, sim_device="cuda", add_ground=False)
    try:
        ball_xml = Path(ke.__file__).resolve().parent / "assets" / "objects" / "ball.xml"
        data = world.load_mjcf(str(ball_xml))
        world.add_rigid(data, env_id=0, obj_id=0, pos=[0.0, 0.0, 2.0], density=10.0)
        world.add_rigid(data, env_id=1, obj_id=0, pos=[1.0, 0.0, 2.0], density=10.0)

        world.init_gpu_system(cuda_device_id=0)
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
        from kangengine._core import _ke
        from kangengine.utils import to_gpu_array_view

        rigid_state = world.get_gpu_rigid_data()
        render_transforms = torch.empty((1, 4, 4), device="cuda:0")
        _ke.indexed_rigid_state_to_mat4_cuda(
            world.gpu_system.rigid_data(),
            index_view,
            to_gpu_array_view(render_transforms),
        )
        torch.cuda.synchronize(0)
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
        np.testing.assert_allclose(
            render_transforms[0, 3, :3].detach().cpu().numpy(),
            selected,
            rtol=1e-5,
            atol=1e-5,
            err_msg="indexed rigid CUDA Mat4 translation mismatch",
        )

        cuda_pos = torch.tensor(
            [[-2.0, 1.5, 4.0]], dtype=torch.float32, device="cuda:0"
        )
        cuda_rot = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0"
        )
        cuda_vel = torch.tensor(
            [[0.25, -0.5, 0.75]], dtype=torch.float32, device="cuda:0"
        )
        cuda_ang_vel = torch.tensor(
            [[-0.1, 0.2, -0.3]], dtype=torch.float32, device="cuda:0"
        )
        world.set_root_state(
            [0],
            0,
            cuda_pos,
            cuda_rot,
            linear_velocity=cuda_vel,
            angular_velocity=cuda_ang_vel,
        )
        pending_root = world.resets[(0, 0)].root
        if pending_root is None:
            raise AssertionError("CUDA rigid root reset was not queued")
        if pending_root.pos.device.type != "cuda":
            raise AssertionError("CUDA rigid root reset was staged through CPU")
        world.apply_resets()
        rigid_state = world.get_gpu_rigid_data()
        torch.cuda.synchronize(0)
        cuda_row = world.rigid_gpu_row(0, 0)
        np.testing.assert_allclose(
            rigid_state[cuda_row, 0:3].detach().cpu().numpy(),
            cuda_pos[0].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CUDA tensor rigid root position reset mismatch",
        )
        np.testing.assert_allclose(
            rigid_state[cuda_row, 7:10].detach().cpu().numpy(),
            cuda_vel[0].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CUDA tensor rigid root velocity reset mismatch",
        )

        print("PASS: KangSimWorld GPU root-state and CUDA Mat4 gather path")
    finally:
        world.release()


if __name__ == "__main__":
    main()
