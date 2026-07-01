"""Validate KangSimWorld articulation resets through PhysX GPU buffers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kangengine as ke


def main():
    world = ke.KangSimWorld(num_envs=2, sim_device="cuda", add_ground=False)
    try:
        mjcf_path = (
            Path(ke.__file__).resolve().parent
            / "assets"
            / "characters"
            / "kw"
            / "kw5.xml"
        )
        data = world.load_mjcf(str(mjcf_path))
        config = ke.physics.ArticulationConfig.fixed_base()
        world.add_articulation(data, env_id=0, obj_id=0, config=config)
        world.add_articulation(data, env_id=1, obj_id=0, config=config)

        world.init_gpu_system(cuda_device_id=0)
        rows = [world.articulation_gpu_row(0, 0), world.articulation_gpu_row(1, 0)]
        if sorted(rows) != [0, 1]:
            raise AssertionError(
                f"unexpected KangSimWorld articulation GPU rows {rows}"
            )
        index_view = world.articulation_gpu_index_view([1], 0)
        cached_index_view = world.articulation_gpu_index_view([1], 0)
        if index_view.ptr != cached_index_view.ptr or index_view.shape != [1]:
            raise AssertionError(
                "KangSimWorld articulation GPU index buffer was not cached"
            )

        target_root = np.asarray([0.25, 4.5, 0.75], dtype=np.float32)
        target_qpos = np.zeros(world.articulations[(1, 0)].num_dofs, dtype=np.float32)
        target_qvel = np.zeros_like(target_qpos)
        target_qpos[0] = 0.125
        target_qvel[0] = -0.25
        world.set_root_state(
            [1],
            0,
            target_root,
            [0.0, 0.0, 0.0, 1.0],
            linear_velocity=[0.2, 0.0, -0.1],
            angular_velocity=[0.0, 0.3, 0.0],
        )
        world.set_dof_state([1], 0, target_qpos, target_qvel)
        world.apply_resets()

        import torch
        from kangengine._core import _ke
        from kangengine.utils import to_gpu_array_view

        link_state = world.get_gpu_articulation_link_data()
        qpos = world.get_gpu_articulation_joint_positions()
        qvel = world.get_gpu_articulation_joint_velocities()
        link_count = world.articulations[(0, 0)].num_bodies
        render_transforms = torch.empty(
            (link_count, world.num_envs, 4, 4), device="cuda:0"
        )
        link_indices = torch.tensor(
            world.articulations[(0, 0)].articulation.get_link_indices(),
            dtype=torch.int32,
            device="cuda:0",
        )
        _ke.articulation_link_state_to_mat4_cuda(
            world.gpu_system.articulation_link_data(),
            world.articulation_gpu_index_view(None, 0),
            to_gpu_array_view(link_indices, dtype=torch.int32),
            to_gpu_array_view(render_transforms),
            link_count,
        )
        torch.cuda.synchronize(0)

        selected_row = world.articulation_gpu_row(1, 0)
        other_row = world.articulation_gpu_row(0, 0)
        selected_root = link_state[selected_row, 0, :3].detach().cpu().numpy()
        other_root = link_state[other_row, 0, :3].detach().cpu().numpy()

        np.testing.assert_allclose(
            selected_root,
            target_root,
            rtol=1e-5,
            atol=1e-5,
            err_msg="KangSimWorld GPU articulation root selected row mismatch",
        )
        if np.allclose(other_root, target_root, rtol=1e-5, atol=1e-5):
            raise AssertionError(
                "KangSimWorld GPU articulation root moved the wrong row"
            )
        np.testing.assert_allclose(
            render_transforms[0, 1, 3, :3].detach().cpu().numpy(),
            selected_root,
            rtol=1e-5,
            atol=1e-5,
            err_msg="articulation CUDA Mat4 root translation mismatch",
        )
        np.testing.assert_allclose(
            render_transforms[:, 1, 3, :3].detach().cpu().numpy(),
            link_state[selected_row, link_indices.long(), :3]
            .detach()
            .cpu()
            .numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="articulation visual-to-PhysX link permutation mismatch",
        )

        cuda_root = torch.tensor(
            [[-0.5, 2.25, 1.25]], dtype=torch.float32, device="cuda:0"
        )
        cuda_root_rot = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0"
        )
        cuda_root_vel = torch.tensor(
            [[0.1, -0.2, 0.3]], dtype=torch.float32, device="cuda:0"
        )
        cuda_root_ang_vel = torch.tensor(
            [[-0.4, 0.5, -0.6]], dtype=torch.float32, device="cuda:0"
        )
        world.set_root_state(
            [0],
            0,
            cuda_root,
            cuda_root_rot,
            linear_velocity=cuda_root_vel,
            angular_velocity=cuda_root_ang_vel,
        )
        pending_root = world.resets[(0, 0)].root
        if pending_root is None:
            raise AssertionError("CUDA articulation root reset was not queued")
        if pending_root.pos.device.type != "cuda":
            raise AssertionError("CUDA articulation root reset was staged through CPU")
        world.apply_resets()
        link_state = world.get_gpu_articulation_link_data()
        torch.cuda.synchronize(0)
        np.testing.assert_allclose(
            link_state[other_row, 0, 0:3].detach().cpu().numpy(),
            cuda_root[0].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CUDA tensor articulation root position reset mismatch",
        )
        np.testing.assert_allclose(
            link_state[other_row, 0, 7:10].detach().cpu().numpy(),
            cuda_root_vel[0].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CUDA tensor articulation root velocity reset mismatch",
        )
        np.testing.assert_allclose(
            qpos[selected_row, : target_qpos.size].detach().cpu().numpy(),
            target_qpos,
            rtol=1e-5,
            atol=1e-5,
            err_msg="KangSimWorld GPU articulation qpos selected row mismatch",
        )
        np.testing.assert_allclose(
            qvel[selected_row, : target_qvel.size].detach().cpu().numpy(),
            target_qvel,
            rtol=1e-5,
            atol=1e-5,
            err_msg="KangSimWorld GPU articulation qvel selected row mismatch",
        )

        cuda_qpos = torch.zeros(
            (1, target_qpos.size), dtype=torch.float32, device="cuda:0"
        )
        cuda_qvel = torch.zeros_like(cuda_qpos)
        cuda_qpos[0, 0] = -0.2
        cuda_qvel[0, 0] = 0.3
        world.set_dof_state([0], 0, cuda_qpos, cuda_qvel)
        pending_dof = world.resets[(0, 0)].dof
        if pending_dof is None:
            raise AssertionError("CUDA DOF reset was not queued")
        if pending_dof.positions.device.type != "cuda":
            raise AssertionError("CUDA DOF reset was staged through CPU")
        world.apply_resets()
        qpos = world.get_gpu_articulation_joint_positions()
        qvel = world.get_gpu_articulation_joint_velocities()
        torch.cuda.synchronize(0)
        np.testing.assert_allclose(
            qpos[other_row, : target_qpos.size].detach().cpu().numpy(),
            cuda_qpos[0].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CUDA tensor DOF position reset mismatch",
        )
        np.testing.assert_allclose(
            qvel[other_row, : target_qvel.size].detach().cpu().numpy(),
            cuda_qvel[0].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CUDA tensor DOF velocity reset mismatch",
        )

        command_qpos = np.zeros_like(target_qpos)
        command_qvel = np.zeros_like(target_qvel)
        command_qf = np.zeros_like(target_qpos)
        command_qpos[0] = 0.25
        command_qvel[0] = -0.375
        command_qf[0] = 0.5

        world.set_cmd([1], 0, command_qpos, mode=ke.ControlMode.POS)
        world.apply_commands()
        target_pos = world.get_gpu_articulation_target_joint_positions()
        torch.cuda.synchronize(0)
        np.testing.assert_allclose(
            target_pos[selected_row, : command_qpos.size].detach().cpu().numpy(),
            command_qpos,
            rtol=1e-5,
            atol=1e-5,
            err_msg="KangSimWorld GPU articulation POS command mismatch",
        )

        world.set_cmd([1], 0, command_qvel, mode=ke.ControlMode.VEL)
        world.apply_commands()
        target_vel = world.get_gpu_articulation_target_joint_velocities()
        torch.cuda.synchronize(0)
        np.testing.assert_allclose(
            target_vel[selected_row, : command_qvel.size].detach().cpu().numpy(),
            command_qvel,
            rtol=1e-5,
            atol=1e-5,
            err_msg="KangSimWorld GPU articulation VEL command mismatch",
        )

        world.set_cmd([1], 0, command_qf, mode=ke.ControlMode.TORQUE)
        world.apply_commands()
        qf = world.get_gpu_articulation_joint_forces()
        torch.cuda.synchronize(0)
        np.testing.assert_allclose(
            qf[selected_row, : command_qf.size].detach().cpu().numpy(),
            command_qf,
            rtol=1e-5,
            atol=1e-5,
            err_msg="KangSimWorld GPU articulation TORQUE command mismatch",
        )

        print(
            "PASS: KangSimWorld GPU articulation state/command and CUDA Mat4 "
            "gather path"
        )
    finally:
        world.release()


if __name__ == "__main__":
    main()
