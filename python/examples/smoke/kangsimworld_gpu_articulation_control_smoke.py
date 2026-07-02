"""Compare CPU POS drive, GPU POS target, and GPU explicit PD control."""

from __future__ import annotations

from pathlib import Path

import kangengine as ke
import torch


OBJ_ID = 0


def asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def target_tensor(num_dofs: int, sim_time: float, *, device) -> torch.Tensor:
    dof_ids = torch.arange(num_dofs, dtype=torch.float32, device=device)
    phase = dof_ids * 0.37
    frequency = 1.2 + 0.15 * torch.remainder(dof_ids, 5.0)
    return 0.45 * torch.sin(
        2.0 * torch.pi * frequency * float(sim_time) + phase
    )


def configure_drives(articulation, num_dofs: int, kp: float, kd: float):
    articulation.set_kps([float(kp)] * num_dofs)
    articulation.set_kds([float(kd)] * num_dofs)


def make_world(*, gpu: bool, cuda_device: int = 0):
    world = ke.KangSimWorld(
        num_envs=1,
        sim_device=f"cuda:{cuda_device}" if gpu else "cpu",
        sim_dt=1.0 / 120.0,
        add_ground=False,
    )
    data = world.load_mjcf(asset_path("characters", "kw", "kw5.xml"), order="DFS")
    config = ke.ArticulationConfig.fixed_base()
    record = world.add_articulation(data, env_id=0, obj_id=OBJ_ID, config=config)
    num_dofs = record.num_dofs
    world.set_root_state(
        None,
        OBJ_ID,
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
    )
    world.set_dof_state(
        None,
        OBJ_ID,
        torch.zeros((1, num_dofs), dtype=torch.float32),
        velocities=torch.zeros((1, num_dofs), dtype=torch.float32),
    )
    return world, record, num_dofs


def summarize(
    name: str,
    q_samples: list[torch.Tensor],
    target_samples: list[torch.Tensor],
):
    q = torch.stack(q_samples)
    target = torch.stack(target_samples)
    delta = (q[-1] - q[0]).abs().max().item()
    last_window = q[-60:] if q.shape[0] >= 60 else q
    window_motion = (last_window[1:] - last_window[:-1]).abs().mean().item()
    tracking_error = (q[-1] - target[-1]).abs().mean().item()
    print(
        f"{name:16s} | max_delta={delta: .4f} "
        f"last_motion={window_motion: .6f} "
        f"final_err={tracking_error: .4f}"
    )
    return {
        "max_delta": float(delta),
        "last_motion": float(window_motion),
        "final_err": float(tracking_error),
    }


def run_cpu_pos(steps: int, kp: float, kd: float):
    world, record, num_dofs = make_world(gpu=False)
    try:
        configure_drives(record.articulation, num_dofs, kp, kd)
        world.step(substeps=0, apply_commands=False)
        q_samples = []
        target_samples = []
        for _ in range(steps):
            target = target_tensor(num_dofs, world.sim_time, device="cpu")
            world.set_cmd(
                None,
                OBJ_ID,
                target.numpy(),
                mode=ke.ControlMode.POS,
                kp=None,
                kd=None,
            )
            world.step(substeps=1)
            q = torch.as_tensor(world.state.get_dof_pos(OBJ_ID)[0]).clone()
            q_samples.append(q)
            target_samples.append(target.clone())
        return summarize("CPU POS drive", q_samples, target_samples)
    finally:
        world.release()


def run_gpu_pos(steps: int, kp: float, kd: float, cuda_device: int):
    world, record, num_dofs = make_world(gpu=True, cuda_device=cuda_device)
    try:
        configure_drives(record.articulation, num_dofs, kp, kd)
        world.init_gpu_system(cuda_device_id=cuda_device)
        world.step(substeps=0, refresh=False, apply_commands=False)
        row = world.articulation_gpu_row(0, OBJ_ID)
        q_samples = []
        target_samples = []
        max_target_echo_error = 0.0
        for _ in range(steps):
            target = target_tensor(
                num_dofs, world.sim_time, device=f"cuda:{cuda_device}"
            )
            world.set_cmd(
                None,
                OBJ_ID,
                target,
                mode=ke.ControlMode.POS,
                kp=None,
                kd=None,
            )
            world.step(substeps=1, refresh=False)
            q = world.get_gpu_articulation_joint_positions()[row, :num_dofs]
            target_echo = world.get_gpu_articulation_target_joint_positions()[
                row, :num_dofs
            ]
            max_target_echo_error = max(
                max_target_echo_error,
                float((target_echo - target).abs().max().item()),
            )
            q_samples.append(q.detach().cpu().clone())
            target_samples.append(target.detach().cpu().clone())
        torch.cuda.synchronize(cuda_device)
        result = summarize("GPU POS target", q_samples, target_samples)
        result["max_target_echo_error"] = max_target_echo_error
        print(f"{'GPU POS target':16s} | target_echo_err={max_target_echo_error: .6g}")
        return result
    finally:
        world.release()


def run_gpu_explicit_pd(steps: int, kp: float, kd: float, cuda_device: int):
    world, record, num_dofs = make_world(gpu=True, cuda_device=cuda_device)
    try:
        configure_drives(record.articulation, num_dofs, 0.0, 0.0)
        world.init_gpu_system(cuda_device_id=cuda_device)
        world.step(substeps=0, refresh=False, apply_commands=False)
        row = world.articulation_gpu_row(0, OBJ_ID)
        device = torch.device(f"cuda:{cuda_device}")
        kp_array = torch.full((num_dofs,), float(kp), dtype=torch.float32, device=device)
        kd_array = torch.full((num_dofs,), float(kd), dtype=torch.float32, device=device)
        q_samples = []
        target_samples = []
        for _ in range(steps):
            target = target_tensor(num_dofs, world.sim_time, device=device)
            world.set_cmd(
                None,
                OBJ_ID,
                target,
                mode=ke.ControlMode.PD_EXPLICIT,
                kp=kp_array,
                kd=kd_array,
            )
            world.step(substeps=1, refresh=False)
            next_q = world.get_gpu_articulation_joint_positions()[row, :num_dofs]
            q_samples.append(next_q.detach().cpu().clone())
            target_samples.append(target.detach().cpu().clone())
        torch.cuda.synchronize(cuda_device)
        return summarize("GPU PD_EXPLICIT", q_samples, target_samples)
    finally:
        world.release()


def main():
    steps = 360
    kp = 60.0
    kd = 6.0
    cuda_device = 0

    print("KangSimWorld articulation control comparison")
    print(f"  steps: {steps}")
    print(f"  gains: kp={kp:g}, kd={kd:g}")

    cpu_pos = run_cpu_pos(steps, kp, kd)
    gpu_pos = run_gpu_pos(steps, kp, kd, cuda_device)
    gpu_pd = run_gpu_explicit_pd(steps, kp, kd, cuda_device)

    if cpu_pos["last_motion"] <= 1e-5:
        raise AssertionError("CPU POS drive did not keep moving")
    if gpu_pd["last_motion"] <= 1e-5:
        raise AssertionError("GPU explicit PD did not keep moving")
    if gpu_pos["last_motion"] <= 1e-4:
        if gpu_pos["max_target_echo_error"] > 1e-5:
            raise AssertionError(
                "GPU POS target buffer did not echo the uploaded target"
            )
        raise AssertionError("GPU POS target drive did not keep moving")

    print("PASS: articulation control comparison completed")


if __name__ == "__main__":
    main()
