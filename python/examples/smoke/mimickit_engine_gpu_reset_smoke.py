"""Validate MimicKit adapter reset poses and GPU ground-contact forces."""

from pathlib import Path

import torch

import kangengine as ke
from kangengine.mimickit_engine import KangEngineEngine, MimicObjType


def main():
    device = torch.device("cuda:0")
    engine = KangEngineEngine(
        {
            "control_mode": "none",
            "control_freq": 120,
            "sim_freq": 120,
            "sim_device": "cuda:0",
            "env_spacing": 2.0,
        },
        num_envs=2,
        device=device,
        visualize=False,
    )
    articulation_xml = str(
        Path(ke.__file__).resolve().parent / "assets" / "characters" / "kw" / "kw5.xml"
    )

    try:
        for env_id in range(2):
            engine.create_env()
            engine.create_obj(
                env_id,
                MimicObjType.articulated,
                articulation_xml,
                "character",
                start_pos=[0.0, 0.0, 1.5],
            )
        engine.initialize_sim()

        reset_ids = torch.tensor([1], dtype=torch.long, device=device)
        reset_root_pos = torch.tensor(
            [[0.25, -0.5, 1.25]], dtype=torch.float32, device=device
        )
        reset_root_rot = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        engine.set_root_pos(reset_ids, 0, reset_root_pos)
        engine.set_root_rot(reset_ids, 0, reset_root_rot)
        engine.set_root_vel(reset_ids, 0, 0.0)
        engine.set_root_ang_vel(reset_ids, 0, torch.zeros_like(reset_root_pos))
        engine.set_dof_vel(reset_ids, 0, 0.0)
        if not torch.allclose(
            engine.get_root_pos(0)[reset_ids],
            reset_root_pos,
            atol=1e-6,
            rtol=0.0,
        ):
            raise AssertionError("staged root position was not visible before step")
        if not torch.equal(
            engine.get_root_vel(0)[reset_ids], torch.zeros_like(reset_root_pos)
        ):
            raise AssertionError("scalar root velocity did not broadcast on CUDA")

        num_bodies = engine.get_obj_num_bodies(0)
        reset_body_pos = (
            torch.tensor([0.25, -0.5, 1.25], dtype=torch.float32, device=device)
            .reshape(1, 1, 3)
            .expand(1, num_bodies, 3)
            .contiguous()
        )
        engine.set_body_pos(reset_ids, 0, reset_body_pos)
        observed = engine.get_body_pos(0)[reset_ids]
        if not torch.allclose(observed, reset_body_pos, atol=1e-6, rtol=0.0):
            raise AssertionError(
                f"MimicKit reset body pose was stale before the physics step: {observed.detach().cpu().tolist()}"
            )

        peak_force = 0.0
        for _ in range(240):
            engine.step()
            forces = engine.get_ground_contact_forces(0)
            peak_force = max(
                peak_force,
                float(torch.linalg.vector_norm(forces, dim=-1).max().item()),
            )
            if peak_force > 0.1:
                break
        if peak_force <= 0.1:
            raise AssertionError(
                "MimicKit GPU ground-contact force remained zero after impact"
            )
        sensor = engine._contact_sensors[0]
        body_indices = torch.as_tensor(
            engine._contact_body_indices[0],
            dtype=torch.long,
            device=device,
        )
        expected_forces = sensor.force.index_select(1, body_indices)
        if not torch.allclose(
            engine.get_ground_contact_forces(0),
            expected_forces,
            atol=1e-6,
            rtol=0.0,
        ):
            raise AssertionError(
                "MimicKit ground-contact force did not follow logical body order"
            )

        print(
            f"PASS: MimicKit GPU reset pose and ground contact peak_force={peak_force:.6f}"
        )
    finally:
        engine.release()


if __name__ == "__main__":
    main()
