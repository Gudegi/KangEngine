"""Validate batched ContactSensor aggregation on PhysX GPU contact data."""

from __future__ import annotations

from pathlib import Path

import kangengine as ke


def main():
    world = ke.KangSimWorld(num_envs=2, sim_device="cuda", add_ground=False)
    try:
        ball_xml = Path(ke.__file__).resolve().parent / "assets" / "objects" / "ball.xml"
        data = world.load_mjcf(str(ball_xml))

        positions = (
            ([0.0, 0.0, 1.0], [0.27, 0.0, 1.0]),
            ([5.0, 0.0, 1.0], [8.0, 0.0, 1.0]),
        )
        for env_id, env_positions in enumerate(positions):
            for obj_id, pos in enumerate(env_positions):
                record = world.add_rigid(
                    data,
                    env_id=env_id,
                    obj_id=obj_id,
                    name=f"ball_{obj_id}",
                    pos=pos,
                    density=10.0,
                )
                if env_id == 0:
                    velocity = [0.5 if obj_id == 0 else -0.5, 0.0, 0.0]
                    record.rigid.set_root_state(
                        pos,
                        [0.0, 0.0, 0.0, 1.0],
                        velocity,
                        [0.0, 0.0, 0.0],
                    )

        tracked = world.get_rigid_view([0, 1], obj_id=0)
        sensor = tracked.add_contact_sensor(body_ids=[0], name="ball_contact")
        force_sensor = tracked.add_force_sensor(body_ids=[0], name="ball_force")
        other = world.get_rigid_view([0, 1], obj_id=1)
        other_sensor = other.add_contact_sensor(
            body_ids=[0], name="other_ball_contact"
        )
        world.init_gpu_system(cuda_device_id=0)
        world.step(refresh=False)

        import torch

        torch.cuda.synchronize(0)
        if sensor.contact_count.device.type != "cuda":
            raise AssertionError("ContactSensor output was copied off CUDA")
        if tuple(sensor.contact_count.shape) != (2, 1):
            raise AssertionError(
                f"unexpected ContactSensor shape {tuple(sensor.contact_count.shape)}"
            )
        result = sensor.contact_count.detach().cpu().tolist()
        other_result = other_sensor.contact_count.detach().cpu().tolist()
        if result[0][0] <= 0:
            raise AssertionError("overlapping env did not report a contact")
        if result[1][0] != 0:
            raise AssertionError("separated env reported a false contact")
        if not bool(sensor.in_contact[0, 0]) or bool(sensor.in_contact[1, 0]):
            raise AssertionError("ContactSensor in_contact mask mismatch")
        if other_result != result:
            raise AssertionError(
                f"packed multi-sensor aggregation mismatch: {result} != {other_result}"
            )
        impulse = sensor.net_impulse[0, 0]
        other_impulse = other_sensor.net_impulse[0, 0]
        if float(torch.linalg.vector_norm(impulse)) <= 0.0:
            raise AssertionError("impacting env did not report net normal impulse")
        if not torch.allclose(impulse, -other_impulse, rtol=1e-4, atol=1e-7):
            raise AssertionError("contact sensor impulses violate action/reaction")
        if torch.count_nonzero(sensor.net_impulse[1]).item() != 0:
            raise AssertionError("separated env reported a false impulse")
        if not torch.allclose(force_sensor.impulse, sensor.net_impulse):
            raise AssertionError("ForceSensor impulse does not share contact semantics")
        expected_force = force_sensor.impulse / world.sim_dt
        if not torch.allclose(force_sensor.force, expected_force):
            raise AssertionError("ForceSensor did not convert impulse using sim_dt")

        print(
            "PASS: ContactSensor packed GPU aggregation "
            f"counts={result}, impulse={impulse.detach().cpu().tolist()}"
        )
    finally:
        world.release()


if __name__ == "__main__":
    main()
