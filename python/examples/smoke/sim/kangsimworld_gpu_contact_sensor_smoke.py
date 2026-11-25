"""Validate batched ContactSensor aggregation on PhysX GPU contact data."""

from __future__ import annotations

from pathlib import Path

import kangengine as ke
import torch


def main():
    world = ke.sim.KangSimWorld(num_envs=2, sim_device="cuda", add_ground=False)
    try:
        ball_xml = (
            Path(ke.__file__).resolve().parent / "assets" / "objects" / "ball.xml"
        )
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

        tracked = world.get_rigid_batch([0, 1], obj_id=0)
        sensor = tracked.add_contact_sensor(body_ids=[0], name="ball_contact")
        force_sensor = tracked.add_force_sensor(body_ids=[0], name="ball_force")
        other = world.get_rigid_batch([0, 1], obj_id=1)
        other_sensor = other.add_contact_sensor(body_ids=[0], name="other_ball_contact")
        world.init_gpu_system(cuda_device_id=0)
        world.step(refresh=False)

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
        points = sensor.contact_points()
        if points.position_w.device.type != "cuda":
            raise AssertionError("packed contact points were copied off CUDA")
        if points.count <= 0 or torch.any(points.environment != 0):
            raise AssertionError("sensor contact-point selection mismatch")
        point_impulse = points.normal_impulse_w.sum(dim=0)
        if not torch.allclose(point_impulse, impulse, rtol=1e-4, atol=1e-7):
            raise AssertionError("packed contact points do not sum to net impulse")
        if not torch.isfinite(points.position_w).all():
            raise AssertionError("packed contact positions contain non-finite values")
        wrench = points.normal_impulse_wrench_about(torch.zeros(3, device="cuda:0"))
        expected_angular_impulse = torch.linalg.cross(
            points.position_w, points.normal_impulse_w, dim=-1
        )
        if not torch.allclose(
            wrench[:, :3], points.normal_impulse_w
        ) or not torch.allclose(wrench[:, 3:], expected_angular_impulse):
            raise AssertionError("contact wrench reference-point calculation mismatch")
        if not torch.allclose(
            torch.linalg.vector_norm(points.normal_w, dim=1),
            torch.ones(points.count, device=points.normal_w.device),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise AssertionError("packed contact normals are not unit length")
        if not torch.allclose(force_sensor.impulse, sensor.net_impulse):
            raise AssertionError("ForceSensor impulse does not share contact semantics")
        expected_force = force_sensor.impulse / world.sim_dt
        if not torch.allclose(force_sensor.force, expected_force):
            raise AssertionError("ForceSensor did not convert impulse using sim_dt")

        rotations = torch.zeros((2, 4), dtype=torch.float32, device="cuda:0")
        rotations[:, 3] = 1.0
        zeros3 = torch.zeros((2, 3), dtype=torch.float32, device="cuda:0")
        separated_left = torch.tensor(
            [[0.0, 3.0, 1.0], [5.0, 3.0, 1.0]],
            dtype=torch.float32,
            device="cuda:0",
        )
        separated_right = torch.tensor(
            [[2.0, 3.0, 1.0], [7.0, 3.0, 1.0]],
            dtype=torch.float32,
            device="cuda:0",
        )
        tracked.set_root_state(
            None,
            separated_left,
            rotations,
            linear_velocity=zeros3,
            angular_velocity=zeros3,
        )
        other.set_root_state(
            None,
            separated_right,
            rotations,
            linear_velocity=zeros3,
            angular_velocity=zeros3,
        )
        world.step(substeps=0, refresh=False, apply_commands=False)
        torch.cuda.synchronize(0)
        if torch.count_nonzero(sensor.contact_count).item() != 0:
            raise AssertionError("GPU reset left stale contact counts")
        if torch.count_nonzero(sensor.net_impulse).item() != 0:
            raise AssertionError("GPU reset left stale contact impulses")

        impact_left = torch.tensor(
            [[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]],
            dtype=torch.float32,
            device="cuda:0",
        )
        impact_right = torch.tensor(
            [[0.20, 0.0, 1.0], [5.20, 0.0, 1.0]],
            dtype=torch.float32,
            device="cuda:0",
        )
        left_velocity = torch.zeros_like(impact_left)
        right_velocity = torch.zeros_like(impact_right)
        left_velocity[:, 0] = 0.5
        right_velocity[:, 0] = -0.5
        tracked.set_root_state(
            None,
            impact_left,
            rotations,
            linear_velocity=left_velocity,
            angular_velocity=zeros3,
        )
        other.set_root_state(
            None,
            impact_right,
            rotations,
            linear_velocity=right_velocity,
            angular_velocity=zeros3,
        )
        world.step(substeps=0, refresh=False, apply_commands=False)
        torch.cuda.synchronize(0)
        if torch.count_nonzero(sensor.contact_count).item() != 0:
            raise AssertionError("GPU impact reset-only frame reported stale contacts")
        world.step(refresh=False)
        torch.cuda.synchronize(0)
        reset_counts = sensor.contact_count.detach().cpu().tolist()
        reset_impulse = sensor.net_impulse[:, 0]
        if min(row[0] for row in reset_counts) <= 0:
            raise AssertionError(
                f"Direct GPU reset impact did not report contacts: {reset_counts}"
            )
        if float(torch.linalg.vector_norm(reset_impulse, dim=1).max()) <= 0.0:
            raise AssertionError("Direct GPU reset impact did not report impulse")

        print(
            "PASS: ContactSensor packed GPU aggregation "
            f"counts={result}, impulse={impulse.detach().cpu().tolist()}, "
            f"reset_counts={reset_counts}"
        )
    finally:
        world.release()


if __name__ == "__main__":
    main()
