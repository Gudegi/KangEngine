"""Host property updates must reach the running GPU solver without resetting it."""

import kangengine as ke
import pytest
import torch

from test_mass_properties import make_articulations, make_rigids


@pytest.fixture
def world():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    result = ke.sim.KangSimWorld(
        num_envs=3, sim_device="cuda", state_device="cpu", add_ground=False, sim_dt=.01,
    )
    try:
        yield result
    finally:
        result.release()


@pytest.mark.parametrize("articulated", [False, True])
@pytest.mark.parametrize("torque", [False, True])
def test_runtime_mass_and_inertia_change_solver_response(world, articulated, torque):
    _, batch = make_articulations(world, child=False) if articulated else make_rigids(world)
    world.init_gpu_system()
    world.step(refresh=False)
    setter = world.set_body_mass_space_inertias if torque else world.set_body_masses
    gpu = world.gpu_system
    if articulated:
        buffer = (world.get_gpu_articulation_link_torques() if torque
                  else world.get_gpu_articulation_link_forces())[:, 0]
        apply = lambda: world.apply_gpu_articulation_link_wrenches(forces=not torque, torques=torque)
    else:
        buffer = (gpu.rigid_torque() if torque else gpu.rigid_force()).torch()
        apply = gpu.apply_rigid_torque if torque else gpu.apply_rigid_force

    for factors in ([2., 4.], [4., 2.]):
        values = [[[v, v, v]] for v in factors] if torque else [[v] for v in factors]
        setter([2, 0], batch.obj_id, values)
        # PhysX uploads host properties on simulate; Direct GPU force writes
        # convert forces to accelerations using the properties already on GPU.
        world.step(refresh=False)
        world.set_root_velocity(None, batch.obj_id, torch.zeros(3, device="cuda"), torch.zeros(3, device="cuda"))
        buffer.zero_()
        buffer[:, 2 if torque else 0] = 4.
        apply()
        world.step(refresh=False)
        state = world.state.gpu
        velocity = (state.get_root_ang_vel(batch.obj_id)[:, 2] if torque
                    else state.get_root_vel(batch.obj_id)[:, 0]).cpu()
        torch.testing.assert_close(velocity[[2, 0]], .04 / torch.tensor(factors), rtol=1e-3, atol=1e-6)
        nominal = 1. if torque else 2.
        assert velocity[1].item() == pytest.approx(.04 / nominal, rel=1e-3)


@pytest.mark.parametrize("articulated", [False, True])
def test_runtime_com_rotation_preserves_moving_actor(world, articulated):
    _, batch = make_articulations(world) if articulated else make_rigids(world)
    world.init_gpu_system()
    world.set_root_velocity(None, batch.obj_id, torch.tensor([1., 0., 0.], device="cuda"), torch.zeros(3, device="cuda"))
    for _ in range(10):
        world.step()
    nominal = world.get_body_local_com_poses(None, batch.obj_id)

    for offset in (.1, -.1):
        before_pos = batch.get_body_pos().clone()
        before_vel = batch.get_root_vel().clone()
        changed = nominal[2:3].clone()
        changed[..., 0] += offset
        changed[..., 3:] = torch.tensor([0., 0., .6, .8])
        world.set_body_local_com_poses([2], batch.obj_id, changed)
        world.step()
        # All instances were translated copies; changing COM must not teleport them.
        displacement = batch.get_body_pos() - before_pos
        torch.testing.assert_close(displacement[2], displacement[1], atol=2e-5, rtol=1e-3)
        torch.testing.assert_close(batch.get_root_vel()[2], batch.get_root_vel()[1], atol=2e-5, rtol=1e-3)
        assert batch.get_root_vel()[2, 0] > .9 * before_vel[2, 0]
        torch.testing.assert_close(world.get_body_local_com_poses([0, 1], batch.obj_id), nominal[:2])


def test_runtime_com_changes_pendulum_response(world):
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", mass=1., inertia_diagonal=(1., 1., 1.))
    builder.add_link(
        "pendulum", parent="root", mass=1., inertia_diagonal=(1., 1., 1.),
        joint=ke.physics.ArticulationBuilder.RevoluteJoint(axis=(0., 1., 0.)),
    )
    batch = ke.sim.GridCloner(world, spacing=10).add_articulation(
        builder.build(), config=ke.physics.ArticulationConfig.fixed_base(),
    )
    world.init_gpu_system()
    world.step()
    poses = world.get_body_local_com_poses([2], 0)[:, 1:2].clone()
    poses[..., 0] = .5
    world.set_body_local_com_poses([2], 0, poses, body_ids=[1])
    for _ in range(10):
        world.step()
    assert abs(batch.data.dof_vel[2, 0]) > .05
    torch.testing.assert_close(batch.data.dof_vel[:2], torch.zeros(2, 1), atol=1e-5, rtol=0)


@pytest.mark.parametrize("support", ["static", "kinematic", "articulation", "dynamic"])
def test_runtime_material_changes_contact_friction(world, support):
    # Modify an existing contact on only one support; share the same source geometry.
    box = ke.asset.RigidDesc(collision_geoms=[ke.asset.CollisionGeomDesc(
        type=ke.asset.CollisionGeomDescType.BOX, size=(1., 1., .2),
        position=(0., 0., 0.), quat_xyzw=(0., 0., 0., 1.),
    )])
    cloner = ke.sim.GridCloner(world, spacing=10.)
    if support == "articulation":
        builder = ke.physics.ArticulationBuilder()
        builder.add_link("root", inertia_diagonal=(1., 1., 1.), shapes=[
            ke.physics.ArticulationBuilder.BoxShape((1., 1., .2)),
        ])
        floor = cloner.add_articulation(builder.build(), config=ke.physics.ArticulationConfig.fixed_base())
    else:
        floor = cloner.add_rigid(box, static=support in ("static", "dynamic"), kinematic=support == "kinematic")
    slider = cloner.add_rigid(box, obj_id=1, initial_root_pos=(0., 0., .4))
    target = slider if support == "dynamic" else floor
    low = ke.physics.PhysicsMaterialDesc(0., 0., 0., friction_combine_mode=ke.physics.MaterialCombineMode.MIN)
    high = ke.physics.PhysicsMaterialDesc(1., 1., 0., friction_combine_mode=ke.physics.MaterialCombineMode.MIN)
    target.set_collision_material(None, low)
    world.init_gpu_system()
    for _ in range(20):
        world.step(refresh=False)
    target.set_collision_material([2], high)
    world.set_root_velocity(None, 1, torch.tensor([.5, 0., 0.], device="cuda"), torch.zeros(3, device="cuda"))
    for _ in range(20):
        world.step(refresh=False)
    velocity = world.state.gpu.get_root_vel(1)[:, 0].cpu()
    assert abs(velocity[2]) < .1
    assert torch.all(velocity[:2] > .4)
    target.set_collision_material([2], low)
    world.set_root_velocity(None, 1, torch.tensor([.5, 0., 0.], device="cuda"), torch.zeros(3, device="cuda"))
    for _ in range(10):
        world.step(refresh=False)
    assert torch.all(world.state.gpu.get_root_vel(1)[:, 0] > .4)
    assert world.physics.num_cached_materials() == 1  # High friction reuses the default.


def test_runtime_restitution_changes_bounce(world):
    shape = ke.asset.CollisionGeomDesc(
        type=ke.asset.CollisionGeomDescType.SPHERE, size=(.2, .2, .2),
        position=(0., 0., 0.), quat_xyzw=(0., 0., 0., 1.),
    )
    batch = ke.sim.GridCloner(world, spacing=10.).add_rigid(
        ke.asset.RigidDesc(collision_geoms=[shape]), initial_root_pos=(0., 0., 1.),
    )
    world.physics.add_default_ground()
    world.init_gpu_system()
    world.step()
    bounce = ke.physics.PhysicsMaterialDesc(
        0., 0., 1., restitution_combine_mode=ke.physics.MaterialCombineMode.MAX,
    )
    batch.set_collision_material([2], bounce)
    peak_upward = torch.zeros(3)
    for _ in range(65):
        world.step(refresh=False)
        peak_upward = torch.maximum(peak_upward, world.state.gpu.get_root_vel(0)[:, 2].cpu())
    assert peak_upward[2] > 2.
    assert torch.all(peak_upward[:2] < .5)
