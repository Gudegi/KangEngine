"""Velocity writes change motion without invoking reset side effects."""

import numpy as np
import pytest
import torch

import kangengine as ke


@pytest.fixture(params=["cpu", "cuda"])
def world(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    result = ke.sim.KangSimWorld(
        num_envs=3, sim_device=request.param, state_device="cpu", add_ground=False,
    )
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", mass=2., inertia_diagonal=(1., 1., 1.))
    builder.add_link(
        "arm", parent="root", position=(1., 0., 0.), mass=1.,
        inertia_diagonal=(1., 1., 1.),
        joint=ke.physics.ArticulationBuilder.RevoluteJoint(),
    )
    cloner = ke.sim.GridCloner(result, spacing=10.)
    cloner.add_articulation(builder.build(), obj_id=0, config=ke.physics.ArticulationConfig.free_base())
    cloner.add_rigid(ke.asset.RigidDesc(), obj_id=1)
    for record in result.get_articulation_batch().records:
        record.articulation.set_dof_state([.3], [.2])
    if request.param == "cuda":
        result.init_gpu_system()
    result.refresh()
    try:
        yield result
    finally:
        result.release()


def snapshot(world, obj_id):
    world.refresh()
    state = world.get_object_batch(obj_id=obj_id).data
    return {name: getattr(state, name).clone() for name in (
        "root_pos", "root_quat_xyzw", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
    )}


@pytest.mark.parametrize("obj_id", [0, 1])
def test_velocity_preserves_state_commands_and_force(world, obj_id):
    before = snapshot(world, obj_id)
    other = snapshot(world, 1 - obj_id)
    world.set_body_force([2, 0], obj_id, 0, [[3., 0., 0.], [4., 0., 0.]])
    forces = {key: value.copy() for key, value in world.body_forces.items()}
    if obj_id == 0:
        world.set_cmd(None, 0, [[.5], [.6], [.7]], mode=ke.sim.ControlMode.POS)
        world.apply_commands()
    commands = dict(world.commands)
    batches = dict(world.batch_commands)
    linear = torch.tensor([[1., 2., 3.], [-2., 1., 0.]], device=world.sim_device)
    angular = torch.tensor([[.1, .2, .3], [.4, .5, .6]], device=world.sim_device)

    world.set_root_velocity([2, 0], obj_id, linear, angular)

    after = snapshot(world, obj_id)
    for name in ("root_pos", "root_quat_xyzw", "dof_pos", "dof_vel"):
        torch.testing.assert_close(after[name], before[name])
    torch.testing.assert_close(after["root_vel"][[2, 0]], linear.cpu())
    torch.testing.assert_close(after["root_ang_vel"][[2, 0]], angular.cpu())
    torch.testing.assert_close(after["root_vel"][1], before["root_vel"][1])
    for name, value in other.items():
        torch.testing.assert_close(snapshot(world, 1 - obj_id)[name], value)
    assert not world._pending_reset_keys
    assert all(world.commands[key] is value for key, value in commands.items())
    assert all(world.batch_commands[key] is value for key, value in batches.items())
    for key, value in forces.items():
        np.testing.assert_array_equal(world.body_forces[key], value)
    if obj_id == 0 and world.sim_device.type == "cuda":
        targets = world.get_gpu_articulation_target_joint_positions().cpu()
        torch.testing.assert_close(targets[:, :1], torch.tensor([[.5], [.6], [.7]]))


@pytest.mark.parametrize("obj_id", [0, 1])
def test_record_batch_empty_and_pending_reset(world, obj_id):
    batch = world.get_object_batch(env_ids=[2, 0], obj_id=obj_id)
    velocity = torch.tensor([1., 0., 0.], device=world.sim_device)
    zero = torch.zeros(3, device=world.sim_device)
    assert batch.set_root_velocity(None, velocity, zero) is batch
    assert batch.first.set_root_velocity(None, velocity * 2, zero) is batch.first
    before = snapshot(world, obj_id)
    batch.set_root_velocity([], velocity, zero)
    torch.testing.assert_close(snapshot(world, obj_id)["root_vel"], before["root_vel"])
    with pytest.raises(KeyError, match="outside this view"):
        batch.set_root_velocity([1], velocity, zero)

    # A velocity write does not flush or discard an explicitly queued reset.
    pos = torch.tensor([8., 9., 10.], device=world.sim_device)
    quat = torch.tensor([0., 0., 0., 1.], device=world.sim_device)
    world.set_root_state(2, obj_id, pos, quat)
    world.set_root_velocity([2], obj_id, velocity, zero)
    assert (2, obj_id) in world._pending_reset_keys
    world.apply_resets()
    after = snapshot(world, obj_id)
    torch.testing.assert_close(after["root_pos"][2], pos.cpu())
    torch.testing.assert_close(after["root_vel"][2], zero.cpu())
    world.set_root_velocity([2], obj_id, velocity, zero)
    torch.testing.assert_close(snapshot(world, obj_id)["root_vel"][2], velocity.cpu())


@pytest.mark.parametrize("obj_id", [0, 1])
def test_cuda_selection_and_stream_without_download(world, obj_id, monkeypatch):
    if world.sim_device.type != "cuda":
        pytest.skip("CUDA stream contract")
    before = snapshot(world, obj_id)
    batch = world.get_object_batch(obj_id=obj_id)
    stream = torch.cuda.Stream()

    def reject_download(*args, **kwargs):
        raise AssertionError("velocity write downloaded a CUDA tensor")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", reject_download)
        guard.setattr(torch.Tensor, "numpy", reject_download)
        guard.setattr(torch.Tensor, "tolist", reject_download)
        guard.setattr(torch.Tensor, "item", reject_download)
        with torch.cuda.stream(stream):
            ids = torch.tensor([2, 0], device="cuda")
            # Non-contiguous input must work without keeping temporary buffers alive.
            linear = torch.arange(12., device="cuda").reshape(2, 6)[:, ::2]
            batch.set_root_velocity(ids, linear, torch.zeros(3, device="cuda"))
            batch.set_root_velocity(ids[:1], torch.ones(3, device="cuda"), torch.zeros(3, device="cuda"))
        del linear, ids
    torch.cuda.current_stream().wait_stream(stream)
    after = snapshot(world, obj_id)
    torch.testing.assert_close(after["root_vel"][2], torch.ones(3))
    torch.testing.assert_close(after["root_vel"][0], torch.tensor([6., 8., 10.]))
    torch.testing.assert_close(after["root_vel"][1], before["root_vel"][1])
    torch.testing.assert_close(after["root_pos"], before["root_pos"])


def test_cpu_native_rejects_non_dynamic_targets():
    world = ke.sim.KangSimWorld(num_envs=1, add_ground=False)
    try:
        rigid = world.add_rigid(ke.asset.RigidDesc(), kinematic=True)
        with pytest.raises(ValueError, match="kinematic"):
            rigid.set_root_velocity(None, [1., 0., 0.], [0., 0., 0.])
        builder = ke.physics.ArticulationBuilder()
        builder.add_link("root", mass=1., inertia_diagonal=(1., 1., 1.))
        robot = world.add_articulation(builder.build(), obj_id=1, config=ke.physics.ArticulationConfig.fixed_base())
        with pytest.raises(ValueError, match="floating-base"):
            robot.set_root_velocity(None, [1., 0., 0.], [0., 0., 0.])
    finally:
        world.release()


@pytest.mark.parametrize("obj_id", [0, 1])
def test_velocity_changes_subsequent_motion(world, obj_id):
    before = snapshot(world, obj_id)
    linear = torch.tensor([1., 0., 0.], device=world.sim_device)
    zero = torch.zeros(3, device=world.sim_device)
    world.set_root_velocity([1], obj_id, linear, zero)
    world.step()
    after = snapshot(world, obj_id)
    assert after["root_pos"][1, 0] > before["root_pos"][1, 0] + world.sim_dt * .5


def test_velocity_input_device_contract(world):
    if world.sim_device.type == "cuda":
        with pytest.raises(ValueError, match="float32 tensors"):
            world.set_root_velocity([0], 0, torch.ones(3), torch.zeros(3))
    else:
        with pytest.raises(ValueError, match="shape"):
            world.get_rigid(env_id=0, obj_id=1).rigid.set_root_velocity([1., 2.], [0., 0., 0.])
        if torch.cuda.is_available():
            with pytest.raises(ValueError, match="download"):
                world.set_root_velocity([0], 0, torch.ones(3, device="cuda"), torch.zeros(3, device="cuda"))
