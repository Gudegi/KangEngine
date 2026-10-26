"""Instance mass properties must affect physics without changing shared assets."""

import numpy as np
import pytest
import torch

import kangengine as ke


@pytest.fixture
def world():
    result = ke.sim.KangSimWorld(num_envs=3, state_device="cpu", sim_dt=0.01)
    try:
        yield result
    finally:
        result.release()


def make_articulations(world, *, child=True):
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", mass=2.0, inertia_diagonal=(1, 1, 1))
    if child:
        builder.add_link(
            "child", parent="root", position=(1, 0, 0), mass=3.0,
            inertia_diagonal=(2, 2, 2), com=(0.1, 0, 0),
            joint=ke.physics.ArticulationBuilder.RevoluteJoint(),
        )
    source = builder.build()
    config = ke.physics.ArticulationConfig.free_base()
    config.root_linear_damping = 0.0
    config.root_angular_damping = 0.0
    robots = ke.sim.GridCloner(world, spacing=10).add_articulation(source, config=config)
    return source, robots


def make_rigids(world, *, static=False):
    source = ke.asset.RigidDesc(
        collision_geoms=[
            ke.asset.CollisionGeomDesc(
                type=ke.asset.CollisionGeomDescType.BOX,
                size=(0.2, 0.2, 0.2), position=(x, 0, 0),
                quat_xyzw=(0, 0, 0, 1),
            )
            for x in (-0.5, 0.5)
        ],
        inertial=ke.asset.InertialDesc(2.0, diag_inertia=(1, 1, 1), com=(0, 0, 0)),
    )
    batch = ke.sim.GridCloner(world, spacing=10).add_rigid(source, obj_id=1, static=static)
    return source, batch


def test_selected_link_and_environment_metadata(world):
    source, robots = make_articulations(world)
    nominal = world.get_body_masses(obj_id=0)
    inertias = world.get_body_mass_space_inertias(obj_id=0)
    world.set_body_masses([2, 0], 0, [[6.0], [4.5]], body_ids=[1])
    torch.testing.assert_close(world.get_body_masses(obj_id=0), torch.tensor([[2., 4.5], [2., 3.], [2., 6.]]))
    torch.testing.assert_close(world.get_body_mass_space_inertias(obj_id=0), inertias)
    torch.testing.assert_close(world.state.get_obj_body_masses(0), torch.tensor([2., 4.5]))
    assert world.state.calc_obj_mass(2, 0) == pytest.approx(8.0)
    assert source.inertials[1].mass == 3.0
    assert robots.records[0].articulation.template is robots.records[1].articulation.template
    assert robots.records[0].source_data.inertials[1].mass == 3.0
    torch.testing.assert_close(nominal, torch.tensor([[2., 3.]]).repeat(3, 1))
    torch.testing.assert_close(world.get_body_masses([2, 0], 0)[:, 1], torch.tensor([6., 4.5]))


def test_compound_rigid_has_one_mass_and_preserves_source(world):
    source, bodies = make_rigids(world)
    assert bodies.num_bodies == 1
    assert bodies.num_collision_shapes == 2
    assert world.get_body_masses(obj_id=1).shape == (3, 1)
    world.set_body_masses([1], 1, [[5.0]])
    world.set_body_mass_space_inertias([1], 1, [[[2.0, 3.0, 4.0]]])
    torch.testing.assert_close(world.get_body_masses(obj_id=1), torch.tensor([[2.], [5.], [2.]]))
    torch.testing.assert_close(world.get_body_mass_space_inertias(obj_id=1)[1], torch.tensor([[2., 3., 4.]]))
    torch.testing.assert_close(world.state.record(1, 1).cache.body_masses, torch.tensor([5.]))
    assert source.inertial.mass == 2.0
    assert all(record.source_data is source for record in bodies.records)
    with pytest.raises(ValueError, match="out of range"):
        world.set_body_masses([0], 1, [[3.0]], body_ids=[1])


@pytest.mark.parametrize("property_name, values", [
    ("masses", [[9.], [6.]]),
    ("mass_space_inertias", [[[9., 8., 7.]], [[6., 5., 4.]]]),
    ("local_com_poses", [[[.1, .2, .3, 0., 0., 0., 1.]],
                         [[.4, .5, .6, 0., 0., 0., 1.]]]),
])
def test_property_selection_uses_environment_then_object(world, property_name, values):
    make_articulations(world)
    make_rigids(world)
    getter = getattr(world, f"get_body_{property_name}")
    setter = getattr(world, f"set_body_{property_name}")
    default_object = getter(env_ids=None, obj_id=0)
    expected = torch.tensor(values)
    setter([2, 0], 1, expected)

    # Distinct objects and reordered environments expose swapped positional arguments.
    torch.testing.assert_close(getter([2, 0], 1), expected)
    torch.testing.assert_close(getter(2, 1), expected[:1])
    torch.testing.assert_close(getter(env_ids=[2, 0], obj_id=1), expected)
    torch.testing.assert_close(getter(None, 1), getter(obj_id=1))
    torch.testing.assert_close(getter(), default_object)
    torch.testing.assert_close(getter(2), default_object[2:3])
    torch.testing.assert_close(getter(1, 0), default_object[1:2])


@pytest.mark.parametrize("articulated", [False, True])
def test_com_pose_preserves_body_pose_and_unselected_instances(world, articulated):
    source, bodies = make_articulations(world) if articulated else make_rigids(world)
    obj_id = bodies.obj_id
    world.step(substeps=0)
    positions = bodies.get_body_pos().clone()
    rotations = bodies.get_body_quat_xyzw().clone()
    nominal = world.get_body_local_com_poses(obj_id=obj_id)
    inertias = world.get_body_mass_space_inertias(obj_id=obj_id)
    body_id = 1 if articulated else 0
    changed = nominal[1:2, body_id:body_id + 1].clone()
    changed[:, :, :3] += torch.tensor([0.02, -0.03, 0.01])
    changed[:, :, 3:] = torch.tensor([0., 0., 2**-0.5, 2**-0.5])
    world.set_body_local_com_poses([1], obj_id, changed, body_ids=[body_id])
    world.step(substeps=0)
    torch.testing.assert_close(bodies.get_body_pos(), positions)
    torch.testing.assert_close(bodies.get_body_quat_xyzw(), rotations)
    torch.testing.assert_close(world.get_body_local_com_poses(obj_id=obj_id)[[0, 2]], nominal[[0, 2]])
    torch.testing.assert_close(world.get_body_local_com_poses(obj_id=obj_id)[1:2, body_id:body_id + 1], changed)
    torch.testing.assert_close(world.get_body_mass_space_inertias(obj_id=obj_id), inertias)


@pytest.mark.parametrize("articulated", [False, True])
def test_mass_change_between_steps_changes_force_response(world, articulated):
    _, bodies = make_articulations(world, child=False) if articulated else make_rigids(world)
    world.step()  # Exercise a running world at a completed-step boundary.
    world.set_body_masses([0, 1], bodies.obj_id, [[2.0], [4.0]])
    for record in bodies.records[:2]:
        if articulated:
            record.articulation.add_link_force(0, [4., 0., 0.])
        else:
            record.rigid.add_force([4., 0., 0.])
    world.step(apply_commands=False)
    velocity = bodies.get_root_vel()[:2, 0]
    torch.testing.assert_close(velocity, torch.tensor([0.02, 0.01]), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("articulated", [False, True])
def test_inertia_change_changes_torque_response(world, articulated):
    _, bodies = make_articulations(world, child=False) if articulated else make_rigids(world)
    world.step()
    world.set_body_mass_space_inertias(
        [0, 1], bodies.obj_id, [[[1., 1., 1.]], [[2., 2., 2.]]]
    )
    for record in bodies.records[:2]:
        actor = record.articulation if articulated else record.rigid
        position = np.asarray(actor.get_root_position()) + [0., 1., 0.]
        if articulated:
            actor.add_link_force_at_position(0, [1., 0., 0.], position)
        else:
            actor.add_force_at_position([1., 0., 0.], position)
    world.step(apply_commands=False)
    angular_velocity = bodies.get_root_ang_vel()[:2, 2]
    torch.testing.assert_close(angular_velocity[0], angular_velocity[1] * 2, rtol=1e-4, atol=1e-6)
    assert abs(angular_velocity[0]) > 0.009


def test_native_validation_precedes_link_mutation(world):
    _, bodies = make_articulations(world)
    actor = bodies.first.articulation
    nominal = world.get_body_masses(obj_id=0)
    with pytest.raises(ValueError, match="positive"):
        actor.set_link_masses([0, 1], [10.0, -1.0])
    torch.testing.assert_close(world.get_body_masses(obj_id=0), nominal)
    with pytest.raises((IndexError, ValueError)):
        actor.set_link_masses([99], [1.0])
    with pytest.raises(ValueError, match="shape"):
        # The binding also checks row width before the native call.
        actor.set_link_mass_space_inertias([0], [[1.0, 2.0]])

    poses = actor.get_link_local_com_poses()
    invalid = poses.copy()
    invalid[0, 0] += 0.5
    invalid[1, 3:] = 0.0
    with pytest.raises(ValueError, match="unit quaternion"):
        actor.set_link_local_com_poses([0, 1], invalid)
    np.testing.assert_array_equal(actor.get_link_local_com_poses(), poses)


def test_invalid_rigid_values_and_static_bodies(world):
    _, bodies = make_rigids(world)
    actor = bodies.first.rigid
    for mass in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            actor.set_mass(mass)
    with pytest.raises(ValueError):
        actor.set_mass_space_inertia([1., -1., 1.])
    with pytest.raises(ValueError, match="unit quaternion"):
        actor.set_local_com_pose([0., 0., 0., 0., 0., 0., 2.])
    with pytest.raises(ValueError, match="shape"):
        world.set_body_masses([0, 1], 1, [[2.]])
    assert world.get_body_masses([], 1).shape == (0, 1)


def test_static_rigid_has_no_mass_properties(world):
    make_rigids(world, static=True)
    with pytest.raises(ValueError, match="dynamic/kinematic"):
        world.get_body_masses(obj_id=1)


def test_host_api_rejects_device_inputs(world):
    make_rigids(world)
    with pytest.raises(ValueError, match="CPU"):
        world.set_body_masses([0], 1, torch.empty((1, 1), device="meta"))
    with pytest.raises(ValueError, match="CPU"):
        world.get_body_masses(torch.empty(1, dtype=torch.long, device="meta"), 1)
