"""Collision materials are instance-local and cached by coefficients and modes."""

import kangengine as ke
import pytest


Mode = ke.physics.MaterialCombineMode


@pytest.fixture
def world():
    result = ke.sim.KangSimWorld(num_envs=2, state_device="cpu")
    try:
        yield result
    finally:
        result.release()


def rigid_source():
    return ke.asset.RigidDesc(name="box", collision_geoms=[
        ke.asset.CollisionGeomDesc(
            type=ke.asset.CollisionGeomDescType.BOX, size=(0.2, 0.2, 0.2),
            position=(x, 0, 0), quat_xyzw=(0, 0, 0, 1),
        ) for x in (-0.5, 0.5)
    ])


def articulation_source():
    builder = ke.physics.ArticulationBuilder()
    shape = ke.physics.ArticulationBuilder.BoxShape((0.1, 0.1, 0.1))
    builder.add_link("root", inertia_diagonal=(1, 1, 1), shapes=[shape, shape])
    builder.add_link("child", parent="root", position=(1, 0, 0),
                     inertia_diagonal=(1, 1, 1), shapes=[shape])
    return builder.build()


@pytest.mark.parametrize("static", [False, True])
def test_rigid_partial_overrides_preserve_other_shapes_and_instances(world, static):
    source = rigid_source()
    batch = ke.sim.GridCloner(world, spacing=5).add_rigid(source, obj_id=0, static=static)
    first, second = batch.records
    high = ke.physics.PhysicsMaterialDesc(0.8, 0.7, 0.2, friction_combine_mode=Mode.MAX)
    low = ke.physics.PhysicsMaterialDesc(0.3, 0.2, 0.1, restitution_combine_mode=Mode.MULTIPLY)
    first.set_collision_material(high)
    override = ke.physics.CollisionMaterialOverride.for_indices(0, 1, low)
    assert first.set_collision_material_overrides([override]) == 1
    assert first.set_collision_material_overrides([]) == 0
    materials = first.rigid.get_collision_materials()
    assert materials[0].as_tuple() == pytest.approx(high.as_tuple())
    assert materials[0].friction_combine_mode == Mode.MAX
    assert materials[1].as_tuple() == pytest.approx(low.as_tuple())
    assert materials[1].restitution_combine_mode == Mode.MULTIPLY
    assert second.rigid.get_collision_materials()[1].as_tuple() == (1, 1, 0)
    assert source.collision_geoms[1].physics_material.as_tuple() == (1, 1, 0)
    materials[0].static_friction = 99
    assert first.rigid.get_collision_materials()[0].static_friction == pytest.approx(0.8)


@pytest.mark.parametrize("mode", ["dynamic", "kinematic", "static"])
def test_rigid_batch_materials_respect_environment_and_shape_selection(world, mode):
    source = rigid_source()
    batch = ke.sim.GridCloner(world, spacing=5).add_rigid(
        source, obj_id=0, static=mode == "static", kinematic=mode == "kinematic",
    )
    high = ke.physics.PhysicsMaterialDesc(0.8, 0.7, 0.2, friction_combine_mode=Mode.MAX)
    low = ke.physics.PhysicsMaterialDesc(0.3, 0.2, 0.1)
    override = ke.physics.CollisionMaterialOverride.for_indices(0, 1, low)

    assert batch.set_collision_material(env_ids=[1], material=high) == 2
    assert batch.set_collision_material_overrides(env_ids=[1], material_overrides=[override]) == 1
    first, second = batch.records
    assert [m.as_tuple() for m in first.rigid.get_collision_materials()] == [(1, 1, 0)] * 2
    materials = second.rigid.get_collision_materials()
    assert materials[0].as_tuple() == pytest.approx(high.as_tuple())
    assert materials[0].friction_combine_mode == Mode.MAX
    assert materials[1].as_tuple() == pytest.approx(low.as_tuple())

    # None selects only the environments in this batch view.
    subset = world.get_rigid_batch(env_ids=[0], obj_id=0)
    assert subset.set_collision_material(env_ids=None, material=high) == 2
    assert subset.set_collision_material_overrides(env_ids=None, material_overrides=[override]) == 1
    assert first.rigid.get_collision_materials()[0].as_tuple() == pytest.approx(high.as_tuple())
    assert first.rigid.get_collision_materials()[1].as_tuple() == pytest.approx(low.as_tuple())
    assert [m.as_tuple() for m in second.rigid.get_collision_materials()] == [m.as_tuple() for m in materials]
    assert batch.set_collision_material(env_ids=[], material=low) == 0
    with pytest.raises(KeyError, match="outside this view"):
        subset.set_collision_material(env_ids=[1], material=low)
    assert all(geom.physics_material.as_tuple() == (1, 1, 0) for geom in source.collision_geoms)


def test_articulation_partial_overrides_preserve_previous_event(world):
    source = articulation_source()
    batch = ke.sim.GridCloner(world, spacing=5).add_articulation(source)
    robot = batch.first.articulation
    first = ke.physics.CollisionMaterialOverride.for_indices(0, 0, (0.3, 0.2, 0.1))
    second = ke.physics.CollisionMaterialOverride.for_body("child", (0.8, 0.7, 0.6))
    assert robot.set_collision_material_overrides(world.physics, [first]) == 1
    assert robot.set_collision_material_overrides(world.physics, [second]) == 1
    actual = robot.get_collision_materials()
    assert actual[0][0].as_tuple() == pytest.approx((0.3, 0.2, 0.1))
    assert actual[0][1].as_tuple() == (1, 1, 0)
    assert actual[1][0].as_tuple() == pytest.approx((0.8, 0.7, 0.6))
    assert batch.records[1].articulation.get_collision_materials()[1][0].as_tuple() == (1, 1, 0)


def test_articulation_record_and_batch_material_interfaces(world):
    source = articulation_source()
    batch = ke.sim.GridCloner(world, spacing=5).add_articulation(source)
    first, second = batch.records
    high = ke.physics.PhysicsMaterialDesc(0.8, 0.7, 0.2, friction_combine_mode=Mode.MAX)
    low = ke.physics.PhysicsMaterialDesc(0.3, 0.2, 0.1)
    child = ke.physics.CollisionMaterialOverride.for_body("child", low)

    assert first.set_collision_material(material=high) == 3
    assert first.set_collision_material_overrides(material_overrides=[child]) == 1
    actual = first.articulation.get_collision_materials()
    assert actual[0][0].as_tuple() == pytest.approx(high.as_tuple())
    assert actual[0][0].friction_combine_mode == Mode.MAX
    assert actual[1][0].as_tuple() == pytest.approx(low.as_tuple())
    assert second.articulation.get_collision_materials()[1][0].as_tuple() == (1, 1, 0)

    assert batch.set_collision_material(env_ids=[1], material=high) == 3
    assert batch.set_collision_material_overrides(env_ids=[1], material_overrides=[child]) == 1
    assert second.articulation.get_collision_materials()[0][1].as_tuple() == pytest.approx(high.as_tuple())
    assert second.articulation.get_collision_materials()[1][0].as_tuple() == pytest.approx(low.as_tuple())

    subset = world.get_articulation_batch(env_ids=[0], obj_id=0)
    assert subset.set_collision_material(env_ids=None, material=low) == 3
    root_shape = ke.physics.CollisionMaterialOverride.for_indices(0, 0, high)
    assert subset.set_collision_material_overrides(env_ids=None, material_overrides=[root_shape]) == 1
    actual = first.articulation.get_collision_materials()
    assert actual[0][0].as_tuple() == pytest.approx(high.as_tuple())
    assert actual[0][1].as_tuple() == pytest.approx(low.as_tuple())
    assert actual[1][0].as_tuple() == pytest.approx(low.as_tuple())
    assert second.articulation.get_collision_materials()[0][1].as_tuple() == pytest.approx(high.as_tuple())
    assert subset.set_collision_material(env_ids=[], material=high) == 0
    assert subset.set_collision_material_overrides(env_ids=None, material_overrides=[]) == 0
    with pytest.raises(KeyError, match="outside this view"):
        subset.set_collision_material_overrides(env_ids=[1], material_overrides=[root_shape])
    assert all(
        geom.physics_material.as_tuple() == (1, 1, 0)
        for geoms in source.collision_geoms.values() for geom in geoms
    )
    assert world.physics.num_cached_materials() == 2


def test_cache_distinguishes_both_modes_and_reuses_buckets(world):
    body = world.add_rigid(rigid_source(), env_id=0, obj_id=0)
    modes = (Mode.AVERAGE, Mode.MIN, Mode.MULTIPLY, Mode.MAX)
    for _ in range(3):
        for friction in modes:
            for restitution in modes:
                material = ke.physics.PhysicsMaterialDesc(
                    friction_combine_mode=friction, restitution_combine_mode=restitution,
                )
                body.set_collision_material(material)
                actual = body.rigid.get_collision_materials()[0]
                assert actual.friction_combine_mode == friction
                assert actual.restitution_combine_mode == restitution
        # One of the sixteen combinations is the world's default material.
        assert world.physics.num_cached_materials() == 15


@pytest.mark.parametrize("values", [(-1, 1, 0), (1, float("nan"), 0), (1, 1, 1.1)])
def test_invalid_material_does_not_replace_shapes(world, values):
    body = world.add_rigid(rigid_source(), env_id=0, obj_id=0)
    with pytest.raises(ValueError):
        body.set_collision_material(ke.physics.PhysicsMaterialDesc(*values))
    assert body.rigid.get_collision_materials()[0].as_tuple() == (1, 1, 0)


def test_restitution_combine_changes_measured_bounce():
    def bounce(mode):
        config = ke.physics.PhysicsConfig.z_up()
        config.dt = 0.005
        config.restitution = 0.5
        config.restitution_combine_mode = mode
        world = ke.sim.KangSimWorld(physics_config=config, add_ground=True, state_device="cpu")
        try:
            body = world.physics.create_dynamic_sphere(0.1, (0, 0, 1))
            maximum = 0.0
            for _ in range(120):
                world.physics.step()
                maximum = max(maximum, float(body.get_root_linear_velocity()[2]))
            return maximum
        finally:
            world.release()

    average, multiply = bounce(Mode.AVERAGE), bounce(Mode.MULTIPLY)
    assert average > 1.0
    assert 0.4 < multiply / average < 0.6
