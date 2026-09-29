import numpy as np
import pytest
import kangengine as ke


def make_model(offset, joint_type=ke.physics.ArticulationBuilder.RevoluteJoint):
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("parent", inertia_diagonal=(1., 1., 1.))
    options = {"joint_offset": offset}
    if joint_type is not ke.physics.ArticulationBuilder.SphericalJoint:
        options["axis"] = (0., 1., 0.)
    builder.add_link("child", parent="parent", position=(0., 0., 1.),
                     joint=joint_type(**options), inertia_diagonal=(1., 1., 1.))
    return builder.build()


@pytest.mark.parametrize("joint_type", [ke.physics.ArticulationBuilder.RevoluteJoint,
                                       ke.physics.ArticulationBuilder.SphericalJoint, ke.physics.ArticulationBuilder.PrismaticJoint])
def test_offset_mapper_rejects_unrepresentable_model(joint_type):
    data = make_model((0., 0., -1.), joint_type)
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data)
    offset = layout.blocks[0].joint_offset
    np.testing.assert_allclose([offset.x, offset.y, offset.z], [0., 0., -1.])
    # Guard at construction covers both state and motion conversion directions.
    with pytest.raises(ValueError, match="joint_offset.*child.*SkeletonState/SkeletonMotion"):
        ke.animation.ArticulationMotionMapper(layout)


def test_zero_offset_conversion_still_round_trips():
    layout = ke.animation.ArticulationCoordinateLayout.from_data(make_model((0., 0., 0.)))
    mapper = ke.animation.ArticulationMotionMapper(layout)
    q = np.array([np.pi / 2], dtype=np.float32)
    state = mapper.to_skeleton_state(q)
    position = state.compute_global_positions()[1]
    np.testing.assert_allclose([position.x, position.y, position.z], [0., 0., 1.])
    np.testing.assert_allclose(mapper.to_articulation_coordinates(state).q, q, atol=1e-6)


def test_offset_affects_physics_and_motion_identity(tmp_path):
    layouts = [ke.animation.ArticulationCoordinateLayout.from_data(make_model(offset))
               for offset in [(0., 0., 0.), (0., 0., -1.), (0., 0., -.5)]]
    assert len({layout.model_signature for layout in layouts}) == 3
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layouts[1], q=np.array([[0.], [np.pi/2]], dtype=np.float32),
        qd=np.zeros((2, 1), dtype=np.float32), fps=30., motion_name="offset")
    from kangengine.animation.articulation_io import (
        save_articulation_motion_npz, load_articulation_motion_npz)
    path = save_articulation_motion_npz(motion, tmp_path / "offset.npz")
    restored = load_articulation_motion_npz(path, layouts[1])
    np.testing.assert_array_equal(restored.q, motion.q)
    with pytest.warns(RuntimeWarning, match="signature differs"):
        load_articulation_motion_npz(path, layouts[0])

    world = ke.physics.PhysicsWorld()
    robot = ke.physics.Articulation.build(world, make_model((0., 0., -1.)))
    try:
        robot.set_dof_state([np.pi/2], [0.])
        positions = np.asarray(robot.get_link_positions()).reshape(-1, 3)
        np.testing.assert_allclose(positions[1], [1., 0., 0.], atol=1e-5)
    finally:
        robot.release()
