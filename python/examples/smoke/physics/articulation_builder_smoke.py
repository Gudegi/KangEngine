"""Procedural DOF layout, local pivots, independent instances and D6 lifetime."""

import numpy as np
import kangengine as ke


def main():
    p = ke.physics
    Builder = p.ArticulationBuilder
    builder = Builder()
    builder.add_link("root", position=(1.2, 0., 3.), inertia_diagonal=(1., 1., 1.))
    for name, parent, joint in (
        ("hinge", "root", Builder.RevoluteJoint(joint_offset=(0., 0., -.5))),
        ("slider", "hinge", Builder.PrismaticJoint(axis=(1., 0., 0.))),
        ("ball", "slider", Builder.SphericalJoint()),
        ("fixed", "ball", Builder.FixedJoint()),
    ):
        builder.add_link(name, parent=parent, position=(0., 0., 1.), joint=joint,
                         inertia_diagonal=(1., 1., 1.), shapes=[Builder.SphereShape(.1)])
    for name, parent in (("root", None), ("bad", "missing"), ("second_root", None)):
        try:
            builder.add_link(name, parent=parent, inertia_diagonal=(1., 1., 1.))
            raise AssertionError("invalid tree accepted")
        except ValueError:
            pass
    template = builder.build_template()
    assert template.num_links() == 5 and template.num_dofs() == 5
    world = p.PhysicsWorld()
    first = p.Articulation.build_from_template(world, template)
    second = p.Articulation.build_from_template(world, template)
    assert first.num_dofs() == 5 and second.num_dofs() == 5
    first.set_dof_state([.2, .1, .1, 0., 0.], [0.] * 5)
    np.testing.assert_allclose(second.get_dof_positions(), 0.)
    frames = first.get_local_debug_frames()
    np.testing.assert_allclose(frames[1]["joint"][:3], [0., 0., -.5])
    first.release()
    assert second.link("root").valid
    second.release()

    a = world.create_dynamic_box([.1]*3, [0., 0., 3.], density=1000.)
    b = world.create_dynamic_box([.1]*3, [.6, 0., 3.], density=1000.)
    config = p.D6JointConfig()
    config.enabled = True
    config.locked_axes = [True]*6
    config.frame0 = [0., 0., 3., 0., 0., 0., 1.]
    pin = world.create_d6_joint(None, a, config=config)
    config.frame0 = [.6, 0., 0., 0., 0., 0., 1.]
    pair = world.create_d6_joint(a, b, config=config)
    robot = p.Articulation.build_from_template(world, template, config=p.ArticulationConfig.free_base())
    mixed = world.create_d6_joint(b, robot.link("root"), config=config)
    for _ in range(120):
        world.step()
    np.testing.assert_allclose(a.get_root_position(), [0., 0., 3.], atol=.01)
    np.testing.assert_allclose(b.get_root_position(), [.6, 0., 3.], atol=.01)
    for endpoints in ((None, None), (a, a)):
        try:
            world.create_d6_joint(*endpoints)
            raise AssertionError("invalid endpoints accepted")
        except ValueError:
            pass
    old = pin.frame0
    try:
        pin.set_frames([0.]*7, config.frame1)
        raise AssertionError("invalid quaternion accepted")
    except ValueError:
        pass
    np.testing.assert_allclose(pin.frame0, old)
    robot.release()
    assert not mixed.valid and pin.valid and pair.valid
    a.release()
    assert not pin.valid and not pair.valid
    for joint in (pin, pair, mixed):
        try:
            joint.set_enabled(True)
            raise AssertionError("invalid joint accepted update")
        except RuntimeError:
            pass
        joint.release()
        joint.release()
    b.release()
    print("BUILDER/D6 LIFETIME PASS")


if __name__ == "__main__":
    main()
