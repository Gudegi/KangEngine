"""Measured joint wrenches/effort: gravity, authored frames, CPU/CUDA mapping."""

import os

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.state import KangWorldState

p = ke.physics


def load_model(scale=1.0):
    builder = p.ArticulationBuilder()
    builder.add_link("root", inertia_diagonal=(1., 1., 1.))
    # Rotate the child frame and offset the anchor. The COM-to-anchor lever
    # stays one metre, and authored +Y becomes world -X.
    builder.add_link(
        "hinge", parent="root", mass=2*scale, com=(.75, 0., 0.),
        rotation_xyzw=(0., 0., 2**-.5, 2**-.5),
        inertia_diagonal=(1., 1., 1.),
        joint=p.ArticulationBuilder.RevoluteJoint(axis=(0., 1., 0.), joint_offset=(-.25, 0., 0.),
                             kp=10000., kd=200.),
    )
    builder.add_link(
        "slide", parent="root", mass=3*scale, inertia_diagonal=(1., 1., 1.),
        joint=p.ArticulationBuilder.PrismaticJoint(axis=(0., 0., -1.), kp=10000., kd=200.),
    )
    builder.add_link(
        "ball", parent="root", mass=4*scale, com=(.3, .5, .2),
        rotation_xyzw=(0., 0., 2**-.5, 2**-.5),
        inertia_diagonal=(1., 1., 1.), joint=p.ArticulationBuilder.SphericalJoint(kp=10000., kd=200.),
    )
    # A fixed joint has no DOF effort but still transmits a complete wrench.
    builder.add_link(
        "fixed", parent="root", mass=2*scale, com=(.4, .3, 0.),
        rotation_xyzw=(0., 0., 2**-.5, 2**-.5),
        inertia_diagonal=(1., 1., 1.), joint=p.ArticulationBuilder.FixedJoint(),
    )
    return builder.build()


@pytest.mark.parametrize("backend", ["pgs", "tgs", "gpu"])
def test_projected_gravity_loads(backend, monkeypatch):
    if backend == "gpu" and os.environ.get("KANGENGINE_TEST_PHYSX_GPU") != "1":
        pytest.skip("set KANGENGINE_TEST_PHYSX_GPU=1 for Direct GPU checks")
    config = p.PhysicsConfig.z_up()
    config.solver_type = 0 if backend == "pgs" else 1
    config.enable_gpu = backend == "gpu"
    world = p.PhysicsWorld(config)
    # An unrelated larger articulation introduces raw buffer padding/extra rows.
    padding = p.ArticulationBuilder()
    padding.add_link("root", inertia_diagonal=(1., 1., 1.))
    for i in range(6):
        padding.add_link(str(i), parent="root", inertia_diagonal=(1., 1., 1.))
    dummy = p.Articulation.build(world, padding.build())
    robots = [p.Articulation.build(world, load_model(scale)) for scale in (1., 2.)]
    gpu = None
    state = KangWorldState(
        num_envs=2, device="cuda" if backend == "gpu" else "cpu",
        canonical_source="gpu" if backend == "gpu" else "cpu",
        gpu_system_provider=lambda: gpu,
    )
    # Reverse environment order relative to native construction order.
    for env_id, robot in enumerate(reversed(robots)):
        state.add_articulation(robot, env_id=env_id, obj_id=7, physics=world)
        robot.set_drive_targets([0.] * robot.num_dofs())
    try:
        if backend == "gpu":
            gpu = p.PhysicsGpuSystem(world, p.GpuPhysicsConfig())
            gpu.init()
        for _ in range(300):
            world.step()
        value = state.get_dof_projected_joint_forces(7)
        assert value.shape == (2, 5)
        assert value.is_cuda == (backend == "gpu")
        # Static balance: -gravity along prismatic axes, -(r x gravity)
        # along angular axes. Small deflections from finite drive stiffness.
        expected = torch.tensor([-19.62, -29.43, 19.62, -11.772, 0.])
        expected = torch.stack((2*expected, expected)).to(value.device)
        torch.testing.assert_close(value, expected, atol=.25, rtol=.015)
        wrenches = state.get_link_incoming_joint_forces(obj_id=7)
        assert wrenches.shape == (2, 5, 6)
        assert wrenches.is_cuda == (backend == "gpu")
        torch.testing.assert_close(wrenches[:, 0], torch.zeros_like(wrenches[:, 0]))
        # Gravity support at the fixed child joint origin, in its rotated axes.
        fixed = torch.tensor([0., 0., 19.62, 5.886, -7.848, 0.], device=value.device)
        torch.testing.assert_close(wrenches[:, 4], torch.stack((2*fixed, fixed)),
                                   atol=.02, rtol=.001)
        for env_id, robot in enumerate(reversed(robots)):
            # Project the complete logical-link wrench onto each motion axis.
            native_links = list(robot.get_link_indices())
            components = [native_links.index(i // 6)*6 + i % 6
                          for i in robot.get_dof_joint_force_indices()]
            torch.testing.assert_close(wrenches[env_id].flatten()[components], value[env_id])
            if backend != "gpu":
                native = robot.get_link_incoming_joint_forces()
                assert native.shape == (5, 6)
                np.testing.assert_allclose(native, wrenches[env_id].numpy())
        for robot in robots:
            np.testing.assert_array_equal(robot.get_dof_forces(), np.zeros(5))
        if backend == "gpu":
            with pytest.raises(RuntimeError, match="Direct GPU"):
                robots[0].get_dof_projected_joint_forces()
            with pytest.raises(RuntimeError, match="Direct GPU"):
                robots[0].get_link_incoming_joint_forces()
            pointer = value.data_ptr()
            wrench_pointer = wrenches.data_ptr()
            retained_wrenches = wrenches.clone()
            def forbidden(*args, **kwargs):
                raise AssertionError("host readback/metadata queried during warm CUDA read")
            for robot in robots:
                monkeypatch.setattr(robot, "get_dof_projected_joint_forces", forbidden)
                monkeypatch.setattr(robot, "get_dof_joint_force_indices", forbidden)
                monkeypatch.setattr(robot, "get_link_incoming_joint_forces", forbidden)
                monkeypatch.setattr(robot, "get_link_indices", forbidden)
            monkeypatch.setattr(gpu, "fetch_articulation_link_incoming_joint_force", forbidden)
            again_wrenches = state.gpu.get_link_incoming_joint_forces(7, fetch=False)
            assert again_wrenches.data_ptr() == wrench_pointer
            torch.testing.assert_close(again_wrenches, retained_wrenches)
            again = state.gpu.get_dof_projected_joint_forces(7, fetch=False)
            assert again.data_ptr() == pointer
            torch.testing.assert_close(again, expected, atol=.25, rtol=.015)
    finally:
        state.release()
        if gpu is not None:
            gpu.invalidate()
        for robot in robots:
            robot.release()
        dummy.release()


def test_projected_read_after_release():
    world = p.PhysicsWorld()
    robot = p.Articulation.build(world, load_model())
    world.step()
    robot.get_dof_projected_joint_forces()  # allocate lazy cache before release
    robot.release()
    with pytest.raises(RuntimeError, match="in a scene"):
        robot.get_dof_projected_joint_forces()
    with pytest.raises(RuntimeError, match="in a scene"):
        robot.get_link_incoming_joint_forces()
