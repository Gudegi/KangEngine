"""Validate CPU rigid collision and articulation stepping without a viewer."""

from pathlib import Path

import numpy as np

import kangengine as ke


def main():
    config = ke.physics.PhysicsConfig.z_up()
    config.enable_gpu = False
    config.dt = 1.0 / 120.0
    config.restitution = 0.0
    world = ke.sim.KangSimWorld(
        physics_config=config, sim_device="cpu", state_device="cpu", add_ground=True,
    )
    try:
        # Exercise collision response and the packaged MJCF articulation loader.
        sphere = world.physics.create_dynamic_sphere(0.2, (3.0, 0.0, 2.0))
        mjcf_path = Path(ke.__file__).resolve().parent / "assets/characters/kw/kw5.xml"
        desc = ke.asset.MJCFLoader.load(str(mjcf_path))
        robot = world.add_articulation(
            desc, config=ke.physics.ArticulationConfig.fixed_base(),
        )
        for _ in range(12):
            world.physics.step()
        assert sphere.get_root_position()[2] < 1.99, "CPU gravity did not advance the body"
        assert sphere.get_root_linear_velocity()[2] < 0.0

        # A resting sphere must remain on the plane rather than fall through it.
        for _ in range(348):
            world.physics.step()
        np.testing.assert_allclose(sphere.get_root_position()[2], 0.2, atol=0.03)
        np.testing.assert_allclose(sphere.get_root_linear_velocity(), 0.0, atol=0.05)
        world.refresh()
        assert robot.articulation.num_dofs() > 0
        assert np.isfinite(robot.articulation.get_dof_positions()).all()
    finally:
        world.release()

    print("PASS: PhysX CPU gravity, rigid collision, and articulation stepping")


if __name__ == "__main__":
    main()
