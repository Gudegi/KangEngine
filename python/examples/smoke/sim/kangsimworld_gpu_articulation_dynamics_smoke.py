"""Validate lazy PhysX GPU articulation dynamics tensors."""

from pathlib import Path

import torch

import kangengine as ke


def main():
    world = ke.sim.KangSimWorld(num_envs=2, sim_device="cuda", add_ground=False)
    try:
        asset = (
            Path(ke.__file__).resolve().parent
            / "assets"
            / "characters"
            / "kw"
            / "kw5.xml"
        )
        data = world.load_mjcf(str(asset))
        config = ke.physics.ArticulationConfig(fix_base=False)
        for env_id in range(world.num_envs):
            world.add_articulation(data, env_id=env_id, obj_id=0, config=config)

        world.init_gpu_system(cuda_device_id=0)
        world.step(substeps=1)

        # Optional dynamics storage must remain unallocated until first use.
        if world.gpu_system.articulation_mass_matrices().ptr != 0:
            raise AssertionError("mass-matrix buffer was allocated eagerly")

        jacobian = world.get_gpu_articulation_dense_jacobians()
        mass = world.get_gpu_articulation_mass_matrices()
        gravity = world.get_gpu_articulation_gravity_forces()
        coriolis = world.get_gpu_articulation_coriolis_forces()
        link_acceleration = world.get_gpu_articulation_link_accelerations()
        com_world = world.get_gpu_articulation_com_world()
        com_root = world.get_gpu_articulation_com_root()
        centroidal, centroidal_bias = (
            world.get_gpu_articulation_centroidal_dynamics()
        )
        torch.cuda.synchronize(0)

        # Repeated reads in one simulation generation must reuse computed
        # dynamics buffers instead of dispatching PhysX again.
        dynamics_versions = {
            "jacobian": world.gpu_system.articulation_dense_jacobians().version,
            "mass": world.gpu_system.articulation_mass_matrices().version,
            "gravity": world.gpu_system.articulation_gravity_forces().version,
            "coriolis": world.gpu_system.articulation_coriolis_forces().version,
            "com_world": world.gpu_system.articulation_com_world().version,
            "com_root": world.gpu_system.articulation_com_root().version,
            "centroidal": (
                world.gpu_system.articulation_centroidal_momentum_matrices().version
            ),
        }
        world.get_gpu_articulation_dense_jacobians()
        world.get_gpu_articulation_mass_matrices()
        world.get_gpu_articulation_gravity_forces()
        world.get_gpu_articulation_coriolis_forces()
        world.get_gpu_articulation_com_world()
        world.get_gpu_articulation_com_root()
        world.get_gpu_articulation_centroidal_dynamics()
        repeated_versions = {
            "jacobian": world.gpu_system.articulation_dense_jacobians().version,
            "mass": world.gpu_system.articulation_mass_matrices().version,
            "gravity": world.gpu_system.articulation_gravity_forces().version,
            "coriolis": world.gpu_system.articulation_coriolis_forces().version,
            "com_world": world.gpu_system.articulation_com_world().version,
            "com_root": world.gpu_system.articulation_com_root().version,
            "centroidal": (
                world.gpu_system.articulation_centroidal_momentum_matrices().version
            ),
        }
        if repeated_versions != dynamics_versions:
            raise AssertionError("articulation dynamics cache recomputed in one step")

        rows, dofs = world.get_gpu_articulation_dynamics_shape(0)
        max_rows = world.gpu_system.articulation_max_jacobian_rows()
        max_dofs = world.gpu_system.articulation_max_generalized_dofs()
        expected = {
            "jacobian": (world.num_envs, max_rows * max_dofs),
            "mass": (world.num_envs, max_dofs * max_dofs),
            "gravity": (world.num_envs, max_dofs),
            "coriolis": (world.num_envs, max_dofs),
            "link_acceleration": (
                world.num_envs,
                world.gpu_system.articulation_max_links(),
                6,
            ),
            "com_world": (world.num_envs, 3),
            "com_root": (world.num_envs, 3),
            "centroidal": (world.num_envs, 6 * max_dofs),
            "centroidal_bias": (world.num_envs, 6),
        }
        for name, tensor in (
            ("jacobian", jacobian),
            ("mass", mass),
            ("gravity", gravity),
            ("coriolis", coriolis),
            ("link_acceleration", link_acceleration),
            ("com_world", com_world),
            ("com_root", com_root),
            ("centroidal", centroidal),
            ("centroidal_bias", centroidal_bias),
        ):
            if tuple(tensor.shape) != expected[name]:
                raise AssertionError(
                    f"unexpected {name} shape {tuple(tensor.shape)}, expected {expected[name]}"
                )
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"{name} contains non-finite values")

        if rows <= 0 or dofs <= 0:
            raise AssertionError(f"invalid dynamics shape rows={rows}, dofs={dofs}")
        mass0 = mass[0, : dofs * dofs].reshape(dofs, dofs)
        torch.testing.assert_close(mass0, mass0.T, rtol=1e-4, atol=1e-5)
        if torch.linalg.diagonal(mass0).min() <= 0:
            raise AssertionError("mass matrix has a non-positive diagonal")

        if world.gpu_system.articulation_link_forces().ptr != 0:
            raise AssertionError("link command buffers were allocated eagerly")
        before = world.get_gpu_articulation_link_data().clone()
        link_forces = world.get_gpu_articulation_link_forces()
        link_forces.zero_()
        link_forces[:, 0, 0] = 1000.0
        world.apply_gpu_articulation_link_wrenches(forces=True, torques=False)
        world.step(substeps=1)
        world.get_gpu_articulation_dense_jacobians()
        if (
            world.gpu_system.articulation_dense_jacobians().version
            <= dynamics_versions["jacobian"]
        ):
            raise AssertionError("articulation dynamics cache survived a physics step")
        link_forces.zero_()
        world.apply_gpu_articulation_link_wrenches(forces=True, torques=False)
        after = world.get_gpu_articulation_link_data()
        torch.cuda.synchronize(0)
        if not torch.all(after[:, 0, 7] > before[:, 0, 7]):
            raise AssertionError("batched CUDA link force did not increase root x velocity")
        before_torque = after.clone()
        link_torques = world.get_gpu_articulation_link_torques()
        link_torques.zero_()
        link_torques[:, 0, 2] = 1000.0
        world.apply_gpu_articulation_link_wrenches(forces=False, torques=True)
        world.step(substeps=1)
        link_torques.zero_()
        world.apply_gpu_articulation_link_wrenches(forces=False, torques=True)
        after_torque = world.get_gpu_articulation_link_data()
        torch.cuda.synchronize(0)
        if not torch.all(after_torque[:, 0, 12] > before_torque[:, 0, 12]):
            raise AssertionError(
                "batched CUDA link torque did not increase root z angular velocity"
            )

        print(
            "PASS: lazy GPU articulation dynamics",
            f"jacobian=({rows}, {dofs})",
            f"mass=({dofs}, {dofs})",
        )
    finally:
        world.release()


if __name__ == "__main__":
    main()
