"""Verify animation -> CPU PhysX -> animation without loading a viewer."""

import gc
import sys

import numpy as np
import kangengine as ke
from kangengine.adapters.physx import PhysXMotionAdapter
from kangengine.utils.math import quat_wxyz_to_matrix_numpy


def main():
    # A branched hierarchy with nonalphabetical names exposes order mistakes.
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", inertia_diagonal=(0.1, 0.1, 0.1))
    for name, parent, offset, axis in (
        ("z_arm", "root", (0.4, 0, 0), (0, 0, 1)),
        ("z_tip", "z_arm", (0.4, 0, 0), (1, 0, 0)),
        ("a_arm", "root", (0, 0.4, 0), (0, 1, 0)),
    ):
        builder.add_link(
            name, parent=parent, position=offset,
            joint=builder.RevoluteJoint(axis=axis),
            inertia_diagonal=(0.1, 0.1, 0.1),
        )
    data = builder.build()
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)
    mapper = ke.animation.ArticulationMotionMapper(layout)
    adapter = PhysXMotionAdapter(layout)

    # Animation uses WXYZ; the physics adapter must preserve the world root
    # translation and convert its nonidentity rotation to XYZW.
    q = np.zeros(layout.nq, dtype=np.float32)
    q[:3] = (0.3, -0.2, 2.0)
    q[3:7] = (np.cos(0.3), 0, 0, np.sin(0.3))
    q[7:] = (0.2, -0.1, 0.3)
    state = mapper.to_skeleton_state(q=q)
    original_matrices = state.compute_global_matrices().copy()
    motion = ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=state.skeleton_tree,
        root_translations=q[None, :3],
        local_rotations_wxyz=np.array([
            [state.get_joint_quat(i).to_wxyz() for i in range(state.num_joints())]
        ], dtype=np.float32),
        fps=120.0,
    )
    buffers = adapter.pack_skeleton_motion(motion)
    np.testing.assert_allclose(buffers.joint_pos[0], q[7:], atol=1e-6)
    np.testing.assert_allclose(buffers.root_pos[0], q[:3], atol=1e-6)
    np.testing.assert_allclose(buffers.root_quat_xyzw[0], q[[4, 5, 6, 3]], atol=1e-6)

    config = ke.physics.PhysicsConfig.z_up()
    config.enable_gpu = False
    config.dt = 1.0 / 120.0
    world = ke.physics.PhysicsWorld(config=config)
    articulation = ke.physics.Articulation.build(
        physics=world, data=data, config=ke.physics.ArticulationConfig.free_base(),
    )
    try:
        adapter.validate_articulation(articulation)
        names = [data.skeleton_tree.node_name(i) for i in range(state.num_joints())]
        assert list(articulation.body_names) == names
        articulation.set_root_state(
            pos=buffers.root_pos[0], quat_xyzw=buffers.root_quat_xyzw[0],
            linear_velocity=buffers.root_vel[0], angular_velocity=buffers.root_ang_vel[0],
        )
        articulation.set_dof_state(positions=buffers.joint_pos[0], velocities=buffers.joint_vel[0])
        initial_joints = articulation.get_dof_positions()
        initial_root = articulation.get_root_position()
        expected_joints = initial_joints.copy()
        expected_root = initial_root.copy()

        # Setters copy input; native snapshot getters and animation state do not
        # alias the changing physics state or the caller's command buffers.
        buffers.joint_pos[:] = 0
        np.testing.assert_allclose(articulation.get_dof_positions(), expected_joints)
        for _ in range(8):
            world.step()
        root = articulation.get_root_position()
        assert root[2] < expected_root[2] - 0.001
        returned = adapter.unpack(
            joint_pos=articulation.get_dof_positions()[None, :],
            joint_vel=articulation.get_dof_velocities()[None, :],
            root_pos=root[None, :],
            root_quat_xyzw=articulation.get_root_quat_xyzw()[None, :],
            root_vel=articulation.get_root_linear_velocity()[None, :],
            root_ang_vel=articulation.get_root_angular_velocity()[None, :],
            fps=120.0,
        )
        reconstructed = mapper.to_skeleton_motion(returned).frame(0)
        matrices = reconstructed.compute_global_matrices()
        np.testing.assert_allclose(
            matrices[:, :3, 3], articulation.get_link_positions().reshape(-1, 3), atol=2e-4,
        )
        rotations = articulation.get_link_quats_xyzw().reshape(-1, 4)
        expected_rotations = np.array([quat_wxyz_to_matrix_numpy(r[[3, 0, 1, 2]]) for r in rotations])
        np.testing.assert_allclose(matrices[:, :3, :3], expected_rotations, atol=2e-4)
        np.testing.assert_allclose(initial_joints, expected_joints)
        np.testing.assert_allclose(initial_root, expected_root)
        np.testing.assert_allclose(state.compute_global_matrices(), original_matrices)
        roundtrip_q = np.asarray(mapper.to_articulation_coordinates(reconstructed).q)
        np.testing.assert_allclose(roundtrip_q, returned.q[0], atol=2e-4)
    finally:
        articulation.release()

    del articulation, world, data, builder
    gc.collect()
    np.testing.assert_allclose(reconstructed.compute_global_matrices(), matrices)
    np.testing.assert_allclose(initial_root, expected_root)
    assert "kangengine.app.application" not in sys.modules
    assert "kangengine.visual.sim" not in sys.modules
    print("PASS: viewer-free animation/PhysX round trip, frames, ordering and ownership")


if __name__ == "__main__":
    main()
