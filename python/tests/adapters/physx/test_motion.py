"""PhysX articulation motion buffer conversion tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.adapters.physx import PhysXMotionAdapter, PhysXMotionBuffers


class _LogicalArticulation:
    def __init__(self, names):
        self._names = list(names)

    def get_dof_names(self):
        return self._names


def test_physx_adapter_round_trips_kw5_logical_dofs_and_root():
    path = Path(__file__).parents[4] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)
    adapter = PhysXMotionAdapter(layout)
    adapter.validate_articulation(_LogicalArticulation(adapter.joint_names))

    rng = np.random.default_rng(11)
    # Keep sequential hinge coordinates away from equivalent Euler branches.
    q = (0.3 * rng.normal(size=(5, layout.nq))).astype(np.float32)
    qd = rng.normal(size=(5, layout.nv)).astype(np.float32)
    root = layout.blocks[0]
    quaternion = rng.normal(size=(5, 4)).astype(np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    q[:, root.q_offset + 3 : root.q_offset + 7] = quaternion
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=120.0,
        motion_name="PhysX round trip",
    )

    buffers = adapter.pack(motion)
    expected_root_xyzw = quaternion[:, [1, 2, 3, 0]]
    assert np.allclose(
        np.abs(np.sum(buffers.root_rotations_xyzw * expected_root_xyzw, axis=1)),
        1.0,
        atol=1e-6,
    )
    first_body_blocks = [
        block
        for block in layout.blocks
        if block.body_name == "LeftHip" and block.q_size == 1
    ]
    assert len(first_body_blocks) == 3
    # PhysX represents the three authored hinges as one spherical joint, so
    # its coordinates are not copied canonical XYZ hinge angles.
    assert not np.allclose(
        buffers.joint_positions[:, :3],
        np.stack([motion.q[:, block.q_offset] for block in first_body_blocks], axis=1),
    )

    restored = adapter.unpack(
        buffers.joint_positions,
        buffers.joint_velocities,
        fps=motion.fps(),
        root_positions=buffers.root_positions,
        root_rotations_xyzw=buffers.root_rotations_xyzw,
        root_linear_velocities=buffers.root_linear_velocities,
        root_angular_velocities=buffers.root_angular_velocities,
        motion_name=motion.motion_name(),
    )
    root_slice = slice(root.q_offset + 3, root.q_offset + 7)
    non_quaternion = np.ones(layout.nq, dtype=bool)
    non_quaternion[root_slice] = False
    assert np.allclose(
        restored.q[:, non_quaternion], motion.q[:, non_quaternion], atol=2e-6
    )
    assert np.allclose(
        np.abs(np.sum(restored.q[:, root_slice] * motion.q[:, root_slice], axis=1)),
        1.0,
        atol=1e-6,
    )
    # Packing a sampled trajectory derives PhysX velocities from its position
    # samples, matching MotionLib rather than trusting unrelated input qd.
    assert restored.qd.shape == motion.qd.shape
    assert np.isfinite(restored.qd).all()


def test_physx_validation_rejects_a_different_logical_order():
    path = Path(__file__).parents[4] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)
    adapter = PhysXMotionAdapter(layout)
    try:
        adapter.validate_articulation(
            _LogicalArticulation(reversed(adapter.joint_names))
        )
    except ValueError:
        return
    raise AssertionError("different PhysX logical order was accepted")


def test_physx_adapter_reorders_body_grouped_joint_state():
    path = Path(__file__).parents[4] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)
    adapter = PhysXMotionAdapter(layout)

    body_counts = {}
    for block in layout.blocks:
        if block.joint_name in adapter.joint_names:
            body_counts[block.body_name] = body_counts.get(block.body_name, 0) + 1
    body_names = tuple(reversed(body_counts))
    offsets = [0]
    for body_name in body_names:
        offsets.append(offsets[-1] + body_counts[body_name])

    mapping = adapter.make_dof_mapping(body_names, dof_offsets=offsets)
    assert sorted(mapping) == list(range(len(adapter.joint_names)))

    frames = 2
    values = np.arange(frames * len(mapping), dtype=np.float32).reshape(
        frames, len(mapping)
    )
    state = PhysXMotionBuffers(
        joint_positions=values,
        joint_velocities=-values,
        root_positions=None,
        root_rotations_xyzw=None,
        root_linear_velocities=None,
        root_angular_velocities=None,
    )
    assert adapter.reorder_joint_state(state) is state

    reordered = adapter.reorder_joint_state(state, mapping)
    np.testing.assert_array_equal(reordered.joint_positions, values[:, mapping])
    restored = adapter.restore_joint_state_order(reordered, mapping)
    np.testing.assert_array_equal(restored.joint_positions, state.joint_positions)
    np.testing.assert_array_equal(restored.joint_velocities, state.joint_velocities)
