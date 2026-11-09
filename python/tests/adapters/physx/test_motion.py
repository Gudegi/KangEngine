"""PhysX articulation motion buffer conversion tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.adapters.physx import PhysXMotionAdapter, PhysXMotionBuffers


class _LogicalArticulation:
    def __init__(self, names):
        self._names = list(names)

    def get_dof_names(self):
        return self._names


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_joint_mapping_reuses_device_indices_and_preserves_gradients(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path = Path(__file__).parents[4] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=True
    )
    adapter = PhysXMotionAdapter(layout)
    count = len(adapter.joint_names)
    mapping = tuple(reversed(range(count)))
    values = torch.arange(2 * count, device=device, dtype=torch.float32).reshape(
        2, count
    )
    values.requires_grad_()

    def reorder_and_restore():
        reordered = adapter.reorder_joint_values(values, mapping)
        return reordered, adapter.restore_joint_values_order(reordered, mapping)

    reorder_and_restore()
    if device.startswith("cuda"):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            reordered, restored = reorder_and_restore()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    else:
        reordered, restored = reorder_and_restore()
    torch.testing.assert_close(reordered, values.flip(-1))
    torch.testing.assert_close(restored, values)
    restored.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))

    subset = adapter.reorder_joint_values(values, mapping[:2])
    torch.testing.assert_close(subset, values.flip(-1)[..., :2])
    with pytest.raises(ValueError, match="complete DOF mapping"):
        adapter.restore_joint_values_order(subset, mapping[:2])


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
        np.abs(np.sum(buffers.root_quat_xyzw * expected_root_xyzw, axis=1)),
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
        buffers.joint_pos[:, :3],
        np.stack([motion.q[:, block.q_offset] for block in first_body_blocks], axis=1),
    )

    restored = adapter.unpack(
        buffers.joint_pos,
        buffers.joint_vel,
        fps=motion.fps(),
        root_pos=buffers.root_pos,
        root_quat_xyzw=buffers.root_quat_xyzw,
        root_vel=buffers.root_vel,
        root_ang_vel=buffers.root_ang_vel,
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
        joint_pos=values,
        joint_vel=-values,
        root_pos=None,
        root_quat_xyzw=None,
        root_vel=None,
        root_ang_vel=None,
    )
    assert adapter.reorder_joint_state(state) is state

    reordered = adapter.reorder_joint_state(state, mapping)
    np.testing.assert_array_equal(reordered.joint_pos, values[:, mapping])
    restored = adapter.restore_joint_state_order(reordered, mapping)
    np.testing.assert_array_equal(restored.joint_pos, state.joint_pos)
    np.testing.assert_array_equal(restored.joint_vel, state.joint_vel)
