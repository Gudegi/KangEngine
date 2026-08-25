"""Quaternion MotionLibrary tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import kangengine as ke


def _layout():
    path = Path(__file__).parents[2] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    return ke.animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)


def _motion(layout, *, name: str, fps: float = 2.0):
    q = np.zeros((2, layout.nq), dtype=np.float32)
    qd = np.zeros((2, layout.nv), dtype=np.float32)
    root = layout.blocks[0]
    q[:, root.q_offset + 3] = 1.0
    q[1, root.q_offset : root.q_offset + 3] = [2.0, 0.0, 0.0]
    q[1, root.q_offset + 3 : root.q_offset + 7] = [0.0, 0.0, 0.0, 1.0]
    hinge = next(
        block
        for block in layout.blocks
        if block.type == ke.animation.ArticulationCoordinateType.REVOLUTE
    )
    q[1, hinge.q_offset] = 0.5
    return ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=fps,
        motion_name=name,
    )


def test_motion_library_normalizes_articulation_motion_and_samples_frames():
    layout = _layout()
    motion = _motion(layout, name="clip")
    library = ke.animation.MotionLibrary([motion], keep_source_motions=True)
    skeleton_motion = library.motion(0)

    assert library.motion_names == ("clip",)
    assert library.motion_id("clip") == 0
    assert torch.equal(library.frame_counts, torch.tensor([2]))
    sample = library.sample_frames([0, 0], [0, -1])
    assert sample.local_rotations_wxyz.shape == (
        2,
        skeleton_motion.num_joints(),
        4,
    )
    assert torch.allclose(
        sample.root_positions,
        torch.from_numpy(skeleton_motion.root_translations()),
    )
    assert torch.equal(sample.frame_indices, torch.tensor([0, 1]))


def test_motion_library_slerps_quaternions_and_root_position():
    layout = _layout()
    library = ke.animation.MotionLibrary([_motion(layout, name="clip")])
    sample = library.sample(0, 0.25, loop=False)

    assert torch.allclose(sample.root_positions, torch.tensor([1.0, 0.0, 0.0]))
    expected = np.sqrt(0.5)
    assert torch.allclose(
        sample.root_rotations_wxyz,
        torch.tensor([expected, 0.0, 0.0, expected], dtype=torch.float32),
        atol=1.0e-6,
    )
    assert int(sample.frame_indices) == 0
    assert int(sample.next_frame_indices) == 1
    assert torch.isclose(sample.blend, torch.tensor(0.5))


def test_motion_library_does_not_retain_sources_by_default():
    layout = _layout()
    library = ke.animation.MotionLibrary([_motion(layout, name="clip")])

    assert not library.keep_source_motions
    with np.testing.assert_raises_regex(RuntimeError, "keep_source_motions=True"):
        library.motion(0)


def test_motion_library_runs_fk_after_quaternion_sampling():
    layout = _layout()
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")], keep_source_motions=True
    )
    motion = library.motion(0)

    exact_sample = library.sample_frames([0, 0], [0, 1])
    exact = library.forward_kinematics(exact_sample)
    assert torch.allclose(
        exact.body_positions,
        torch.from_numpy(motion.global_positions()),
        atol=1.0e-6,
    )
    expected_rotations = torch.from_numpy(motion.global_rotations_wxyz())
    assert torch.allclose(
        torch.abs(torch.sum(exact.body_rotations_wxyz * expected_rotations, dim=-1)),
        torch.ones_like(expected_rotations[..., 0]),
        atol=1.0e-6,
    )

    midpoint_sample = library.sample(0, 0.25, loop=False)
    midpoint = library.forward_kinematics(midpoint_sample)
    expected_state = motion.sample(0.25, loop=False)
    expected_positions = torch.tensor(
        [
            [value.x, value.y, value.z]
            for value in expected_state.compute_global_positions()
        ]
    )
    assert torch.allclose(midpoint.body_positions, expected_positions, atol=1.0e-6)


def test_motion_library_physx_adapter_returns_backend_torch_state():
    layout = _layout()
    motion = _motion(layout, name="clip")
    adapter = ke.adapters.physx.PhysXMotionAdapter(layout)
    library = ke.animation.MotionLibrary(
        [motion], adapter=adapter, keep_source_motions=True
    )
    sample = library.sample_frames(0, 1)
    expected = adapter.pack_skeleton_motion(library.motion(0))

    assert sample.backend_state is not None
    assert torch.allclose(
        sample.backend_state.joint_positions,
        torch.from_numpy(expected.joint_positions[1]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        sample.backend_state.joint_velocities,
        torch.from_numpy(expected.joint_velocities[1]),
        atol=1.0e-6,
    )
    assert sample.backend_state.root_rotations_xyzw is not None
    assert sample.backend_state.root_rotations_xyzw.device == library.device


def test_motion_library_weighted_sampling_and_cuda_residency():
    layout = _layout()
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="first"), _motion(layout, name="second")],
        weights=[0.0, 1.0],
    )
    assert torch.equal(library.sample_motion_ids(16), torch.ones(16, dtype=torch.long))

    if torch.cuda.is_available():
        library.to("cuda:0")
        ids = library.sample_motion_ids(16)
        sample = library.sample(ids, library.sample_times(ids), loop=False)
        assert sample.local_rotations_wxyz.device.type == "cuda"
        assert sample.root_positions.device.type == "cuda"
