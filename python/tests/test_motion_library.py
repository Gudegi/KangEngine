"""Quaternion MotionLibrary tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.utils import batched_rotations


def _layout():
    path = Path(__file__).parents[2] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    return ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=True
    )


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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("free_root", [False, True])
def test_physx_state_fk_matches_native_coordinates(
    tmp_path, monkeypatch, device, free_root
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    # Nonidentity bind rotations, a grouped joint, a hinge, and a fixed child.
    path = tmp_path / "fk.xml"
    path.write_text(
        '<mujoco><worldbody><body name="root" pos="1 2 3" quat="0.9238795 0 0 0.3826834">'
        '<geom type="sphere" size="0.1"/>'
        '<body name="hip" pos="0 0 1" quat="0.9659258 0.258819 0 0">'
        '<joint name="rx" axis="1 0 0"/><joint name="ry" axis="0 1 0"/>'
        '<joint name="rz" axis="0 0 1"/><geom type="sphere" size="0.1"/>'
        '<body name="knee" pos="0.3 0 -1" quat="0.9238795 0 0.3826834 0">'
        '<joint name="hinge" axis="0 1 0"/><geom type="sphere" size="0.1"/>'
        '<body name="tip" pos="0.2 0 -1" quat="0.9659258 0 0 0.258819">'
        '<geom type="sphere" size="0.1"/></body></body></body>'
        '<body name="branch" pos="0 0.5 0"><joint name="branch" axis="1 1 0"/>'
        '<geom type="sphere" size="0.1"/></body></body>'
        "</worldbody></mujoco>"
    )
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=free_root
    )
    q = np.zeros((2, layout.nq), dtype=np.float32)
    if free_root:
        q[:, :7] = [2, 3, 4, 0.9238795, 0, 0, 0.3826834]
    hinges = [block for block in layout.blocks if block.q_size == 1]
    for idx, block in enumerate(hinges):
        q[:, block.q_offset] = [0.1 * (idx + 1), -0.15 * (idx + 1)]
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout, q=q, qd=np.zeros((2, layout.nv), dtype=np.float32), fps=2.0
    )
    adapter = ke.adapters.physx.PhysXMotionAdapter(layout)
    library = ke.animation.MotionLibrary([motion], adapter=adapter).to(device)
    sample = library.sample_arti_state(
        0, times=torch.tensor([[0, 0.2], [0.3, 0.5]], device=device)
    )
    # FK must use the supplied state, including edits after sampling.
    sample = replace(sample, joint_pos=sample.joint_pos + 0.07)

    def array(value):
        return (
            None
            if value is None
            else value.detach().cpu().reshape(-1, value.shape[-1]).numpy()
        )

    canonical = adapter.unpack(
        array(sample.joint_pos),
        array(sample.joint_vel),
        fps=2.0,
        root_pos=array(sample.root_pos),
        root_quat_xyzw=array(sample.root_quat_xyzw),
        root_vel=array(sample.root_vel),
        root_ang_vel=array(sample.root_ang_vel),
    )
    native = ke.animation.ArticulationMotionMapper(layout).to_skeleton_motion(canonical)
    expected_pos = torch.tensor(native.global_positions(), device=device).reshape(
        2, 2, data.skeleton_tree.num_joints(), 3
    )
    expected_quat = torch.tensor(native.global_rotations_wxyz(), device=device).reshape(
        2, 2, data.skeleton_tree.num_joints(), 4
    )
    result = library.forward_kinematics(sample)
    torch.testing.assert_close(result.body_pos_w, expected_pos, atol=2e-5, rtol=2e-5)
    dot = (result.body_quat_wxyz_w * expected_quat).sum(dim=-1).abs()
    torch.testing.assert_close(dot, torch.ones_like(dot), atol=2e-5, rtol=2e-5)
    assert result.body_vel_w is None and result.body_ang_vel_w is None
    assert result.body_pos_w.device == sample.joint_pos.device

    # Repeated queries use device metadata and do not call CPU conversion paths.
    def forbidden(*args, **kwargs):
        raise AssertionError("FK must not convert state through CPU")

    monkeypatch.setattr(adapter, "unpack", forbidden)
    monkeypatch.setattr(
        ke.animation.ArticulationMotionMapper, "to_skeleton_state", forbidden
    )
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    if device.startswith("cuda"):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            again = library.forward_kinematics(sample)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    else:
        again = library.forward_kinematics(sample)
    torch.testing.assert_close(again.body_pos_w, result.body_pos_w)
    scalar = replace(
        sample,
        **{
            name: None if value is None else value[0, 0]
            for name, value in vars(sample).items()
        },
    )
    torch.testing.assert_close(
        library.forward_kinematics(scalar).body_pos_w, result.body_pos_w[0, 0]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_level_fk_matches_sequential_values_and_gradients(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layout = _layout()
    library = ke.animation.MotionLibrary([_motion(layout, name="clip")]).to(device)
    sample = library.sample_skel_state(0, torch.full((2, 3), 0.2, device=device))
    sample = replace(
        sample,
        root_pos=sample.root_pos.detach().requires_grad_(),
        local_quat_wxyz=sample.local_quat_wxyz.detach().requires_grad_(),
    )

    positions = [sample.root_pos]
    rotations = [sample.local_quat_wxyz[..., 0, :]]
    for idx, parent in enumerate(library._parent_indices[1:], start=1):
        positions.append(
            positions[parent]
            + batched_rotations.quat_wxyz_rotate(
                rotations[parent], library._body_local_offsets[idx]
            )
        )
        rotations.append(
            batched_rotations.quat_wxyz_multiply(
                rotations[parent], sample.local_quat_wxyz[..., idx, :]
            )
        )
    expected_pos = torch.stack(positions, dim=-2)
    expected_quat = torch.stack(rotations, dim=-2)
    result = library.forward_kinematics(sample)
    torch.testing.assert_close(result.body_pos_w, expected_pos)
    torch.testing.assert_close(result.body_quat_wxyz_w, expected_quat)

    inputs = (sample.root_pos, sample.local_quat_wxyz)
    expected_grad = torch.autograd.grad(
        expected_pos.square().sum() + expected_quat.sum(), inputs
    )
    actual_grad = torch.autograd.grad(
        result.body_pos_w.square().sum() + result.body_quat_wxyz_w.sum(), inputs
    )
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("free_root", [False, True])
def test_physx_fk_without_scalar_joints(tmp_path, device, free_root):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path = tmp_path / "root.xml"
    path.write_text(
        '<mujoco><worldbody><body name="root" pos="1 2 3">'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=free_root
    )
    q = np.zeros((1, layout.nq), dtype=np.float32)
    if free_root:
        q[0] = [4, 5, 6, 1, 0, 0, 0]
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout, q=q, qd=np.zeros((1, layout.nv), dtype=np.float32), fps=30.0
    )
    library = ke.animation.MotionLibrary(
        [motion], adapter=ke.adapters.physx.PhysXMotionAdapter(layout), device=device
    )
    result = library.forward_kinematics(library.sample_arti_state(0, frame_ids=0))
    expected = torch.tensor(
        [[4.0, 5.0, 6.0] if free_root else [1.0, 2.0, 3.0]], device=device
    )
    torch.testing.assert_close(result.body_pos_w, expected)
    torch.testing.assert_close(
        result.body_quat_wxyz_w, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    )


def test_articulation_fk_requires_adapter_support():
    layout = _layout()
    adapter = ke.adapters.mujoco.MuJoCoMotionAdapter(layout)
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")], adapter=adapter
    )
    with pytest.raises(NotImplementedError, match="does not support state FK"):
        library.forward_kinematics(library.sample_arti_state(0, frame_ids=0))


def _write_bvh(path: Path) -> None:
    path.write_text(
        "HIERARCHY\n"
        "ROOT Hips\n"
        "{\n"
        "  OFFSET 0 0 0\n"
        "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        "  End Site\n"
        "  {\n"
        "    OFFSET 0 1 0\n"
        "  }\n"
        "}\n"
        "MOTION\n"
        "Frames: 2\n"
        "Frame Time: 0.0333333333\n"
        "0 0 0 0 0 0\n"
        "1 0 0 0 0 0\n",
        encoding="utf-8",
    )


def test_motion_library_normalizes_articulation_motion_and_samples_frames():
    layout = _layout()
    motion = _motion(layout, name="clip")
    library = ke.animation.MotionLibrary([motion], keep_source_motions=True)
    skeleton_motion = library.source_motion(0)

    assert library.motion_names == ("clip",)
    assert library.motion_id("clip") == 0
    assert torch.equal(library.frame_counts, torch.tensor([2]))
    sample = library.sample_skel_state([0, 0], frame_ids=[0, -1])
    assert sample.local_quat_wxyz.shape == (
        2,
        skeleton_motion.num_joints(),
        4,
    )
    assert torch.allclose(
        sample.root_pos,
        torch.from_numpy(skeleton_motion.root_translations()),
    )
    assert torch.equal(sample.frame_indices, torch.tensor([0, 1]))


def test_motion_library_loads_weighted_yaml_of_relative_bvh_files(tmp_path):
    motion_path = tmp_path / "walk.bvh"
    _write_bvh(motion_path)
    manifest = tmp_path / "motions.yaml"
    manifest.write_text(
        "motions:\n- file: walk.bvh\n  weight: 0.0\n- file: walk.bvh\n  weight: 1.0\n",
        encoding="utf-8",
    )

    library = ke.animation.MotionLibrary.from_file(manifest)

    assert library.num_motions == 2
    assert library.motion_names == ("0:walk", "1:walk")
    assert torch.equal(library.frame_counts, torch.tensor([2, 2]))
    buffers = library.source_buffers()
    assert torch.equal(buffers.frame_offsets, torch.tensor([0, 2]))
    assert buffers.root_pos.shape == (4, 3)
    assert buffers.root_pos.is_contiguous()
    assert buffers.local_quat_wxyz.is_contiguous()
    assert torch.equal(library.sample_motion_ids(16), torch.ones(16, dtype=torch.long))
    sample = library.sample_skel_state([0, 1], frame_ids=[1, 1], loop=False)
    torch.testing.assert_close(sample.root_pos[0], sample.root_pos[1])


def test_motion_library_slerps_quaternions_and_root_position():
    layout = _layout()
    library = ke.animation.MotionLibrary([_motion(layout, name="clip")])
    sample = library.sample_skel_state(0, 0.25, loop=False)

    assert torch.allclose(sample.root_pos, torch.tensor([1.0, 0.0, 0.0]))
    expected = np.sqrt(0.5)
    assert torch.allclose(
        sample.root_quat_wxyz,
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
        library.source_motion(0)


def test_motion_library_runs_fk_after_quaternion_sampling():
    layout = _layout()
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")], keep_source_motions=True
    )
    motion = library.source_motion(0)

    exact_sample = library.sample_skel_state([0, 0], frame_ids=[0, 1])
    exact = library.forward_kinematics(exact_sample)
    assert torch.allclose(
        exact.body_pos_w,
        torch.from_numpy(motion.global_positions()),
        atol=1.0e-6,
    )
    expected_rotations = torch.from_numpy(motion.global_rotations_wxyz())
    assert torch.allclose(
        torch.abs(torch.sum(exact.body_quat_wxyz_w * expected_rotations, dim=-1)),
        torch.ones_like(expected_rotations[..., 0]),
        atol=1.0e-6,
    )

    midpoint_sample = library.sample_skel_state(0, 0.25, loop=False)
    midpoint = library.forward_kinematics(midpoint_sample)
    expected_state = motion.sample(0.25, loop=False)
    expected_positions = torch.tensor(
        [
            [value.x, value.y, value.z]
            for value in expected_state.compute_global_positions()
        ]
    )
    assert torch.allclose(midpoint.body_pos_w, expected_positions, atol=1.0e-6)


def test_motion_library_exposes_precomputed_packed_kinematics():
    layout = _layout()
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")],
        keep_source_motions=True,
        precompute_kinematics=True,
    )
    motion = library.source_motion(0)
    buffers = library.source_buffers()

    assert buffers.body_pos_w is not None
    assert buffers.body_quat_wxyz_w is not None
    assert buffers.body_vel_w is not None
    assert buffers.body_ang_vel_w is not None
    assert torch.allclose(
        buffers.body_pos_w,
        torch.from_numpy(motion.global_positions()),
    )
    assert torch.allclose(
        buffers.body_vel_w,
        torch.from_numpy(motion.global_linear_velocities()),
    )
    expected_rotations = torch.from_numpy(motion.global_rotations_wxyz())
    assert torch.allclose(
        torch.abs(torch.sum(buffers.body_quat_wxyz_w * expected_rotations, dim=-1)),
        torch.ones_like(expected_rotations[..., 0]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        buffers.body_ang_vel_w,
        torch.from_numpy(motion.global_angular_velocities()),
    )
    assert buffers.root_pos.data_ptr() == library.source_buffers().root_pos.data_ptr()
    assert torch.equal(buffers.fps, torch.tensor([motion.fps()]))

    kinematics = library.sample_body_state(0, 0.5 / motion.fps(), loop=False)
    expected_positions = 0.5 * (buffers.body_pos_w[0] + buffers.body_pos_w[1])
    assert torch.allclose(kinematics.body_pos_w, expected_positions)


def test_motion_library_physx_adapter_returns_backend_torch_state():
    layout = _layout()
    motion = _motion(layout, name="clip")
    adapter = ke.adapters.physx.PhysXMotionAdapter(layout)
    library = ke.animation.MotionLibrary(
        [motion], adapter=adapter, keep_source_motions=True
    )
    sample = library.sample_arti_state(0, frame_ids=1)
    expected = adapter.pack_skeleton_motion(library.source_motion(0))
    packed = library.adapted_buffers()

    assert packed is not None
    assert torch.allclose(
        packed.joint_pos,
        torch.from_numpy(expected.joint_pos),
        atol=1.0e-6,
    )
    assert torch.allclose(
        packed.joint_vel,
        torch.from_numpy(expected.joint_vel),
        atol=1.0e-6,
    )
    assert torch.allclose(
        sample.joint_pos,
        torch.from_numpy(expected.joint_pos[1]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        sample.joint_vel,
        torch.from_numpy(expected.joint_vel[1]),
        atol=1.0e-6,
    )
    assert sample.root_quat_xyzw is not None
    assert sample.root_quat_xyzw.device == library.device


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
        sample = library.sample_skel_state(ids, library.sample_times(ids), loop=False)
        assert sample.local_quat_wxyz.device.type == "cuda"
        assert sample.root_pos.device.type == "cuda"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("backend", ["physx", "newton", "mujoco"])
def test_prepared_reference_sampling_does_not_reconvert_sources(
    monkeypatch, device, backend
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    adapter_types = {
        "physx": ke.adapters.physx.PhysXMotionAdapter,
        "newton": ke.adapters.newton.NewtonMotionAdapter,
        "mujoco": ke.adapters.mujoco.MuJoCoMotionAdapter,
    }
    layout = _layout()
    adapter = adapter_types[backend](layout)
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")],
        adapter=adapter,
        device=device,
        precompute_kinematics=True,
    )
    ids = torch.zeros(3, dtype=torch.long, device=device)
    times = torch.tensor([0.0, 0.25, 0.5], device=device)
    buffers = library.adapted_buffers()
    packed = library.source_buffers()

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling must use prepared device buffers")

    monkeypatch.setattr(adapter, "prepare_motion", forbidden)
    monkeypatch.setattr(library, "_sample_skel_state", forbidden)
    as_tensor = torch.as_tensor

    def device_tensor_only(value, **kwargs):
        assert isinstance(value, torch.Tensor)
        result = as_tensor(value, **kwargs)
        assert result is value
        return result

    monkeypatch.setattr(torch, "as_tensor", device_tensor_only)
    state, reference_bodies = library.sample_robot_state(ids, times, loop=False)
    if backend == "physx":
        positions, velocities = state.joint_pos, state.joint_vel
        stored_positions, stored_velocities = (
            buffers.joint_pos,
            buffers.joint_vel,
        )
        roots, rotations = state.root_pos, state.root_quat_xyzw
    else:
        positions = state.joint_q if backend == "newton" else state.qpos
        velocities = state.joint_qd if backend == "newton" else state.qvel
        stored_positions = buffers.joint_q if backend == "newton" else buffers.qpos
        stored_velocities = buffers.joint_qd if backend == "newton" else buffers.qvel
        roots, rotations = positions[..., :3], positions[..., 3:7]
    assert positions.device == library.device
    assert (
        stored_positions.untyped_storage().data_ptr()
        == (
            library.adapted_buffers().joint_pos
            if backend == "physx"
            else library.adapted_buffers().joint_q
            if backend == "newton"
            else library.adapted_buffers().qpos
        )
        .untyped_storage()
        .data_ptr()
    )
    torch.testing.assert_close(positions[[0, 2]], stored_positions)
    torch.testing.assert_close(
        velocities,
        torch.stack(
            (stored_velocities[0], stored_velocities.mean(dim=0), stored_velocities[1])
        ),
    )
    torch.testing.assert_close(roots[:, 0], roots.new_tensor([0.0, 1.0, 2.0]))
    half = np.sqrt(0.5)
    expected = [half, 0.0, 0.0, half] if backend == "mujoco" else [0.0, 0.0, half, half]
    torch.testing.assert_close(rotations[1], rotations.new_tensor(expected))
    torch.testing.assert_close(
        reference_bodies.body_pos_w[1], packed.body_pos_w.mean(dim=0)
    )
    torch.testing.assert_close(
        reference_bodies.body_vel_w,
        torch.stack(
            (packed.body_vel_w[0], packed.body_vel_w.mean(dim=0), packed.body_vel_w[1])
        ),
    )

    monkeypatch.setattr(adapter, "pack_sample", forbidden)
    bodies = library.sample_body_state(ids, times, loop=False)
    torch.testing.assert_close(bodies.body_pos_w, reference_bodies.body_pos_w)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("backend", ["physx", "newton", "mujoco"])
def test_motion_velocities_interpolate_between_distinct_frames(backend, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layout = _layout()
    q = np.zeros((3, layout.nq), dtype=np.float32)
    phase = np.array([0, 0.1, 0.5], dtype=np.float32)
    q[:, 0] = phase
    q[:, 3] = np.cos(phase / 2)
    q[:, 6] = np.sin(phase / 2)
    hinge = next(block for block in layout.blocks if block.q_size == 1)
    q[:, hinge.q_offset] = phase
    qd = np.arange(3 * layout.nv, dtype=np.float32).reshape(3, layout.nv) * 0.01
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout, q=q, qd=qd, fps=2.0
    )
    adapter_type = {
        "physx": ke.adapters.physx.PhysXMotionAdapter,
        "newton": ke.adapters.newton.NewtonMotionAdapter,
        "mujoco": ke.adapters.mujoco.MuJoCoMotionAdapter,
    }[backend]
    library = ke.animation.MotionLibrary(
        [motion], adapter=adapter_type(layout), device=device
    )
    times = torch.tensor([[0, 0.125], [0.375, 0.5]], device=device)
    alpha = torch.tensor([[0, 0.25], [0.75, 1]], device=device)
    for sampler, fields in (
        (library.sample_skel_state, ("root_vel", "root_ang_vel")),
        (library.sample_body_state, ("body_vel_w", "body_ang_vel_w")),
        (
            library.sample_arti_state,
            {
                "physx": ("joint_vel", "root_vel", "root_ang_vel"),
                "newton": ("joint_qd",),
                "mujoco": ("qvel",),
            }[backend],
        ),
    ):
        start = sampler(0, frame_ids=0)
        end = sampler(0, frame_ids=1)
        sampled = sampler(0, times=times)
        held = sampler(0, times=times, interpolate=False)
        for field in fields:
            v0, v1 = getattr(start, field), getattr(end, field)
            assert not torch.allclose(v0, v1), field
            blend = alpha.reshape(*alpha.shape, *([1] * v0.ndim))
            expected = (1 - blend) * v0 + blend * v1
            torch.testing.assert_close(getattr(sampled, field), expected)
            torch.testing.assert_close(getattr(held, field)[0, 1], v0)
            torch.testing.assert_close(getattr(held, field)[1, 1], v1)


def test_frame_selection_broadcasts_clamps_and_loops():
    layout = _layout()
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="slow"), _motion(layout, name="fast", fps=4.0)]
    )
    frames = library.sample_skel_state([[0], [1]], [-0.25, 0.125, 1.0], loop=False)
    assert frames.motion_ids.shape == (2, 3)
    torch.testing.assert_close(
        frames.frame_indices, torch.tensor([[0, 0, 1], [0, 0, 1]])
    )
    torch.testing.assert_close(
        frames.blend, torch.tensor([[0.0, 0.25, 0.0], [0.0, 0.5, 0.0]])
    )
    looped = library.sample_skel_state(0, [-0.25, 0.5, 0.75])
    torch.testing.assert_close(looped.blend, torch.tensor([0.5, 0.0, 0.5]))
    exact = library.sample_skel_state(0, 0.25, interpolate=False)
    assert exact.frame_indices == exact.next_frame_indices == 0
    assert exact.blend == 0.0


def test_body_only_selection_without_precomputation_skips_adapter(monkeypatch):
    layout = _layout()
    adapter = ke.adapters.physx.PhysXMotionAdapter(layout)
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")], adapter=adapter, precompute_kinematics=False
    )
    expected = library.forward_kinematics(
        library.sample_skel_state(0, 0.25, loop=False)
    )

    def forbidden(**kwargs):
        raise AssertionError("body-only FK must not sample backend coordinates")

    monkeypatch.setattr(adapter, "pack_sample", forbidden)
    result = library.sample_body_state(0, 0.25, loop=False)
    torch.testing.assert_close(result.body_pos_w, expected.body_pos_w)
    assert result.body_vel_w is None


@pytest.mark.parametrize("backend", ["physx", "newton", "mujoco"])
def test_native_hinge_interpolation_preserves_scalar_coordinates(backend):
    layout = _layout()
    motion = _motion(layout, name="wrap")
    q = motion.q.copy()
    hinge = next(
        block
        for block in layout.blocks
        if block.q_size == 1
        and sum(other.body_index == block.body_index for other in layout.blocks) == 1
    )
    q[:, hinge.q_offset] = np.deg2rad([170.0, -170.0])
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=motion.qd,
        fps=2.0,
    )
    adapter_types = {
        "physx": ke.adapters.physx.PhysXMotionAdapter,
        "newton": ke.adapters.newton.NewtonMotionAdapter,
        "mujoco": ke.adapters.mujoco.MuJoCoMotionAdapter,
    }
    adapter = adapter_types[backend](layout)
    library = ke.animation.MotionLibrary([motion], adapter=adapter)
    state = library.sample_arti_state(0, 0.25, loop=False)
    if backend == "physx":
        angle = state.joint_pos[adapter.joint_names.index(hinge.joint_name)]
    elif backend == "newton":
        angle = state.joint_q[hinge.q_offset]
    else:
        angle = state.qpos[hinge.q_offset]
    torch.testing.assert_close(angle, torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_physx_multiaxis_interpolation_uses_native_components(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layout = _layout()
    motion = _motion(layout, name="multiaxis")
    groups = {}
    for block in layout.blocks:
        if block.q_size == 1:
            groups.setdefault(block.body_index, []).append(block)
    group = next(blocks for blocks in groups.values() if len(blocks) > 1)
    q = motion.q.copy()
    q[:, group[0].q_offset] = [1.2, -0.3]
    q[:, group[1].q_offset] = [-0.4, 1.4]
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout, q=q, qd=motion.qd, fps=2.0
    )
    adapter = ke.adapters.physx.PhysXMotionAdapter(layout)
    prepared = adapter.prepare_motion(motion)
    assert prepared.shape == (2, 2 * len(adapter.joint_names) + 13)

    library = ke.animation.MotionLibrary([motion], adapter=adapter, device=device)
    endpoints = library.sample_arti_state(0, frame_ids=[0, 1])
    midpoint = library.sample_arti_state(0, 0.25, loop=False)
    torch.testing.assert_close(midpoint.joint_pos, endpoints.joint_pos.mean(dim=0))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_motion_view_uses_clip_local_frames_and_survives_library_changes(
    device, monkeypatch
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layout = _layout()
    first = _motion(layout, name="first")
    q = first.q[:1].copy()
    q[0, :3] = [10.0, 20.0, 30.0]
    second = ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=first.qd[:1],
        fps=2.0,
        motion_name="second",
    )
    library = ke.animation.MotionLibrary(
        [first, second],
        adapter=ke.adapters.physx.PhysXMotionAdapter(layout),
    )
    source = library.source_buffers()
    view = library.motion(1)
    assert len(view) == 1
    assert source.root_pos.data_ptr() == library.source_buffers().root_pos.data_ptr()
    assert not hasattr(source, "backend_frame_data")
    assert source.body_vel_w is not None

    # The view follows the owner, not a stale slice after repacking or moving.
    library.add_motion(_motion(layout, name="third"))
    library.to(device)

    if device.startswith("cuda"):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            assert len(view) == 1
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    else:
        assert len(view) == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("exact frames must not interpolate")

    monkeypatch.setattr(library._adapter, "pack_sample", forbidden)
    arti, body = view[-1]
    torch.testing.assert_close(
        arti.root_pos, torch.tensor([10.0, 20.0, 30.0], device=device)
    )
    torch.testing.assert_close(body.body_pos_w[0], arti.root_pos)
    assert body.body_vel_w is not None
    for frame_idx in (1, -2):
        with pytest.raises(IndexError):
            view[frame_idx]
    with pytest.raises(TypeError):
        view[0.5]
    for motion_id in (-1, 3):
        with pytest.raises(IndexError):
            library.motion(motion_id)

    ids = torch.tensor([[0], [1]], device=device)
    frames = torch.tensor([0, 1, -1], device=device)
    batch_arti, batch_body = library.sample_robot_state(ids, frame_ids=frames)
    assert batch_arti.root_pos.shape == (2, 3, 3)
    torch.testing.assert_close(batch_arti.root_pos[1], arti.root_pos.expand(3, -1))
    assert batch_body.body_pos_w.shape[:2] == (2, 3)
    scalar_skel = library.sample_skel_state(1, frame_ids=0)
    torch.testing.assert_close(scalar_skel.root_pos, arti.root_pos)


@pytest.mark.parametrize(
    "method",
    [
        "sample_skel_state",
        "sample_arti_state",
        "sample_body_state",
        "sample_robot_state",
    ],
)
def test_motion_queries_require_one_time_or_frame_input(method):
    layout = _layout()
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")],
        adapter=ke.adapters.physx.PhysXMotionAdapter(layout),
    )
    query = getattr(library, method)
    with pytest.raises(ValueError, match="exactly one"):
        query(0)
    with pytest.raises(ValueError, match="exactly one"):
        query(0, 0.25, frame_ids=1)
    with pytest.raises(TypeError, match="integers"):
        query(0, frame_ids=0.5)
    with pytest.raises(IndexError):
        query(library.num_motions, times=0.25)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("truncate_time", [0.0, 0.25, 0.5])
def test_time_sampling_bounds_without_cuda_host_sync(device, truncate_time):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layout = _layout()
    library = ke.animation.MotionLibrary([_motion(layout, name="clip")], device=device)
    ids = torch.zeros((2, 3), dtype=torch.long, device=device)

    def sample():
        times = library.sample_times(ids, truncate_time=truncate_time)
        selected_ids, selected_times = library.sample_motion_times(
            32, truncate_time=truncate_time
        )
        return times, selected_ids, selected_times

    if device.startswith("cuda"):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            times, selected_ids, selected_times = sample()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    else:
        times, selected_ids, selected_times = sample()

    assert times.shape == ids.shape
    assert times.device == ids.device
    assert selected_times.shape == selected_ids.shape == (32,)
    assert selected_times.device == selected_ids.device == ids.device
    for values in (times, selected_times):
        assert torch.all(values >= 0)
        if truncate_time == 0.5:
            torch.testing.assert_close(values, torch.zeros_like(values))
        else:
            assert torch.all(values < 0.5 - truncate_time)


@pytest.mark.parametrize(
    "method",
    [
        "sample_skel_state",
        "sample_arti_state",
        "sample_body_state",
        "sample_robot_state",
    ],
)
@pytest.mark.parametrize("by_frame", [False, True])
def test_cuda_motion_queries_do_not_synchronize_host(method, by_frame):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layout = _layout()
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")],
        adapter=ke.adapters.physx.PhysXMotionAdapter(layout),
        device="cuda:0",
    )
    ids = torch.zeros(16, dtype=torch.long, device="cuda:0")
    times = None if by_frame else torch.full((16,), 0.25, device="cuda:0")
    frames = torch.ones_like(ids) if by_frame else None
    query = getattr(library, method)
    query(ids, times, frame_ids=frames)

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        query(ids, times, frame_ids=frames)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@pytest.mark.parametrize("backend", ["newton", "mujoco"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize(
    "free_root, ball_count", [(False, 0), (True, 0), (False, 1), (True, 3)]
)
def test_native_quaternion_batch_matches_per_joint_slerp(
    tmp_path, backend, device, batch_shape, free_root, ball_count
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path = tmp_path / "quaternion_joints.xml"
    balls = "".join(
        f'<body name="ball{idx}"><joint type="ball" name="ball{idx}"/>'
        '<geom type="sphere" size="0.1"/></body>'
        for idx in range(ball_count)
    )
    path.write_text(
        '<mujoco><worldbody><body name="root"><geom type="sphere" size="0.1"/>'
        '<body name="hinge"><joint name="hinge" axis="0 1 0"/>'
        '<geom type="sphere" size="0.1"/></body>'
        f"{balls}</body></worldbody></mujoco>"
    )
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=free_root
    )
    adapter_type = (
        ke.adapters.newton.NewtonMotionAdapter
        if backend == "newton"
        else ke.adapters.mujoco.MuJoCoMotionAdapter
    )
    adapter = adapter_type(layout)
    rng = np.random.default_rng(17)
    shape = (*batch_shape, layout.nq + layout.nv)
    first = torch.tensor(rng.normal(size=shape), dtype=torch.float32, device=device)
    following = torch.tensor(rng.normal(size=shape), dtype=torch.float32, device=device)
    blend = torch.tensor(
        rng.uniform(size=batch_shape), dtype=torch.float32, device=device
    )

    # The original per-joint path is the reference for both quaternion orders.
    expected = torch.lerp(first, following, blend.unsqueeze(-1))
    for block in layout.blocks:
        if block.type not in (
            ke.animation.ArticulationCoordinateType.FREE,
            ke.animation.ArticulationCoordinateType.SPHERICAL,
        ):
            continue
        offset = block.q_offset + (
            3 if block.type == ke.animation.ArticulationCoordinateType.FREE else 0
        )
        a = torch.nn.functional.normalize(first[..., offset : offset + 4], dim=-1)
        b = torch.nn.functional.normalize(following[..., offset : offset + 4], dim=-1)
        first[..., offset : offset + 4] = a
        following[..., offset : offset + 4] = b
        if backend == "newton":
            a = batched_rotations.quat_xyzw_to_wxyz(a)
            b = batched_rotations.quat_xyzw_to_wxyz(b)
        rotation = batched_rotations.quat_wxyz_slerp(a, b, blend)
        if backend == "newton":
            rotation = batched_rotations.quat_wxyz_to_xyzw(rotation)
        expected[..., offset : offset + 4] = rotation

    def sample():
        return adapter.pack_sample(
            backend_frame_data=first, next_backend_frame_data=following, blend=blend
        )

    sample()
    if device.startswith("cuda"):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            result = sample()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    else:
        result = sample()
    pos, vel = (
        (result.joint_q, result.joint_qd)
        if backend == "newton"
        else (result.qpos, result.qvel)
    )
    torch.testing.assert_close(pos, expected[..., : layout.nq])
    torch.testing.assert_close(vel, expected[..., layout.nq :])


def test_robot_query_resolves_frames_once_and_skeleton_query_skips_adapter(monkeypatch):
    layout = _layout()
    adapter = ke.adapters.physx.PhysXMotionAdapter(layout)
    library = ke.animation.MotionLibrary(
        [_motion(layout, name="clip")], adapter=adapter
    )
    resolve = library._resolve_frames
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(library, "_resolve_frames", tracked)
    arti, body = library.sample_robot_state(0, 0.25, loop=False)
    assert len(calls) == 1
    assert body.body_vel_w is not None

    def forbidden(**kwargs):
        raise AssertionError("skeleton queries must not invoke the adapter")

    monkeypatch.setattr(adapter, "pack_sample", forbidden)
    skel = library.sample_skel_state(0, 0.25, loop=False)
    assert isinstance(skel, ke.animation.SkeletonMotionState)
    assert not hasattr(skel, "backend_state")
    torch.testing.assert_close(skel.root_pos, arti.root_pos)
