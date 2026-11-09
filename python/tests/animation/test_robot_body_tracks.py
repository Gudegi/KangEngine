"""Robot tracks discard source rotations that the robot cannot represent."""

import numpy as np
import pytest
import torch

import kangengine as ke


@pytest.mark.parametrize("backend", ["physx", "newton", "mujoco"])
@pytest.mark.parametrize("precompute", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_robot_tracks_preserve_sources_and_project_unavailable_axes(
    tmp_path, monkeypatch, backend, precompute, device
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = tmp_path / "robot.xml"
    path.write_text(
        '<mujoco><worldbody><body name="root"><geom type="sphere" size=".1"/>'
        '<body name="hinge" pos="0 0 -1"><joint axis="0 1 0"/>'
        '<geom type="sphere" size=".1"/><body name="tip" pos="0 0 -1">'
        '<geom type="sphere" size=".1"/></body></body></body></worldbody></mujoco>'
    )
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=True
    )
    adapter = {
        "physx": ke.adapters.physx.PhysXMotionAdapter,
        "newton": ke.adapters.newton.NewtonMotionAdapter,
        "mujoco": ke.adapters.mujoco.MuJoCoMotionAdapter,
    }[backend](layout)
    roots = np.array([[0, 0, 3], [0.2, 0, 3], [0.6, 0, 3]], dtype=np.float32)
    rotations = np.zeros((3, 3, 4), dtype=np.float32)
    rotations[..., 0] = 1
    angles = np.array([0.2, 0.5, 0.9])
    # The source bends the y-axis hinge around x and animates its fixed child.
    rotations[:, 1, 0] = np.cos(angles / 2)
    rotations[:, 1, 1] = np.sin(angles / 2)
    rotations[:, 2, 0] = np.cos(angles / 4)
    rotations[:, 2, 3] = np.sin(angles / 4)
    motion = ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=data.skeleton_tree,
        root_translations=roots,
        local_rotations_wxyz=rotations,
        fps=10,
        motion_name="source",
    )
    molib = ke.animation.MotionLibrary(
        [motion],
        adapter=adapter,
        precompute_kinematics=precompute,
        keep_source_motions=True,
    )
    molib.add_motion(motion, name="second")
    molib.to(device)
    source = molib.source_buffers()
    torch.testing.assert_close(
        source.local_quat_wxyz[:3], torch.tensor(rotations, device=device)
    )
    np.testing.assert_array_equal(
        molib.source_motion(0).local_rotations_wxyz(), rotations
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("queries must reuse prepared robot tracks")

    monkeypatch.setattr(adapter, "prepare_body_motion", forbidden)
    ids = torch.tensor([0, 1, 1], device=device)
    frames = torch.arange(3, device=device)
    _, body = molib.sample_robot_state(ids, frame_ids=frames)
    expected = torch.tensor(roots, device=device)[:, None, :] + torch.tensor(
        [[0, 0, 0], [0, 0, -1], [0, 0, -2]], device=device
    )
    torch.testing.assert_close(body.body_pos_w, expected, atol=2e-5, rtol=0)
    identity = torch.zeros_like(body.body_quat_wxyz_w)
    identity[..., 0] = 1
    torch.testing.assert_close(body.body_quat_wxyz_w, identity, atol=2e-5, rtol=0)
    midpoint = molib.sample_body_state(1, 0.05, loop=False)
    torch.testing.assert_close(
        midpoint.body_pos_w, expected[:2].mean(dim=0), atol=2e-5, rtol=0
    )

    if precompute:
        buffers = molib.body_buffers()
        assert (
            buffers.body_pos_w.data_ptr() == molib.body_buffers().body_pos_w.data_ptr()
        )
        torch.testing.assert_close(buffers.body_pos_w[:3], expected, atol=2e-5, rtol=0)
        torch.testing.assert_close(
            body.body_ang_vel_w,
            torch.zeros_like(body.body_ang_vel_w),
            atol=2e-5,
            rtol=0,
        )
        assert not torch.allclose(source.body_pos_w[:3], expected)
        assert not torch.allclose(source.body_ang_vel_w[:3], body.body_ang_vel_w)
        torch.testing.assert_close(
            source.body_pos_w[:3],
            torch.tensor(motion.global_positions(), device=device),
        )
    else:
        assert body.body_vel_w is None
        with pytest.raises(RuntimeError, match="precomputed"):
            molib.body_buffers()
