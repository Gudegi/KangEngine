"""MuJoCo articulation motion buffer conversion tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.adapters.mujoco import MuJoCoMotionAdapter


def _kw5():
    import mujoco

    path = Path(__file__).parents[4] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    return data, layout, model


def test_mujoco_adapter_maps_kw5_offsets_and_round_trips():
    data, layout, model = _kw5()
    tree = data.skeleton_tree
    frames = 3
    rotations = np.asarray(
        [
            [
                tree.local_rotation(joint).w,
                tree.local_rotation(joint).x,
                tree.local_rotation(joint).y,
                tree.local_rotation(joint).z,
            ]
            for joint in range(tree.num_joints())
        ],
        dtype=np.float32,
    )
    rotations = np.repeat(rotations[None], frames, axis=0)
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = [0.0, 0.1, 0.2]
    source = ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=tree,
        root_translations=roots,
        local_rotations_wxyz=rotations,
        fps=60.0,
        motion_name="MuJoCo round trip",
    )
    canonical = (
        ke.animation.ArticulationMotionMapper(layout).to_articulation_motion(source).motion
    )

    adapter = MuJoCoMotionAdapter(layout)
    adapter.validate_model(model)
    buffers = adapter.pack(canonical)
    assert buffers.qpos.shape == (frames, layout.nq)
    assert buffers.qvel.shape == (frames, layout.nv)
    assert not np.shares_memory(buffers.qpos, canonical.q)

    restored = adapter.unpack(
        buffers.qpos,
        buffers.qvel,
        fps=canonical.fps(),
        motion_name=canonical.motion_name(),
    )
    assert np.allclose(restored.q, canonical.q, atol=1e-6)
    assert np.allclose(restored.qd, canonical.qd, atol=1e-6)


def test_mujoco_adapter_converts_free_root_angular_velocity_frame():
    _, layout, _ = _kw5()
    q = np.zeros((1, layout.nq), dtype=np.float32)
    qd = np.zeros((1, layout.nv), dtype=np.float32)
    root = layout.blocks[0]
    half = np.sqrt(0.5)
    q[0, root.q_offset : root.q_offset + 7] = [0, 0, 0, half, 0, 0, half]
    qd[0, root.qd_offset + 3 : root.qd_offset + 6] = [1, 0, 0]
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=60.0,
    )

    adapter = MuJoCoMotionAdapter(layout)
    buffers = adapter.pack(motion)
    assert np.allclose(buffers.qvel[0, 3:6], [0, -1, 0], atol=1e-6)

    restored = adapter.unpack(buffers.qpos, buffers.qvel, fps=60.0)
    assert np.allclose(restored.qd, motion.qd, atol=1e-6)
