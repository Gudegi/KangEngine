"""Newton articulation motion buffer conversion tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.adapters.newton import NewtonMotionAdapter


def _kw5():
    import newton

    path = Path(__file__).parents[4] / "assets/characters/kw/kw5.xml"
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(path),
        floating=True,
        ignore_names=["floor", "ground"],
    )
    model = builder.finalize(device="cpu")
    return layout, model


def test_newton_adapter_round_trips_kw5_d6_and_quaternion_layout():
    layout, model = _kw5()
    rng = np.random.default_rng(7)
    q = np.zeros((4, layout.nq), dtype=np.float32)
    qd = rng.normal(size=(4, layout.nv)).astype(np.float32)
    root = layout.blocks[0]
    q[:, root.q_offset : root.q_offset + 3] = rng.normal(size=(4, 3))
    quaternion = rng.normal(size=(4, 4)).astype(np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    q[:, root.q_offset + 3 : root.q_offset + 7] = quaternion
    for block in layout.blocks[1:]:
        if block.q_size:
            q[:, block.q_offset : block.q_offset + block.q_size] = rng.uniform(
                low=-0.1,
                high=0.1,
                size=(4, block.q_size),
            )
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=60.0,
        motion_name="Newton round trip",
    )

    adapter = NewtonMotionAdapter(layout)
    adapter.validate_model(model)
    buffers = adapter.pack(motion)
    assert buffers.joint_q.shape == (4, layout.nq)
    assert buffers.joint_qd.shape == (4, layout.nv)
    assert np.allclose(buffers.joint_q[:, 3:7], quaternion[:, [1, 2, 3, 0]])

    restored = adapter.unpack(
        buffers.joint_q,
        buffers.joint_qd,
        fps=motion.fps(),
        motion_name=motion.motion_name(),
    )
    assert np.allclose(restored.q, motion.q, atol=1e-6)
    assert np.allclose(restored.qd, motion.qd, atol=1e-6)


def test_motion_library_samples_newton_native_state():
    layout, _ = _kw5()
    q = np.zeros((2, layout.nq), dtype=np.float32)
    qd = np.zeros((2, layout.nv), dtype=np.float32)
    root = layout.blocks[0]
    q[:, root.q_offset + 3] = 1.0
    for block in layout.blocks:
        if block.type == ke.animation.ArticulationCoordinateType.REVOLUTE:
            q[1, block.q_offset] = 0.1
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=60.0,
        motion_name="Newton library",
    )
    adapter = NewtonMotionAdapter(layout)
    library = ke.animation.MotionLibrary([motion], adapter=adapter)
    expected = adapter.pack(motion)

    sample = library.sample_frames(0, 1)
    assert sample.backend_state is not None
    np.testing.assert_allclose(
        sample.backend_state.joint_q, expected.joint_q[1], atol=1e-6
    )
    np.testing.assert_allclose(
        sample.backend_state.joint_qd,
        expected.joint_qd[1],
        atol=1e-6,
    )
