from pathlib import Path

import numpy as np
import kangengine as ke


_ASSET = (
    Path(__file__).resolve().parents[2] / "assets" / "characters" / "kw" / "kw5.xml"
)


def _layout(order: str, *, free_root: bool = False):
    articulation = ke.asset.MJCFLoader.load(str(_ASSET), order=order)
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=articulation,
        free_root=free_root,
    )
    return articulation, layout


def test_layout_preserves_articulation_traversal_order():
    for order in ("BFS", "DFS"):
        articulation, layout = _layout(order)

        assert articulation.traversal_order == order
        assert layout.traversal_order == order

        body_indices = [block.body_index for block in layout.blocks]
        assert body_indices == sorted(body_indices)
        for block in layout.blocks:
            assert block.body_name == articulation.skeleton_tree.node_name(
                block.body_index
            )
            assert block.parent_body_index == articulation.skeleton_tree.parent_index(
                block.body_index
            )


def test_layout_blocks_capture_reference_pose_and_limits():
    articulation, layout = _layout("DFS")

    for block in layout.blocks:
        tree = articulation.skeleton_tree
        reference_translation = tree.local_translation(block.body_index)
        reference_rotation = tree.local_rotation(block.body_index)
        assert np.allclose(
            [
                block.reference_translation.x,
                block.reference_translation.y,
                block.reference_translation.z,
            ],
            [reference_translation.x, reference_translation.y, reference_translation.z],
        )
        assert np.allclose(
            [
                block.reference_rotation.w,
                block.reference_rotation.x,
                block.reference_rotation.y,
                block.reference_rotation.z,
            ],
            [
                reference_rotation.w,
                reference_rotation.x,
                reference_rotation.y,
                reference_rotation.z,
            ],
        )
        assert block.lower_limit <= block.upper_limit


def test_layout_offsets_and_free_root_are_manifold_aware():
    _, fixed_layout = _layout("DFS")
    _, free_layout = _layout("DFS", free_root=True)

    assert free_layout.blocks[0].type == ke.animation.ArticulationCoordinateType.FREE
    assert free_layout.blocks[0].q_offset == 0
    assert free_layout.blocks[0].q_size == 7
    assert free_layout.blocks[0].qd_offset == 0
    assert free_layout.blocks[0].qd_size == 6
    assert free_layout.nq == fixed_layout.nq + 7
    assert free_layout.nv == fixed_layout.nv + 6

    q_offset = 0
    qd_offset = 0
    for block in free_layout.blocks:
        assert block.q_offset == q_offset
        assert block.qd_offset == qd_offset
        q_offset += block.q_size
        qd_offset += block.qd_size
    assert q_offset == free_layout.nq
    assert qd_offset == free_layout.nv


def test_layout_model_signature_is_stable_and_order_sensitive():
    _, bfs_a = _layout("BFS", free_root=True)
    _, bfs_b = _layout("BFS", free_root=True)
    _, dfs = _layout("DFS", free_root=True)

    assert bfs_a.model_signature == bfs_b.model_signature
    assert bfs_a.model_signature != dfs.model_signature


def test_mapper_converts_reference_pose_to_canonical_q():
    articulation, layout = _layout("DFS", free_root=True)
    tree = articulation.skeleton_tree
    rotations = np.asarray(
        [
            [
                tree.local_rotation(i).w,
                tree.local_rotation(i).x,
                tree.local_rotation(i).y,
                tree.local_rotation(i).z,
            ]
            for i in range(tree.num_joints())
        ],
        dtype=np.float32,
    )
    root_translation = tree.local_translation(0)
    state = ke.animation.SkeletonState.from_rotation_and_root_translation(
        tree=tree,
        rotations_wxyz=rotations,
        root_translation=[root_translation.x, root_translation.y, root_translation.z],
    )
    result = ke.animation.ArticulationMotionMapper(layout).to_articulation_coordinates(state)
    q = np.asarray(result.q, dtype=np.float32)

    assert q.shape == (layout.nq,)
    root = layout.blocks[0]
    assert np.allclose(
        q[root.q_offset : root.q_offset + 3],
        [
            state.root_translation().x,
            state.root_translation().y,
            state.root_translation().z,
        ],
    )
    assert np.allclose(
        q[root.q_offset + 3 : root.q_offset + 7],
        [
            state.rotation(0).w,
            state.rotation(0).x,
            state.rotation(0).y,
            state.rotation(0).z,
        ],
    )
    for block in layout.blocks[1:]:
        if block.type == ke.animation.ArticulationCoordinateType.REVOLUTE:
            assert abs(q[block.q_offset]) < 1e-5
    assert max(result.residual_angles) < 1e-5


def test_mapper_round_trips_canonical_q_through_skeleton_state():
    articulation, layout = _layout("DFS", free_root=True)
    tree = articulation.skeleton_tree
    rotations = np.asarray(
        [
            [
                tree.local_rotation(i).w,
                tree.local_rotation(i).x,
                tree.local_rotation(i).y,
                tree.local_rotation(i).z,
            ]
            for i in range(tree.num_joints())
        ],
        dtype=np.float32,
    )
    root_translation = tree.local_translation(0)
    reference = ke.animation.SkeletonState.from_rotation_and_root_translation(
        tree=tree,
        rotations_wxyz=rotations,
        root_translation=[root_translation.x, root_translation.y, root_translation.z],
    )
    mapper = ke.animation.ArticulationMotionMapper(layout)
    q = np.asarray(mapper.to_articulation_coordinates(reference).q, dtype=np.float32)
    q[0] += 0.25
    revolute = next(
        block
        for block in layout.blocks
        if block.type == ke.animation.ArticulationCoordinateType.REVOLUTE
    )
    q[revolute.q_offset] = 0.1

    reconstructed = mapper.to_skeleton_state(q)
    round_trip = np.asarray(
        mapper.to_articulation_coordinates(reconstructed).q, dtype=np.float32
    )
    assert np.allclose(round_trip, q, atol=2e-4)


def test_mapper_clamps_out_of_limit_pose_and_reports_residual():
    articulation, layout = _layout("DFS", free_root=True)
    tree = articulation.skeleton_tree
    rotations = np.asarray(
        [
            [
                tree.local_rotation(i).w,
                tree.local_rotation(i).x,
                tree.local_rotation(i).y,
                tree.local_rotation(i).z,
            ]
            for i in range(tree.num_joints())
        ],
        dtype=np.float32,
    )
    root_translation = tree.local_translation(0)
    reference = ke.animation.SkeletonState.from_rotation_and_root_translation(
        tree=tree,
        rotations_wxyz=rotations,
        root_translation=[root_translation.x, root_translation.y, root_translation.z],
    )
    mapper = ke.animation.ArticulationMotionMapper(layout)
    q = np.asarray(mapper.to_articulation_coordinates(reference).q, dtype=np.float32)
    block = next(
        block
        for block in layout.blocks
        if block.type == ke.animation.ArticulationCoordinateType.REVOLUTE
        and block.lower_limit >= 0.0
    )
    q[block.q_offset] = block.lower_limit - 0.1
    outside = mapper.to_skeleton_state(q)
    mapped = mapper.to_articulation_coordinates(outside, clamp_to_limits=True)

    assert abs(mapped.q[block.q_offset] - block.lower_limit) < 1e-6
    assert mapped.residual_angles[block.body_index] > 0.09

    unconstrained = mapper.to_articulation_coordinates(outside)
    assert abs(unconstrained.q[block.q_offset] - q[block.q_offset]) < 1e-5
    assert unconstrained.residual_angles[block.body_index] < 1e-5


def test_mapper_converts_motion_and_computes_canonical_qd():
    articulation, layout = _layout("DFS", free_root=True)
    tree = articulation.skeleton_tree
    frames = 3
    rotations = np.asarray(
        [
            [
                tree.local_rotation(i).w,
                tree.local_rotation(i).x,
                tree.local_rotation(i).y,
                tree.local_rotation(i).z,
            ]
            for i in range(tree.num_joints())
        ],
        dtype=np.float32,
    )
    rotations = np.repeat(rotations[None, :, :], frames, axis=0)
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = [0.0, 0.5, 1.0]
    source = ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=tree,
        root_translations=roots,
        local_rotations_wxyz=rotations,
        fps=10.0,
        motion_name="linear root",
    )

    mapper = ke.animation.ArticulationMotionMapper(layout)
    result = mapper.to_articulation_motion(source)
    mapped = result.motion
    assert mapped.q.shape == (frames, layout.nq)
    assert mapped.qd.shape == (frames, layout.nv)
    assert np.allclose(mapped.qd[:, :3], [[5.0, 0.0, 0.0]] * frames)
    assert np.allclose(mapped.qd[:, 3:], 0.0, atol=1e-5)
    assert len(result.residual_angles) == frames * tree.num_joints()
    assert max(result.residual_angles) < 1e-5
    reconstructed = mapper.to_skeleton_motion(mapped)
    assert reconstructed.num_frames() == source.num_frames()
    assert reconstructed.fps() == source.fps()
    remapped = mapper.to_articulation_motion(reconstructed).motion
    assert np.allclose(remapped.q, mapped.q, atol=2e-5)
    assert np.allclose(remapped.qd, mapped.qd, atol=2e-4)


def test_motion_owns_float32_arrays_and_exposes_mutable_views():
    _, layout = _layout("DFS", free_root=True)
    frames = 4
    q = np.arange(frames * layout.nq, dtype=np.float64).reshape(frames, layout.nq)
    qd = np.arange(frames * layout.nv, dtype=np.float64).reshape(frames, layout.nv)

    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=60.0,
        motion_name="test",
    )

    assert motion.num_frames() == frames
    assert motion.fps() == 60.0
    assert np.isclose(motion.duration(), (frames - 1) / 60.0)
    assert motion.motion_name() == "test"
    assert motion.layout.model_signature == layout.model_signature
    assert motion.q.shape == (frames, layout.nq)
    assert motion.qd.shape == (frames, layout.nv)
    assert motion.q.dtype == np.float32
    assert motion.qd.dtype == np.float32
    assert motion.q.flags.c_contiguous
    assert motion.qd.flags.c_contiguous

    motion.q[0, 0] = 123.0
    motion.qd[0, 0] = 456.0
    assert motion.q[0, 0] == 123.0
    assert motion.qd[0, 0] == 456.0
