from unittest.mock import Mock

import numpy as np

from kangengine.animation import SkeletonState, SkeletonTree
from kangengine.asset.smpl import SMPLBody
from kangengine.visual.deformable import SkinnedSurface, SkinnedSurfaceAsset


def test_pose_correctives_from_state_match_raw_wxyz_rotations():
    identity = np.tile([1., 0., 0., 0.], (2, 1)).astype(np.float32)
    tree = SkeletonTree(
        ["root", "child"], [-1, 0], np.zeros((2, 3), np.float32), identity,
    )
    asset = SkinnedSurfaceAsset(
        skeleton_tree=tree,
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        normals=np.tile([0., 0., 1.], (3, 1)).astype(np.float32),
        indices=np.array([0, 1, 2], np.int32),
        bone_indices=np.zeros((3, 4), np.int32),
        bone_weights=np.tile([1., 0., 0., 0.], (3, 1)).astype(np.float32),
        bone_node_indices=np.arange(2, dtype=np.int32),
        inverse_bind_matrices=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    )
    body = SMPLBody(
        skeleton_tree=tree,
        surface_asset=asset,
        joints=np.zeros((2, 3), np.float32),
        discarded_weight_max=0.,
        pose_directions=np.eye(9, dtype=np.float32) * .1,
    )
    # Capture the renderer upload while exercising real surface geometry updates.
    view = Mock()
    surface = SkinnedSurface(view, asset, None, None, None, None)
    rotations = np.array([[4, 1, 2, 3], [1, 2, 3, 4]], np.float32)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    state = SkeletonState.from_rotation_and_root_translation(
        tree=tree, rotations_wxyz=rotations,
        root_translation=(1, 2, 3), is_local=True,
    )

    assert body.update_pose_correctives_from_rotations(surface, rotations) is surface
    expected_positions, expected_normals = view.update_geometry.call_args.args
    assert not np.allclose(expected_positions, asset.positions)

    assert body.update_pose_correctives(surface, state) is surface
    positions, normals = view.update_geometry.call_args.args
    assert view.update_geometry.call_count == 2
    np.testing.assert_allclose(positions, expected_positions, atol=1e-6)
    np.testing.assert_allclose(normals, expected_normals, atol=1e-6)
