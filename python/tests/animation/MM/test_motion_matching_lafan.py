"""BVH range reader and opt-in LAFAN turning/Geno skin integration checks."""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from kangengine import animation, asset, visual
from kangengine.animation.matching import make_walking_query
from kangengine.utils.math import quat_wxyz_rotate_numpy


EXAMPLES = Path(__file__).parents[3] / "examples/animation"
sys.path.insert(0, str(EXAMPLES))
try:
    spec = importlib.util.spec_from_file_location(
        "lafan_example", EXAMPLES / "motion_matching_lafan.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
finally:
    sys.path.remove(str(EXAMPLES))


def test_resolved_reader_range_units_and_euler_order(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text("""HIERARCHY
ROOT Hips
{
 OFFSET 0 100 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Child
 {
  OFFSET 0 20 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  End Site
  {
   OFFSET 0 0 0
  }
 }
}
MOTION
Frames: 3
Frame Time: 0.016667
0 100 0 0 0 0 0 20 0 0 0 0
10 100 0 90 0 0 0 20 0 0 90 0
20 100 0 0 0 0 0 20 0 0 0 0
""")
    tree, positions, rotations = example._load_range(path, 1, 2)
    assert tree.node_names() == ["Hips", "Child"]
    np.testing.assert_allclose(positions[0], ((0.1, 1, 0), (0, 0.2, 0)), atol=1e-7)
    np.testing.assert_allclose(
        quat_wxyz_rotate_numpy(rotations[0, 0], (1, 0, 0)), (0, 1, 0), atol=1e-6
    )
    np.testing.assert_allclose(
        quat_wxyz_rotate_numpy(rotations[0, 1], (0, 0, 1)), (1, 0, 0), atol=1e-6
    )


@pytest.fixture(scope="module")
def database():
    dataset_path = os.environ.get("KANGENGINE_LAFAN_DATASET")
    if not dataset_path:
        pytest.skip("set KANGENGINE_LAFAN_DATASET to opt into LAFAN integration checks")
    dataset = Path(dataset_path).expanduser()
    if not all((dataset / f"{name}.bvh").is_file() for name, _, _ in example.CLIPS):
        pytest.fail(f"LAFAN integration dataset is missing required BVHs: {dataset}")
    return animation.MotionMatchingDatabase.from_motions(
        example._build_motions(example.parse_args([], default_dataset=dataset)),
        config=example.CONFIG,
        feature_builder=animation.WalkingFeatures(
            position_filter_window=2.0,
            rotation_filter_window=2.0,
        ),
    )


@pytest.mark.parametrize("angle", (90, -90, 180))
def test_abrupt_turn_reaches_requested_facing(database, angle):
    controller = animation.MotionMatchingController(
        database, transition_half_life=0.075
    )
    positions = []
    for frame in range(360):
        desired = 0 if frame < 120 else np.deg2rad(angle)
        state = controller.step_with_query(
            1 / 60,
            query_provider=lambda c: make_walking_query(
                c,
                desired_velocity=1.4 * np.array((np.sin(desired), 0, np.cos(desired))),
                facing_angle=desired,
                half_life=0.2,
            ),
        )
        assert np.isfinite(state.compute_global_matrices()).all()
        positions.append(controller.root_position)
    forward = quat_wxyz_rotate_numpy(controller.heading_rotation_wxyz, (0, 0, 1))
    goal = np.array((np.sin(np.deg2rad(angle)), 0, np.cos(np.deg2rad(angle))))
    assert np.dot(forward, goal) > np.cos(np.deg2rad(15))
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=-1)) < 0.15
    assert (positions[-1] - positions[180]) @ goal > 1.0


def test_geno_bind_baking_and_named_palette(database):
    if not example.DEFAULT_CHARACTER.is_file():
        pytest.skip("external Geno_bind.fbx is required")
    tree = database.motions[0].skeleton_tree
    mesh = asset.FBXLoader.load_skinned_meshes(
        fbx_path=str(example.DEFAULT_CHARACTER), scale=0.01, skeleton_tree=tree
    )[0]
    surface = visual.SkinnedSurfaceAsset.from_fbx(skeleton_tree=tree, mesh_info=mesh)
    state = animation.MotionMatchingController(database).state
    result = animation.cpu_skin(
        bind_positions=surface.positions,
        bind_normals=surface.normals,
        skin_bone_indices=surface.bone_indices,
        skin_bone_weights=surface.bone_weights,
        skin_bone_node_indices=surface.bone_node_indices,
        inverse_bind_matrices=surface.inverse_bind_matrices,
        skeleton_global_matrices=state.compute_global_matrices(),
    )
    vertices = result["positions"]
    assert np.isfinite(vertices).all()
    # The loader must bake bind coordinates once, including this surface path.
    assert -0.08 < vertices[:, 1].min() < 0.08
    assert 1.5 < vertices[:, 1].max() < 1.9
    assert np.ptp(vertices[:, 0]) < 1.0
