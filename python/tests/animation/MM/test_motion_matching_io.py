"""Minimal NPZ round trips preserve queries, exact search, and playback."""

import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from kangengine.animation import (
    MotionMatchingConfig,
    MotionMatchingController,
    MotionMatchingDatabase,
    SkeletonMotion,
    SkeletonTree,
    WalkingFeatures,
)
from test_motion_matching import make_config, make_motion
from test_motion_matching_custom import DisplacementFeatures


def make_database():
    return MotionMatchingDatabase.from_motions(
        [make_motion(animated=True), make_motion(fps=60, velocity=(2, 0, 0))],
        config=make_config(feature_weights=(0, 0.7, 1.3, 2)),
    )


def read_payload(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def metadata(payload):
    return json.loads(payload["metadata"].tobytes().decode("utf-8"))


def replace_metadata(payload, info):
    payload["metadata"] = np.frombuffer(
        json.dumps(info).encode("utf-8"), dtype=np.uint8
    )


@pytest.mark.parametrize("legacy_unit", ("m", "cm", "mm", "source"))
def test_fixed_meter_config_and_legacy_cache_units(tmp_path, legacy_unit):
    database = make_database()
    path = database.save_npz(tmp_path / "db.npz")
    payload = read_payload(path)
    info = metadata(payload)
    assert "length_unit" not in info["config"]
    info["config"]["length_unit"] = legacy_unit
    replace_metadata(payload, info)
    np.savez(path, **payload)
    with pytest.raises(TypeError, match="length_unit"):
        MotionMatchingDatabase.from_npz(path)


def test_roundtrip_preserves_search_queries_and_playback_without_rebuilding(tmp_path):
    original = make_database()
    path = original.save_npz(tmp_path / "nested" / "walking")
    assert path == (tmp_path / "nested" / "walking.npz").resolve()
    assert not path.with_suffix(".summary.json").exists()
    with patch.object(
        WalkingFeatures, "build_clip", side_effect=AssertionError("must not rebuild")
    ):
        restored = MotionMatchingDatabase.from_npz(path)
    assert restored.config == original.config
    assert restored.schema == original.schema
    for name in (
        "features",
        "mean",
        "scale",
        "weights",
        "motion_indices",
        "frame_indices",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
        assert not getattr(restored, name).flags.writeable
    for i, (before, after) in enumerate(zip(original.motions, restored.motions)):
        assert before.fps() == after.fps()
        assert before.motion_name() == after.motion_name()
        np.testing.assert_array_equal(
            before.root_translations(), after.root_translations()
        )
        np.testing.assert_array_equal(
            before.local_rotations_wxyz(), after.local_rotations_wxyz()
        )
        np.testing.assert_array_equal(
            original.reference_quats_wxyz(i), restored.reference_quats_wxyz(i)
        )
        for time in (0, 0.137, (before.num_frames() - 1) / before.fps()):
            inputs = dict(
                trajectory_positions=np.array(((0.2, 0.1, 0), (0.7, 0, 0), (1, 0, 0))),
                trajectory_directions=np.tile((1, 0, 0), (3, 1)),
            )
            query = original.make_query(i, time, **inputs)
            np.testing.assert_array_equal(query, restored.make_query(i, time, **inputs))
            for backend in ("native", "python"):
                options = dict(
                    current_motion_index=i,
                    current_time=time,
                    continuation_bias=0.03,
                    backend=backend,
                )
                assert original.search(query, **options) == restored.search(
                    query, **options
                )
    first, second = (
        MotionMatchingController(original),
        MotionMatchingController(restored),
    )
    for dt in (0.01, 0.23, 0.08, 0.7):
        a = first.step(dt, desired_velocity=(1.2, 0.3, 0))
        b = second.step(dt, desired_velocity=(1.2, 0.3, 0))
        np.testing.assert_array_equal(
            a.compute_global_matrices(), b.compute_global_matrices()
        )
    # A second save is data-identical; no renormalization or precision drift.
    second_path = restored.save_npz(tmp_path / "again.npz")
    for key, value in read_payload(path).items():
        np.testing.assert_array_equal(value, read_payload(second_path)[key])


def test_payload_has_only_required_arrays_and_deduplicates_skeletons(tmp_path):
    path = make_database().save_npz(tmp_path / "db.npz")
    payload = read_payload(path)
    assert set(payload) == {
        "metadata",
        "parents",
        "rest_translations",
        "rest_rotations",
        "features",
        "mean",
        "scale",
        *(
            f"clip_{i}_{name}"
            for i in range(2)
            for name in ("roots", "rotations", "pose_features", "reference_rotations")
        ),
    }
    assert payload["rest_translations"].shape == (1, 3, 3)
    assert payload["features"].dtype == np.float64
    assert payload["clip_0_rotations"].dtype == np.float32
    assert all(not array.dtype.hasobject for array in payload.values())
    assert set(metadata(payload)) == {
        "format",
        "version",
        "has_reference_track",
        "config",
        "feature_spec",
        "joint_names",
        "clips",
    }
    assert metadata(payload)["version"] == 1
    assert metadata(payload)["has_reference_track"] is False
    with zipfile.ZipFile(path) as archive:
        assert all(
            entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist()
        )


def test_custom_builder_is_supplied_explicitly_and_not_executed_during_load(tmp_path):
    config = MotionMatchingConfig(
        position_joints=(), velocity_joints=(), trajectory_times=(0.2,)
    )
    original = MotionMatchingDatabase.from_motions(
        [make_motion(velocity=(0, 0, 1))],
        config=config,
        feature_builder=DisplacementFeatures(),
    )
    path = original.save_npz(tmp_path / "custom.npz")
    with pytest.raises(ValueError, match="feature schema mismatch"):
        MotionMatchingDatabase.from_npz(path)
    builder = DisplacementFeatures()
    builder.build_clip = lambda *a, **kw: pytest.fail("must not build features on load")
    restored = MotionMatchingDatabase.from_npz(path, feature_builder=builder)
    assert restored._clips[0].pose_features.shape[1] == 0
    query = restored.make_query(0, 0.13, desired_displacement=(0, 0, 0.2))
    assert restored.search(query) == original.search(query)
    describe = builder.describe
    builder.describe = lambda config: {**describe(config), "version": 2}
    with pytest.raises(ValueError, match="schema mismatch"):
        MotionMatchingDatabase.from_npz(path, feature_builder=builder)
    with pytest.raises(ValueError, match="schema changed"):
        restored.save_npz(tmp_path / "drift.npz")
    assert not (tmp_path / "drift.npz").exists()


def test_compatible_rest_variants_and_unicode_are_preserved(tmp_path):
    first = make_motion()
    tree = first.skeleton_tree
    offsets = np.array(
        [[getattr(tree.local_translation(j), a) for a in "xyz"] for j in range(3)]
    )
    offsets[1, 0] += 5e-7  # Compatible, but not identical to the first rig.
    second_tree = SkeletonTree(
        ["root", "left_foot", "right_foot"],
        [-1, 0, 0],
        offsets,
        np.tile((-1, 0, 0, 0), (3, 1)),
    )
    second = SkeletonMotion.from_arrays(
        second_tree,
        first.root_translations(),
        first.local_rotations_wxyz(),
        first.fps(),
        "걷기 테스트",
    )
    original = MotionMatchingDatabase.from_motions(
        [first, second], config=make_config()
    )
    path = original.save_npz(tmp_path / "걷기.npz")
    assert read_payload(path)["rest_translations"].shape[0] == 2
    restored = MotionMatchingDatabase.from_npz(path)
    assert restored.motions[1].motion_name() == "걷기 테스트"
    assert restored.schema == original.schema
    for before, after in zip(original.motions, restored.motions):
        np.testing.assert_array_equal(
            before.frame(0).compute_global_matrices(),
            after.frame(0).compute_global_matrices(),
        )


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "missing",
        "feature_shape",
        "feature_nan",
        "scale_zero",
        "parents",
        "zero_rotation",
        "nonunit_rotation",
        "zero_rest_rotation",
        "zero_reference_rotation",
        "reference_shape",
        "cache_shape",
        "wrong_dtype",
        "object",
    ],
)
def test_invalid_archives_are_rejected(tmp_path, damage):
    path = make_database().save_npz(tmp_path / "db.npz")
    payload = read_payload(path)
    if damage == "version":
        info = metadata(payload)
        info["version"] = 2
        replace_metadata(payload, info)
    elif damage == "missing":
        del payload["scale"]
    elif damage == "feature_shape":
        payload["features"] = payload["features"][:-1]
    elif damage == "feature_nan":
        payload["features"][0, 0] = np.nan
    elif damage == "scale_zero":
        payload["scale"][0] = 0
    elif damage == "parents":
        payload["parents"][1] = 1
    elif damage == "zero_rotation":
        payload["clip_0_rotations"][0, 0] = 0
    elif damage == "nonunit_rotation":
        payload["clip_0_rotations"][0, 0] *= 2
    elif damage == "zero_rest_rotation":
        payload["rest_rotations"][0, 0] = 0
    elif damage == "zero_reference_rotation":
        payload["clip_0_reference_rotations"][0] = 0
    elif damage == "reference_shape":
        payload["clip_0_reference_rotations"] = payload["clip_0_reference_rotations"][
            :-1
        ]
    elif damage == "cache_shape":
        payload["clip_0_pose_features"] = payload["clip_0_pose_features"][:, :-1]
        payload["clip_1_pose_features"] = payload["clip_1_pose_features"][:, :-1]
    elif damage == "wrong_dtype":
        payload["features"] = payload["features"].astype(np.float32)
    elif damage == "object":
        payload["features"] = np.array([{}], dtype=object)
    np.savez(path, **payload)
    with pytest.raises(KeyError if damage == "missing" else ValueError):
        MotionMatchingDatabase.from_npz(path)


@pytest.mark.parametrize("filename", ("walking", "walking.npz"))
def test_optional_summary_is_written_only_on_save(tmp_path, filename):
    database = make_database()
    path = database.save_npz(tmp_path / "nested" / filename, write_summary=True)
    summary_path = path.with_suffix(".summary.json")
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "clips": [
            {"name": "synthetic", "num_frames": 61, "fps": 30.0},
            {"name": "synthetic", "num_frames": 121, "fps": 60.0},
        ]
    }
    summary_path.write_text("human notes", encoding="utf-8")
    MotionMatchingDatabase.from_npz(path)
    assert summary_path.read_text(encoding="utf-8") == "human notes"
    summary_path.unlink()
    MotionMatchingDatabase.from_npz(path)
    assert not summary_path.exists()


def test_failed_save_keeps_existing_archive(tmp_path):
    database = make_database()
    path = database.save_npz(tmp_path / "db.npz")
    before = path.read_bytes()
    with patch(
        "kangengine.animation.matching._io.np.savez", side_effect=OSError("disk full")
    ):
        with pytest.raises(OSError, match="disk full"):
            database.save_npz(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [Path(path)]
