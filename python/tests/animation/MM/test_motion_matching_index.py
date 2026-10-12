"""Native exact search parity, ownership, filtering, and runtime integration."""

from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np

from kangengine.animation import (
    MotionMatch,
    MotionMatchingController,
    MotionMatchingDatabase,
    MotionMatchingIndex,
)
from test_motion_matching import make_config, make_motion


def make_index(features, counts=(4, 3), fps=(30.0, 60.0)):
    return MotionMatchingIndex(
        features,
        clip_frame_counts=[count + 3 for count in counts],
        clip_candidate_counts=counts,
        clip_fps=fps,
    )


class MotionMatchingIndexTests(unittest.TestCase):
    def test_owned_contiguous_copy_and_noncontiguous_query(self):
        original = np.arange(42.0).reshape(7, 6)[:, ::2]
        query = original[5].copy()
        index = make_index(original)
        original[:] = -1000
        result = index.search(np.repeat(query, 2)[::2])
        self.assertEqual(
            (index.num_frames, index.feature_dimension, index.num_motions), (7, 3, 2)
        )
        self.assertEqual(
            (result.row, result.motion_index, result.frame_index), (5, 1, 1)
        )
        self.assertEqual(result.time, 1 / 60)
        self.assertEqual(result.cost, 0.0)

    def test_ties_bias_half_frame_clip_tail_and_filters(self):
        index = make_index(np.zeros((7, 3)))
        self.assertEqual(index.search([0, 0, 0]).row, 0)
        result = index.search(
            [0, 0, 0],
            current_motion_index=1,
            current_time=1.5 / 60,
            continuation_bias=0.1,
        )
        self.assertEqual(result.row, 6)
        self.assertEqual(result.cost, -0.1)
        for time in (2.5 / 60, 5 / 60):
            self.assertEqual(
                index.search(
                    [0, 0, 0],
                    current_motion_index=1,
                    current_time=time,
                    continuation_bias=10,
                ).row,
                0,
            )
        self.assertEqual(index.search([0, 0, 0], allowed_motion_indices=[1]).row, 4)
        self.assertEqual(
            index.search([0, 0, 0], allowed_motion_indices=[1, 0, 1]).row, 0
        )
        self.assertEqual(
            index.search(
                [0, 0, 0],
                current_motion_index=0,
                current_time=0,
                continuation_bias=100,
                allowed_motion_indices=[1],
            ).row,
            4,
        )

    def test_constructor_rejects_bad_shapes_metadata_and_values(self):
        for features in (
            [],
            np.zeros((7,)),
            np.zeros((7, 0)),
            np.zeros((6, 3)),
            np.full((7, 3), np.nan),
        ):
            with self.subTest(shape=np.shape(features)), self.assertRaises(ValueError):
                make_index(features)
        for kwargs in (
            dict(clip_frame_counts=[4], clip_candidate_counts=[4], clip_fps=[30]),
            dict(clip_frame_counts=[4], clip_candidate_counts=[0], clip_fps=[30]),
            dict(clip_frame_counts=[4], clip_candidate_counts=[3], clip_fps=[]),
            dict(clip_frame_counts=[4], clip_candidate_counts=[3], clip_fps=[0]),
            dict(clip_frame_counts=[4], clip_candidate_counts=[3], clip_fps=[np.nan]),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MotionMatchingIndex(np.zeros((3, 2)), **kwargs)

    def test_search_rejects_invalid_inputs_and_distance_overflow(self):
        index = make_index(np.zeros((7, 3)))
        for query in ([0, 0], [[0, 0, 0]], [0, np.nan, 0], [1e308, 0, 0]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                index.search(query)
        for kwargs in (
            dict(current_motion_index=0),
            dict(current_time=0),
            dict(continuation_bias=1),
            dict(continuation_bias=-1),
            dict(continuation_bias=np.inf),
            dict(current_motion_index=2, current_time=0),
            dict(current_motion_index=0, current_time=-1),
            dict(current_motion_index=0, current_time=1),
            dict(current_motion_index=0, current_time=np.nan),
            dict(allowed_motion_indices=[]),
            dict(allowed_motion_indices=[2]),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                index.search([0, 0, 0], **kwargs)

    def test_large_index_matches_independent_brute_force_and_parallel_queries(self):
        rng = np.random.default_rng(124)
        features = rng.normal(size=(10003, 27))
        queries = rng.normal(size=(24, 27))
        index = make_index(features, counts=(5000, 5003))
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(index.search, queries))
        for query, result in zip(queries, results):
            costs = np.sum((features - query) ** 2, axis=1)
            row = int(np.argmin(costs))
            self.assertEqual(result.row, row)
            self.assertAlmostEqual(result.cost, costs[row], places=11)


class MotionMatchingBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = MotionMatchingDatabase.from_motions(
            [
                make_motion(animated=True),
                make_motion(animated=True, fps=60, velocity=(2, 0, 0)),
            ],
            config=make_config(feature_weights=(0, 2, 0.5, 3)),
        )

    def test_reference_and_native_queries_bias_and_filter_parity(self):
        database = self.database
        rng = np.random.default_rng(72)
        for current_motion in range(2):
            for time in (0, 0.5, 1, 1.7, 2):
                query = database.make_query(
                    current_motion,
                    time,
                    trajectory_positions=rng.normal(size=(3, 3)),
                    trajectory_directions=rng.normal(size=(3, 3)),
                )
                for allowed in (None, [0], [1], [1, 0, 1]):
                    kwargs = dict(
                        current_motion_index=current_motion,
                        current_time=time,
                        continuation_bias=0.2,
                        allowed_motion_indices=allowed,
                    )
                    native = database.search(query, **kwargs)
                    reference = database.search(query, backend="python", **kwargs)
                    self.assertIs(type(native), MotionMatch)
                    self.assertIs(type(reference), MotionMatch)
                    self.assertEqual(native.row, reference.row)
                    self.assertEqual(
                        database.motion_indices[native.row], native.motion_index
                    )
                    self.assertEqual(
                        database.frame_indices[native.row], native.frame_index
                    )
                    self.assertEqual(
                        (native.motion_index, native.frame_index, native.time),
                        (reference.motion_index, reference.frame_index, reference.time),
                    )
                    self.assertAlmostEqual(native.cost, reference.cost, places=10)

    def test_result_has_one_public_type_and_readonly_value_semantics(self):
        from kangengine.animation.matching import MotionMatch as MatchingMotionMatch

        self.assertIs(MatchingMotionMatch, MotionMatch)
        result = self.database.native.search(self.database.features[0])
        self.assertIs(type(result), MotionMatch)
        copy = MotionMatch(
            row=result.row,
            motion_index=result.motion_index,
            frame_index=result.frame_index,
            time=result.time,
            cost=result.cost,
        )
        self.assertEqual(result, copy)
        self.assertEqual(hash(result), hash(copy))
        self.assertIn("MotionMatch(row=", repr(result))
        for name in ("row", "motion_index", "frame_index", "time", "cost"):
            with self.subTest(field=name), self.assertRaises(AttributeError):
                setattr(result, name, 1)

    def test_both_backends_reject_invalid_inputs(self):
        for backend in ("native", "python"):
            for kwargs in (
                dict(allowed_motion_indices=[]),
                dict(allowed_motion_indices=[-1]),
                dict(allowed_motion_indices=[2]),
                dict(allowed_motion_indices=[0.5]),
                dict(current_motion_index=-1, current_time=0),
                dict(current_motion_index=0, current_time=-1),
                dict(current_motion_index=0, current_time=np.nan),
                dict(current_motion_index=0, current_time=np.inf),
                dict(current_motion_index=0, current_time=100),
            ):
                with (
                    self.subTest(backend=backend, kwargs=kwargs),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    self.database.search(
                        self.database.features[0], backend=backend, **kwargs
                    )
        with self.assertRaises(ValueError):
            self.database.search(self.database.features[0], backend="invalid")
        with self.assertRaises(ValueError):
            MotionMatchingController(self.database, search_backend="invalid")

    def test_controllers_share_index_and_match_playback(self):
        native = MotionMatchingController(self.database)
        reference = MotionMatchingController(self.database, search_backend="python")
        self.assertIs(native.database.native, reference.database.native)
        self.assertEqual(native.search_backend, "native")
        for step in range(80):
            velocity = (0.8, 0, 0) if step < 40 else (2, 0.3, 0)
            native_state = native.step(0.037, desired_velocity=velocity)
            reference_state = reference.step(0.037, desired_velocity=velocity)
            self.assertEqual(native.motion_index, reference.motion_index)
            self.assertAlmostEqual(native.time, reference.time)
            np.testing.assert_allclose(
                native_state.compute_global_matrices(),
                reference_state.compute_global_matrices(),
                atol=1e-6,
            )
        self.assertEqual(native.transition_count, reference.transition_count)


if __name__ == "__main__":
    unittest.main()
