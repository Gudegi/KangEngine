from __future__ import annotations

import unittest

import kangengine as ke


class SimulationTimingConfigTest(unittest.TestCase):
    def test_derives_time_intervals_and_decimation(self):
        timing = ke.SimulationTimingConfig(
            render_hz=60.0,
            physics_hz=120.0,
            fixed_update_hz=30.0,
        )

        self.assertAlmostEqual(timing.physics_dt, 1.0 / 120.0)
        self.assertAlmostEqual(timing.sim_dt, timing.physics_dt)
        self.assertAlmostEqual(timing.fixed_dt, 1.0 / 30.0)
        self.assertEqual(timing.decimation, 4.0)

    def test_from_dt_normalizes_boundary_values(self):
        timing = ke.sim.SimulationTimingConfig.from_dt(
            physics_dt=1.0 / 240.0,
            fixed_dt=1.0 / 60.0,
            render_hz=0.0,
        )

        self.assertAlmostEqual(timing.physics_hz, 240.0)
        self.assertAlmostEqual(timing.fixed_update_hz, 60.0)
        self.assertEqual(timing.decimation, 4.0)
        self.assertEqual(timing.render_hz, 0.0)

    def test_allows_fractional_general_app_ratio(self):
        timing = ke.SimulationTimingConfig(
            physics_hz=100.0,
            fixed_update_hz=60.0,
        )

        self.assertAlmostEqual(timing.decimation, 5.0 / 3.0)

    def test_rejects_invalid_values(self):
        for kwargs in (
            {"render_hz": -1.0},
            {"physics_hz": 0.0},
            {"fixed_update_hz": float("nan")},
            {"max_catch_up_steps": 0},
            {"max_catch_up_steps": 2.5},
            {"max_frame_delta": -0.1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ke.SimulationTimingConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
