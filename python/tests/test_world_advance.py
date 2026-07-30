from __future__ import annotations

import unittest

import kangengine as ke


class KangSimWorldAdvanceTest(unittest.TestCase):
    def setUp(self):
        self.world = ke.sim.KangSimWorld(
            num_envs=1,
            sim_dt=1.0 / 120.0,
            add_ground=False,
        )

    def tearDown(self):
        self.world.release()

    def test_converts_control_duration_to_physics_substeps(self):
        self.world.advance(1.0 / 60.0)
        self.assertAlmostEqual(self.world.sim_time, 2.0 / 120.0)

        self.world.advance(1.0 / 30.0)
        self.assertAlmostEqual(self.world.sim_time, 6.0 / 120.0)

    def test_accumulates_fractional_physics_step(self):
        self.world.advance(1.0 / 240.0)
        self.assertEqual(self.world.sim_time, 0.0)

        self.world.advance(1.0 / 240.0)
        self.assertAlmostEqual(self.world.sim_time, 1.0 / 120.0)

    def test_rejects_invalid_duration(self):
        with self.assertRaises(ValueError):
            self.world.advance(-0.1)
        with self.assertRaises(ValueError):
            self.world.advance(float("nan"))


if __name__ == "__main__":
    unittest.main()
