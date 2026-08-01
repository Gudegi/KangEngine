from __future__ import annotations

import unittest

import kangengine as ke


class FixedStepClockTest(unittest.TestCase):
    def test_accumulates_wall_time_at_fixed_rate(self):
        clock = ke.FixedStepClock()
        clock.set_step_hz(60.0)

        self.assertEqual(clock.advance(1.0 / 120.0), 0)
        self.assertEqual(clock.advance(1.0 / 120.0), 1)
        self.assertEqual(clock.advance(1.0 / 30.0), 2)
        self.assertAlmostEqual(clock.get_accumulator(), 0.0, places=9)

    def test_bounds_catch_up_and_records_dropped_time(self):
        clock = ke.FixedStepClock()
        clock.set_step_hz(60.0)
        clock.set_max_catch_up_steps(2)

        self.assertEqual(clock.advance(0.1), 2)
        self.assertAlmostEqual(
            clock.get_dropped_wall_time(),
            4.0 / 60.0,
            places=9,
        )

    def test_pause_discards_backlog_and_single_steps_once(self):
        clock = ke.FixedStepClock()
        clock.set_step_hz(60.0)
        clock.set_paused(True)

        self.assertEqual(clock.advance(1.0), 0)
        clock.request_single_step()
        self.assertEqual(clock.advance(0.0), 1)
        self.assertEqual(clock.advance(1.0), 0)

        clock.set_paused(False)
        self.assertEqual(clock.advance(1.0 / 60.0), 1)

    def test_clamps_large_frame_delta(self):
        clock = ke.FixedStepClock()
        clock.set_step_hz(60.0)
        clock.set_max_frame_delta(0.05)

        self.assertEqual(clock.advance(0.2), 3)
        self.assertAlmostEqual(
            clock.get_dropped_wall_time(),
            0.15,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
