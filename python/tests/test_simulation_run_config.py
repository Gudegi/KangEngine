from __future__ import annotations

import unittest

import kangengine as ke


class SimulationRunConfigTest(unittest.TestCase):
    def test_headless_fast_disables_rendering(self):
        config = ke.SimulationRunConfig(mode="headless_fast")

        self.assertEqual(config.mode, ke.SimulationRunMode.HEADLESS_FAST)
        self.assertFalse(config.render_enabled)
        self.assertFalse(config.syncs_to_wall_clock)

    def test_paced_enables_rendering(self):
        config = ke.SimulationRunConfig(mode="paced")

        self.assertTrue(config.render_enabled)
        self.assertTrue(config.syncs_to_wall_clock)

    def test_offscreen_fast_renders_without_wall_clock_sync(self):
        config = ke.SimulationRunConfig(mode="offscreen_fast")

        self.assertEqual(config.mode, ke.SimulationRunMode.OFFSCREEN_FAST)
        self.assertTrue(config.render_enabled)
        self.assertFalse(config.syncs_to_wall_clock)

    def test_defaults_to_headless_fast(self):
        config = ke.SimulationRunConfig()

        self.assertEqual(config.mode, ke.SimulationRunMode.HEADLESS_FAST)
        self.assertFalse(config.render_enabled)

    def test_rejects_invalid_mode(self):
        for mode in ("unknown", None):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    ke.SimulationRunConfig(mode=mode)


if __name__ == "__main__":
    unittest.main()
