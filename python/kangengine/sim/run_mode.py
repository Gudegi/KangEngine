"""Wall-clock and rendering policy for simulation runners."""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass


class SimulationRunMode(enum.StrEnum):
    """Select how a simulation runner relates simulation time to wall time."""

    HEADLESS_FAST = "headless_fast"
    OFFSCREEN_FAST = "offscreen_fast"
    PACED = "paced"


@dataclass(frozen=True, slots=True)
class SimulationRunConfig:
    """Run policy kept separate from physical simulation timing.

    ``HEADLESS_FAST`` disables rendering for training.
    ``OFFSCREEN_FAST`` renders requested frames without waiting.
    ``PACED`` advances one step at a time and waits only when ahead of wall time.
    """

    mode: SimulationRunMode = SimulationRunMode.HEADLESS_FAST

    def __post_init__(self):
        try:
            mode = SimulationRunMode(self.mode)
        except (TypeError, ValueError):
            choices = ", ".join(candidate.value for candidate in SimulationRunMode)
            raise ValueError(f"mode must be one of: {choices}")
        object.__setattr__(self, "mode", mode)

    @property
    def render_enabled(self) -> bool:
        return self.mode is not SimulationRunMode.HEADLESS_FAST

    @property
    def syncs_to_wall_clock(self) -> bool:
        return self.mode is SimulationRunMode.PACED


class SimulationPacer:
    """Pace externally stepped simulations without introducing catch-up steps."""

    def __init__(self, run_config: SimulationRunConfig):
        self.run_config = run_config
        self._deadline: float | None = None

    def reset(self):
        self._deadline = None

    def wait(self, step_dt: float) -> float:
        """Wait in PACED mode and return the requested sleep duration."""
        if self.run_config.mode is not SimulationRunMode.PACED:
            return 0.0
        step_dt = float(step_dt)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive")

        now = time.perf_counter()
        if self._deadline is None:
            self._deadline = now
        self._deadline += step_dt
        delay = self._deadline - now
        if delay > 0.0:
            time.sleep(delay)
            return delay

        # PACED does not catch up missed wall time. Start a fresh interval.
        self._deadline = now
        return 0.0
