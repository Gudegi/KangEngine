"""Shared timing configuration for rendered fixed-step simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationTimingConfig:
    """Rates and safety limits for an App-driven simulation loop.

    Rates are the writable source of truth. Time intervals are derived to
    avoid contradictory ``hz`` and ``dt`` settings.
    """

    render_hz: float = 60.0  # 0 means unlimited
    physics_hz: float = 120.0
    fixed_update_hz: float = 60.0
    max_catch_up_steps: int = 8
    max_frame_delta: float = 0.25

    def __post_init__(self):
        render_hz = self._validate_number("render_hz", self.render_hz, allow_zero=True)
        physics_hz = self._validate_number("physics_hz", self.physics_hz)
        fixed_update_hz = self._validate_number("fixed_update_hz", self.fixed_update_hz)
        try:
            catch_up_steps = int(self.max_catch_up_steps)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("max_catch_up_steps must be a positive integer")
        if (
            isinstance(self.max_catch_up_steps, bool)
            or catch_up_steps < 1
            or catch_up_steps != self.max_catch_up_steps
        ):
            raise ValueError("max_catch_up_steps must be a positive integer")
        max_frame_delta = self._validate_number(
            "max_frame_delta", self.max_frame_delta, allow_zero=True
        )
        object.__setattr__(self, "render_hz", render_hz)
        object.__setattr__(self, "physics_hz", physics_hz)
        object.__setattr__(self, "fixed_update_hz", fixed_update_hz)
        object.__setattr__(self, "max_catch_up_steps", catch_up_steps)
        object.__setattr__(self, "max_frame_delta", max_frame_delta)

    @staticmethod
    def _validate_number(name: str, value: float, *, allow_zero: bool = False) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        if numeric < 0.0 or (not allow_zero and numeric == 0.0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}")
        return numeric

    @property
    def physics_dt(self) -> float:
        """Duration in seconds of one physics substep."""
        return 1.0 / float(self.physics_hz)

    @property
    def sim_dt(self) -> float:
        """Compatibility alias for :attr:`physics_dt`."""
        return self.physics_dt

    @property
    def fixed_dt(self) -> float:
        """Duration in seconds of one App fixed update."""
        return 1.0 / float(self.fixed_update_hz)

    @property
    def decimation(self) -> float:
        """Physics substeps represented by one fixed update."""
        return float(self.physics_hz) / float(self.fixed_update_hz)

    @classmethod
    def from_dt(
        cls,
        *,
        physics_dt: float,
        fixed_dt: float,
        render_hz: float = 60.0,
        max_catch_up_steps: int = 8,
        max_frame_delta: float = 0.25,
    ) -> SimulationTimingConfig:
        """Create a config at a boundary that already expresses time in dt."""
        physics_dt = float(physics_dt)
        fixed_dt = float(fixed_dt)
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError("physics_dt must be finite and positive")
        if not math.isfinite(fixed_dt) or fixed_dt <= 0.0:
            raise ValueError("fixed_dt must be finite and positive")
        return cls(
            render_hz=render_hz,
            physics_hz=1.0 / physics_dt,
            fixed_update_hz=1.0 / fixed_dt,
            max_catch_up_steps=max_catch_up_steps,
            max_frame_delta=max_frame_delta,
        )
