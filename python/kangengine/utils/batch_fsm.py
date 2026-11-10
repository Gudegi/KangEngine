"""Signal-driven state transitions for independent batch entries."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class FSMTransition:
    """A named edge; higher priority wins, then definition order.

    source/target are stable state IDs, condition is a signal name, and
    min_duration is the minimum time in the source state in seconds.
    """

    id: str
    source: str
    target: str
    condition: str
    priority: int = 0
    min_duration: float = 0.0


class BatchFSM:
    """Evaluate at most one state transition per environment per step.

    states maps stable IDs to display labels. Definitions are immutable and
    separate from runtime tensors, so tools can inspect nodes and edges.
    Signals are boolean [N] tensors on device; the caller supplies their meaning.
    state_ids, elapsed, and transition_ids are borrowed, read-only runtime views.
    transition_ids indexes definitions, or is -1 when no transition occurred.
    """

    def __init__(
        self,
        states: Mapping[str, str],
        transitions: Sequence[FSMTransition],
        *,
        initial_state: str,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.states = MappingProxyType(dict(states))
        self.transitions = tuple(transitions)
        self._state_ids = {name: idx for idx, name in enumerate(states)}
        self._initial_state = initial_state
        if len({edge.id for edge in self.transitions}) != len(self.transitions):
            raise ValueError("transition IDs must be unique")
        if any(edge.min_duration < 0 for edge in self.transitions):
            raise ValueError("min_duration must be nonnegative")

        # Resolve names and priority once; per-step work stays on the tensor device.
        self._edges = tuple(
            (idx, self.state_idx(edge.source), self.state_idx(edge.target), edge)
            for idx, edge in sorted(
                enumerate(self.transitions), key=lambda item: -item[1].priority
            )
        )
        self.state_ids = torch.full(
            (num_envs,), self.state_idx(initial_state), dtype=torch.long, device=device
        )
        self.elapsed = torch.zeros(num_envs, device=device)
        self.transition_ids = torch.full_like(self.state_ids, -1)

    def state_idx(self, state_id: str) -> int:
        """Resolve a stable state ID to its runtime tensor index."""
        return self._state_ids[state_id]

    @torch.no_grad()
    def reset(
        self, env_ids: torch.Tensor | None = None, *, state: str | None = None
    ) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.state_ids[env_ids] = self.state_idx(
            self._initial_state if state is None else state
        )
        self.elapsed[env_ids] = 0
        self.transition_ids[env_ids] = -1

    @torch.no_grad()
    def step(self, dt: float, signals: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Advance by nonnegative seconds and return a new [N] changed mask.

        All edges read the state at step entry. Newly entered states start at
        elapsed=0 and cannot transition again until the next call.
        Every declared condition must have a signal, even for inactive states.
        """
        elapsed = self.elapsed + dt
        next_state = self.state_ids.clone()
        transition_ids = torch.full_like(self.transition_ids, -1)
        for idx, source, target, edge in self._edges:
            take = (
                (self.state_ids == source)
                & (transition_ids == -1)
                & (elapsed >= edge.min_duration)
                & signals[edge.condition]
            )
            next_state = torch.where(take, target, next_state)
            transition_ids = torch.where(take, idx, transition_ids)

        changed = transition_ids != -1
        self.state_ids.copy_(next_state)
        self.transition_ids.copy_(transition_ids)
        self.elapsed.copy_(torch.where(changed, 0.0, elapsed))
        return changed
