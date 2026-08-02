"""Shared lifecycle and stepping policy for KangEngine simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .world import KangSimWorld

StateAccess = Literal["snapshot", "gpu"]


@dataclass(slots=True)
class ArticulationStateView:
    """Torch tensors for control and reinforcement-learning loops.

    Shapes:
        root_pos: ``(N, 3)``.
        root_rot: ``(N, 4)``.
        root_vel: ``(N, 3)``.
        root_ang_vel: ``(N, 3)``.
        dof_pos: ``(N, D)``.
        dof_vel: ``(N, D)``.
    """

    root_pos: torch.Tensor
    root_rot: torch.Tensor
    root_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor


class SimulationRuntime:
    """Own a :class:`KangSimWorld` and its execution lifecycle.

    Simulation objects must be registered on :attr:`world` before calling
    :meth:`initialize`.  ``state_access="snapshot"`` keeps the conventional
    ``world.state`` snapshot current.  ``state_access="gpu"`` avoids that
    snapshot refresh and updates the canonical CUDA frame cache instead.
    """

    def __init__(
        self,
        *,
        state_access: StateAccess = "snapshot",
        **world_kwargs,
    ):
        if state_access not in ("snapshot", "gpu"):
            raise ValueError(
                f"state_access must be either 'snapshot' or 'gpu', got {state_access!r}"
            )
        self.world = KangSimWorld(**world_kwargs)
        self.state_access: StateAccess = state_access
        self._is_initialized = False
        self._is_closed = False

        if self.state_access == "gpu" and not self.uses_gpu_sim:
            self.world.release()
            self._is_closed = True
            raise ValueError("state_access='gpu' requires sim_device='cuda'")

    @property
    def device(self) -> torch.device:
        return self.world.device

    @property
    def sim_device(self) -> torch.device:
        return self.world.sim_device

    @property
    def uses_gpu_sim(self) -> bool:
        return self.world.sim_device.type == "cuda"

    @property
    def uses_gpu_state(self) -> bool:
        return self.state_access == "gpu"

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def initialize(self) -> SimulationRuntime:
        """Finalize the runtime after all simulation objects are registered."""
        self._require_open()
        if self._is_initialized:
            return self
        if self.uses_gpu_sim:
            device_id = self.sim_device.index
            self.world.init_gpu_system(
                cuda_device_id=0 if device_id is None else device_id
            )
        if self.uses_gpu_state:
            self.world.state.set_strict_snapshot_reads(True)
        self._is_initialized = True
        return self

    def step(
        self,
        substeps: int = 1,
        *,
        apply_commands: bool = True,
        refresh_state: bool = True,
    ) -> object | None:
        """Advance physics and optionally refresh the selected state path."""
        self._require_initialized()
        refresh_snapshot = refresh_state and not self.uses_gpu_state
        state = self.world.step(
            substeps=substeps,
            refresh=refresh_snapshot,
            apply_commands=apply_commands,
        )
        if refresh_state and self.uses_gpu_state:
            self.world.state.gpu.refresh_frame_cache()
        return state

    def flush(self, *, apply_commands: bool = False) -> object | None:
        """Apply queued state without advancing simulation time."""
        return self.step(
            substeps=0,
            apply_commands=apply_commands,
        )

    def get_articulation_state(
        self,
        obj_id: int,
    ) -> ArticulationStateView:
        """Return current articulation tensors using the selected state path."""
        self._require_initialized()
        if self.uses_gpu_state:
            state = self.world.state.gpu
            return ArticulationStateView(
                root_pos=state.get_root_pos(obj_id, fetch=False),
                root_rot=state.get_root_rot(obj_id, fetch=False),
                root_vel=state.get_root_vel(obj_id, fetch=False),
                root_ang_vel=state.get_root_ang_vel(obj_id, fetch=False),
                dof_pos=state.get_dof_pos(obj_id, fetch=False),
                dof_vel=state.get_dof_vel(obj_id, fetch=False),
            )

        state = self.world.state.object_states(obj_id)
        return ArticulationStateView(
            root_pos=state.root_pos,
            root_rot=state.root_rot,
            root_vel=state.root_vel,
            root_ang_vel=state.root_ang_vel,
            dof_pos=state.dof_pos,
            dof_vel=state.dof_vel,
        )

    def set_articulation_state(
        self,
        obj_id: int,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
    ) -> None:
        """Shapes: env IDs ``(N,)``, root state ``(N, 3|4)``, DOFs ``(N, D)``."""
        self._require_initialized()
        if self.uses_gpu_state:
            root_state = torch.cat(
                (root_pos, root_rot, root_vel, root_ang_vel),
                dim=-1,
            ).contiguous()
            dof_state = torch.stack(
                (dof_pos, dof_vel),
                dim=-1,
            ).contiguous()
            self.world.set_gpu_root_state_batch(
                env_ids,
                obj_id,
                root_state,
            )
            self.world.set_gpu_dof_state_batch(
                env_ids,
                obj_id,
                dof_state,
            )
            return

        selected_env_ids = env_ids.detach().cpu().tolist()
        self.world.set_root_state(
            selected_env_ids,
            obj_id,
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
        )
        self.world.set_dof_state(
            selected_env_ids,
            obj_id,
            dof_pos,
            dof_vel,
        )

    def close(self) -> None:
        """Release the owned world. This operation is idempotent."""
        if self._is_closed:
            return
        self.world.release()
        self._is_closed = True

    def _require_open(self):
        if self._is_closed:
            raise RuntimeError("SimulationRuntime is closed")

    def _require_initialized(self):
        self._require_open()
        if not self._is_initialized:
            raise RuntimeError(
                "SimulationRuntime.initialize() must be called after registering simulation objects"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
