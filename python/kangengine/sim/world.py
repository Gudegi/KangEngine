"""Small Python simulation host built on KangEngine PhysX bindings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
import torch

from .._core import _ke
from ..rigid import rigid_shape_specs
from ..state import KangWorldState, SimObjectState
from ..utils.env_utils import (
    EnvIdLike,
    env_id_list,
    select_env_value,
    select_optional_env_value,
)

_TORCH = None


def _as_cpu_numpy(value, *, shape=None, dtype=np.float32):
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value, dtype=dtype)
    if dtype is not None and array.dtype != np.dtype(dtype):
        array = np.asarray(array, dtype=dtype)
    return array if shape is None else array.reshape(shape)


def _optional_drive_array(value, size: int):
    if value is None:
        return None
    arr = _as_cpu_numpy(value).reshape(-1)
    if arr.size == 1:
        arr = np.full(size, float(arr[0]), dtype=np.float32)
    if arr.size != size:
        raise ValueError(f"drive parameter expected {size} values, got {arr.size}")
    return arr.astype(np.float32, copy=True)


def _clip_forces(forces, limits):
    limits = _as_cpu_numpy(limits).reshape(-1)
    if limits.size == 1:
        limits = np.full_like(forces, float(limits[0]), dtype=np.float32)
    return np.clip(forces, -limits, limits).astype(np.float32, copy=False)


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


def _require_physx():
    physics = getattr(_ke, "physics", None)
    missing = [
        name
        for name in (
            "PhysicsConfig",
            "PhysicsWorld",
            "ArticulationConfig",
            "Articulation",
        )
        if physics is None or not hasattr(physics, name)
    ]
    if missing:
        raise RuntimeError(
            f"KangSimWorld requires KangEngine PhysX bindings. Missing: {', '.join(missing)}"
        )


class SimDevice(str, Enum):
    """Simulation backend/device selector for KangSimWorld."""

    CPU = "cpu"
    CUDA = "cuda"


def _sim_device_uses_gpu(sim_device) -> bool:
    return _resolve_sim_device(sim_device).type == "cuda"


def _resolve_sim_device(sim_device):
    from ..utils.tensor import resolve_device

    if sim_device is None:
        return resolve_device("cpu")
    if isinstance(sim_device, SimDevice):
        sim_device = sim_device.value
    try:
        device = resolve_device(sim_device)
    except Exception as exc:
        raise ValueError(
            f"sim_device must be 'cpu', 'cuda', or 'cuda:<ordinal>' (got {sim_device!r})"
        ) from exc
    if device.type in ("cpu", "cuda"):
        return device
    raise ValueError(
        f"sim_device must be 'cpu', 'cuda', or 'cuda:<ordinal>' (got {sim_device!r})"
    )


def _resolve_state_device(*, state_device=None, device=None):
    from ..utils.tensor import resolve_device

    if state_device is not None and device is not None:
        resolved_state = resolve_device(state_device)
        resolved_compat = resolve_device(device)
        if _device_key(resolved_state) != _device_key(resolved_compat):
            raise ValueError(
                "KangSimWorld received both state_device and device with "
                f"different values: {resolved_state} != {resolved_compat}"
            )
        return resolved_state
    return resolve_device(state_device if state_device is not None else device)


def _device_key(device):
    if device.type == "cuda":
        return device.type, 0 if device.index is None else int(device.index)
    return device.type, device.index


@dataclass(slots=True)
class SimArticulation:
    """Simulation-facing view for one registered articulation.

    This is the lightweight public object users should grow into using. It
    keeps the native articulation accessible for compatibility, while routing
    common commands and state access through ``KangSimWorld``.
    """

    env_id: int
    obj_id: int
    name: str
    articulation: object
    world: KangSimWorld | None = None

    @property
    def key(self):
        return (self.env_id, self.obj_id)

    @property
    def env_ids(self):
        return (self.env_id,)

    @property
    def data(self) -> SimObjectState:
        return self._require_world().state.object_state(self.env_id, self.obj_id)

    def get_data(self, env_ids: EnvIdLike | None = None) -> SimObjectState:
        if env_ids is None:
            return self.data
        return self._require_world().state.object_states(
            self.obj_id, self._selected_env_ids(env_ids)
        )

    def get_root_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 3)``."""
        return self.get_data(env_ids).root_pos

    def get_root_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 4)``."""
        return self.get_data(env_ids).root_rot

    def get_root_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 3)``."""
        return self.get_data(env_ids).root_vel

    def get_root_ang_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 3)``."""
        return self.get_data(env_ids).root_ang_vel

    def get_body_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., B, 3)``."""
        return self.get_data(env_ids).body_pos

    def get_body_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., B, 4)``."""
        return self.get_data(env_ids).body_rot

    def get_dof_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., D)``."""
        return self.get_data(env_ids).dof_pos

    def get_dof_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., D)``."""
        return self.get_data(env_ids).dof_vel

    def get_dof_force(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., D)``."""
        return self.get_data(env_ids).dof_force

    @property
    def num_dofs(self) -> int:
        return int(self.articulation.num_dofs())

    @property
    def num_bodies(self) -> int:
        return int(self.articulation.num_links())

    @property
    def joint_names(self) -> list[str]:
        return list(self.articulation.get_dof_names())

    @property
    def body_names(self) -> list[str]:
        return [f"body_{i}" for i in range(self.num_bodies)]

    def get_joint_id(self, name: str) -> int:
        return _find_name(self.joint_names, name, "joint")

    def get_body_id(self, name: str) -> int:
        return _find_name(self.body_names, name, "body")

    def add_contact_sensor(self, body_ids=None, *, name: str = ""):
        return self._require_world().add_contact_sensor(
            self, body_ids=body_ids, name=name
        )

    def add_force_sensor(self, body_ids=None, *, name: str = ""):
        return self._require_world().add_force_sensor(
            self, body_ids=body_ids, name=name
        )

    def set_cmd(
        self,
        env_ids: EnvIdLike | None,
        cmd: torch.Tensor,
        mode: "ControlMode | str" = "pos",
        kp: float | torch.Tensor | None = 200.0,
        kd: float | torch.Tensor | None = 10.0,
    ) -> Self:
        """Shape: command and tensor gains ``(..., D)``."""
        self._require_world().set_cmd(
            self._selected_env_ids(env_ids),
            self.obj_id,
            cmd,
            mode,
            kp,
            kd,
        )
        return self

    def clear_cmd(self, env_ids: EnvIdLike | None = None):
        self._require_world().clear_cmd(
            self._selected_env_ids(env_ids),
            self.obj_id,
        )
        return self

    def set_root_state(
        self,
        env_ids: EnvIdLike | None,
        pos: torch.Tensor,
        rot_xyzw: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        immediate: bool = False,
    ) -> Self:
        """Shapes: position/velocity ``(..., 3)``, rotation ``(..., 4)``."""
        self._require_world().set_root_state(
            self._selected_env_ids(env_ids),
            self.obj_id,
            pos,
            rot_xyzw,
            linear_velocity,
            angular_velocity,
            immediate=immediate,
        )
        return self

    def set_dof_state(
        self,
        env_ids: EnvIdLike | None,
        positions: torch.Tensor,
        velocities: torch.Tensor | None = None,
        immediate: bool = False,
    ) -> Self:
        """Shape: ``(..., D)``."""
        self._require_world().set_dof_state(
            self._selected_env_ids(env_ids),
            self.obj_id,
            positions,
            velocities,
            immediate=immediate,
        )
        return self

    def set_body_force(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
    ) -> Self:
        """Shape: ``(..., 3)``."""
        self._require_world().set_body_force(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
        )
        return self

    def set_body_force_at_position(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
        position: torch.Tensor,
    ) -> Self:
        """Shape: force and position ``(..., 3)``."""
        self._require_world().set_body_force_at_position(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
            position,
        )
        return self

    def _selected_env_ids(self, env_ids: EnvIdLike | None):
        if env_ids is None:
            return self.env_id
        selected = tuple(env_id_list(env_ids, self.env_id + 1))
        if selected != (self.env_id,):
            raise KeyError(
                f"SimArticulation only contains env_id={self.env_id}, got {selected}"
            )
        return self.env_id

    def _require_world(self):
        if self.world is None:
            raise RuntimeError("SimArticulation is not attached to a KangSimWorld")
        return self.world


@dataclass(slots=True)
class SimRigid:
    """Simulation-facing view for one registered rigid object."""

    env_id: int
    obj_id: int
    name: str
    rigid: object
    world: KangSimWorld | None = None
    source_data: object | None = None

    @property
    def key(self):
        return (self.env_id, self.obj_id)

    @property
    def env_ids(self):
        return (self.env_id,)

    @property
    def data(self) -> SimObjectState:
        return self._require_world().state.object_state(self.env_id, self.obj_id)

    def get_data(self, env_ids: EnvIdLike | None = None) -> SimObjectState:
        if env_ids is None:
            return self.data
        return self._require_world().state.object_states(
            self.obj_id, self._selected_env_ids(env_ids)
        )

    def get_root_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 3)``."""
        return self.get_data(env_ids).root_pos

    def get_root_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 4)``."""
        return self.get_data(env_ids).root_rot

    def get_root_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 3)``."""
        return self.get_data(env_ids).root_vel

    def get_root_ang_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., 3)``."""
        return self.get_data(env_ids).root_ang_vel

    def get_body_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., B, 3)``."""
        return self.get_data(env_ids).body_pos

    def get_body_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(..., B, 4)``."""
        return self.get_data(env_ids).body_rot

    @property
    def num_bodies(self) -> int:
        if (
            self.world is not None
            and self.key in getattr(self.world, "static_rigids", {})
            and self.source_data is not None
        ):
            return len(rigid_shape_specs(self.source_data))
        if self.world is not None:
            return int(
                self.world.state.record(self.env_id, self.obj_id).cache.num_bodies
            )
        return 1

    @property
    def body_names(self) -> list[str]:
        if (
            self.world is not None
            and self.key in getattr(self.world, "static_rigids", {})
            and self.source_data is not None
        ):
            return [spec.name for spec in rigid_shape_specs(self.source_data)]
        if self.world is None:
            return [f"body_{i}" for i in range(self.num_bodies)]
        cache = self.world.state.record(self.env_id, self.obj_id).cache
        return list(
            getattr(cache, "body_names", [f"body_{i}" for i in range(self.num_bodies)])
        )

    def get_body_id(self, name: str) -> int:
        return _find_name(self.body_names, name, "body")

    def add_contact_sensor(self, body_ids=None, *, name: str = ""):
        return self._require_world().add_contact_sensor(
            self, body_ids=body_ids, name=name
        )

    def add_force_sensor(self, body_ids=None, *, name: str = ""):
        return self._require_world().add_force_sensor(
            self, body_ids=body_ids, name=name
        )

    def set_root_state(
        self,
        env_ids: EnvIdLike | None,
        pos: torch.Tensor,
        rot_xyzw: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        immediate: bool = False,
    ) -> Self:
        """Shapes: position/velocity ``(..., 3)``, rotation ``(..., 4)``."""
        self._require_world().set_root_state(
            self._selected_env_ids(env_ids),
            self.obj_id,
            pos,
            rot_xyzw,
            linear_velocity,
            angular_velocity,
            immediate=immediate,
        )
        return self

    def set_body_force(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
    ) -> Self:
        """Shape: ``(..., 3)``."""
        self._require_world().set_body_force(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
        )
        return self

    def set_body_force_at_position(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
        position: torch.Tensor,
    ) -> Self:
        """Shape: force and position ``(..., 3)``."""
        self._require_world().set_body_force_at_position(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
            position,
        )
        return self

    def set_collision_material(self, material):
        """Replace all collision shape materials on this rigid at runtime."""
        world = self._require_world()
        return world.physics.set_rigid_collision_material(self.rigid, material)

    def set_collision_material_overrides(self, material_overrides, data=None):
        """Apply named/indexed collision material overrides at runtime."""
        world = self._require_world()
        source_data = self.source_data if data is None else data
        if source_data is None:
            raise RuntimeError(
                "set_collision_material_overrides requires the ArticulationDesc used to create this rigid"
            )
        return world.physics.set_rigid_collision_material_overrides(
            self.rigid, source_data, material_overrides
        )

    def _selected_env_ids(self, env_ids: EnvIdLike | None):
        if env_ids is None:
            return self.env_id
        selected = tuple(env_id_list(env_ids, self.env_id + 1))
        if selected != (self.env_id,):
            raise KeyError(
                f"SimRigid only contains env_id={self.env_id}, got {selected}"
            )
        return self.env_id

    def _require_world(self):
        if self.world is None:
            raise RuntimeError("SimRigid is not attached to a KangSimWorld")
        return self.world


@dataclass(slots=True)
class SimArticulationBatch:
    """Batched simulation-facing view for one logical articulation object."""

    world: KangSimWorld
    obj_id: int
    env_ids: tuple[int, ...]
    name: str = ""

    @property
    def key(self):
        return (self.env_ids, self.obj_id)

    @property
    def data(self) -> SimObjectState:
        return self.get_data()

    def get_data(self, env_ids: EnvIdLike | None = None) -> SimObjectState:
        return self.world.state.object_states(
            self.obj_id, self._selected_env_ids(env_ids)
        )

    def get_root_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 3)``."""
        return self.get_data(env_ids).root_pos

    def get_root_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 4)``."""
        return self.get_data(env_ids).root_rot

    def get_root_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 3)``."""
        return self.get_data(env_ids).root_vel

    def get_root_ang_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 3)``."""
        return self.get_data(env_ids).root_ang_vel

    def get_body_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, B, 3)``."""
        return self.get_data(env_ids).body_pos

    def get_body_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, B, 4)``."""
        return self.get_data(env_ids).body_rot

    def get_dof_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, D)``."""
        return self.get_data(env_ids).dof_pos

    def get_dof_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, D)``."""
        return self.get_data(env_ids).dof_vel

    def get_dof_force(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, D)``."""
        return self.get_data(env_ids).dof_force

    @property
    def records(self) -> tuple[SimArticulation, ...]:
        return tuple(
            self.world.articulations[(eid, self.obj_id)] for eid in self.env_ids
        )

    @property
    def first(self) -> SimArticulation:
        return self.records[0]

    @property
    def articulation(self):
        return self.first.articulation

    @property
    def num_envs(self) -> int:
        return len(self.env_ids)

    @property
    def num_dofs(self) -> int:
        return self.first.num_dofs

    @property
    def num_bodies(self) -> int:
        return self.first.num_bodies

    @property
    def joint_names(self) -> list[str]:
        return self.first.joint_names

    @property
    def body_names(self) -> list[str]:
        return self.first.body_names

    def get_joint_id(self, name: str) -> int:
        return self.first.get_joint_id(name)

    def get_body_id(self, name: str) -> int:
        return self.first.get_body_id(name)

    def add_contact_sensor(self, body_ids=None, *, name: str = ""):
        return self.world.add_contact_sensor(self, body_ids=body_ids, name=name)

    def add_force_sensor(self, body_ids=None, *, name: str = ""):
        return self.world.add_force_sensor(self, body_ids=body_ids, name=name)

    def set_cmd(
        self,
        env_ids: EnvIdLike | None,
        cmd: torch.Tensor,
        mode: "ControlMode | str" = "pos",
        kp: float | torch.Tensor | None = 200.0,
        kd: float | torch.Tensor | None = 10.0,
    ) -> Self:
        """Shape: command and tensor gains ``(N, D)``."""
        self.world.set_cmd(
            self._selected_env_ids(env_ids),
            self.obj_id,
            cmd,
            mode,
            kp,
            kd,
        )
        return self

    def clear_cmd(self, env_ids: EnvIdLike | None = None):
        self.world.clear_cmd(self._selected_env_ids(env_ids), self.obj_id)
        return self

    def set_root_state(
        self,
        env_ids: EnvIdLike | None,
        pos: torch.Tensor,
        rot_xyzw: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        immediate: bool = False,
    ) -> Self:
        """Shapes: position/velocity ``(N, 3)``, rotation ``(N, 4)``."""
        self.world.set_root_state(
            self._selected_env_ids(env_ids),
            self.obj_id,
            pos,
            rot_xyzw,
            linear_velocity,
            angular_velocity,
            immediate=immediate,
        )
        return self

    def set_dof_state(
        self,
        env_ids: EnvIdLike | None,
        positions: torch.Tensor,
        velocities: torch.Tensor | None = None,
        immediate: bool = False,
    ) -> Self:
        """Shape: ``(N, D)``."""
        self.world.set_dof_state(
            self._selected_env_ids(env_ids),
            self.obj_id,
            positions,
            velocities,
            immediate=immediate,
        )
        return self

    def set_body_force(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
    ) -> Self:
        """Shape: ``(N, 3)``."""
        self.world.set_body_force(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
        )
        return self

    def set_body_force_at_position(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
        position: torch.Tensor,
    ) -> Self:
        """Shape: force and position ``(N, 3)``."""
        self.world.set_body_force_at_position(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
            position,
        )
        return self

    def _selected_env_ids(self, env_ids: EnvIdLike | None):
        if env_ids is None:
            return self.env_ids
        selected = tuple(env_id_list(env_ids, self.world.num_envs))
        missing = [eid for eid in selected if eid not in self.env_ids]
        if missing:
            raise KeyError(
                f"env ids {missing} are outside this view's env_ids={self.env_ids}"
            )
        return selected


@dataclass(slots=True)
class SimRigidBatch:
    """Batched simulation-facing view for one logical rigid object."""

    world: KangSimWorld
    obj_id: int
    env_ids: tuple[int, ...]
    name: str = ""

    @property
    def key(self):
        return (self.env_ids, self.obj_id)

    @property
    def data(self) -> SimObjectState:
        return self.get_data()

    def get_data(self, env_ids: EnvIdLike | None = None) -> SimObjectState:
        return self.world.state.object_states(
            self.obj_id, self._selected_env_ids(env_ids)
        )

    def get_root_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 3)``."""
        return self.get_data(env_ids).root_pos

    def get_root_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 4)``."""
        return self.get_data(env_ids).root_rot

    def get_root_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 3)``."""
        return self.get_data(env_ids).root_vel

    def get_root_ang_vel(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, 3)``."""
        return self.get_data(env_ids).root_ang_vel

    def get_body_pos(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, B, 3)``."""
        return self.get_data(env_ids).body_pos

    def get_body_rot(self, env_ids: EnvIdLike | None = None) -> torch.Tensor:
        """Shape: ``(N, B, 4)``."""
        return self.get_data(env_ids).body_rot

    @property
    def records(self) -> tuple[SimRigid, ...]:
        records = self.world.rigids | self.world.static_rigids
        return tuple(records[(eid, self.obj_id)] for eid in self.env_ids)

    @property
    def first(self) -> SimRigid:
        return self.records[0]

    @property
    def rigid(self):
        return self.first.rigid

    @property
    def num_envs(self) -> int:
        return len(self.env_ids)

    @property
    def num_bodies(self) -> int:
        return self.first.num_bodies

    @property
    def body_names(self) -> list[str]:
        return self.first.body_names

    def get_body_id(self, name: str) -> int:
        return self.first.get_body_id(name)

    def add_contact_sensor(self, body_ids=None, *, name: str = ""):
        return self.world.add_contact_sensor(self, body_ids=body_ids, name=name)

    def add_force_sensor(self, body_ids=None, *, name: str = ""):
        return self.world.add_force_sensor(self, body_ids=body_ids, name=name)

    def set_root_state(
        self,
        env_ids: EnvIdLike | None,
        pos: torch.Tensor,
        rot_xyzw: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        immediate: bool = False,
    ) -> Self:
        """Shapes: position/velocity ``(N, 3)``, rotation ``(N, 4)``."""
        self.world.set_root_state(
            self._selected_env_ids(env_ids),
            self.obj_id,
            pos,
            rot_xyzw,
            linear_velocity,
            angular_velocity,
            immediate=immediate,
        )
        return self

    def set_body_force(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
    ) -> Self:
        """Shape: ``(N, 3)``."""
        self.world.set_body_force(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
        )
        return self

    def set_body_force_at_position(
        self,
        env_ids: EnvIdLike | None,
        body_id: int,
        force: torch.Tensor,
        position: torch.Tensor,
    ) -> Self:
        """Shape: force and position ``(N, 3)``."""
        self.world.set_body_force_at_position(
            self._selected_env_ids(env_ids),
            self.obj_id,
            body_id,
            force,
            position,
        )
        return self

    def set_collision_material(self, env_ids: EnvIdLike | None, material):
        """Replace collision shape materials for selected rigid env instances."""
        selected = self._selected_env_ids(env_ids)
        updated = 0
        for env_id in selected:
            updated += self.world.rigids[(env_id, self.obj_id)].set_collision_material(
                material
            )
        return updated

    def set_collision_material_overrides(
        self, env_ids: EnvIdLike | None, material_overrides, data=None
    ):
        """Apply named/indexed material overrides to selected rigid env instances."""
        selected = self._selected_env_ids(env_ids)
        updated = 0
        for env_id in selected:
            updated += self.world.rigids[
                (env_id, self.obj_id)
            ].set_collision_material_overrides(material_overrides, data=data)
        return updated

    def _selected_env_ids(self, env_ids: EnvIdLike | None):
        if env_ids is None:
            return self.env_ids
        selected = tuple(env_id_list(env_ids, self.world.num_envs))
        missing = [eid for eid in selected if eid not in self.env_ids]
        if missing:
            raise KeyError(
                f"env ids {missing} are outside this view's env_ids={self.env_ids}"
            )
        return selected


def _find_name(names: list[str], name: str, kind: str) -> int:
    try:
        return names.index(str(name))
    except ValueError as exc:
        raise KeyError(f"{kind} {name!r} not found") from exc


class ControlMode(str, Enum):
    NONE = "none"
    POS = "pos"
    VEL = "vel"
    TORQUE = "torque"
    PD_EXPLICIT = "pd_explicit"


@dataclass(slots=True)
class CommandBuffer:
    mode: ControlMode
    cmd: object | None = None
    kp: object | None = None
    kd: object | None = None


@dataclass(slots=True)
class BatchCommandBuffer:
    mode: ControlMode
    env_ids: tuple[int, ...]
    obj_id: int
    cmd: object
    kp: object | None = None
    kd: object | None = None


@dataclass(slots=True)
class RootStateReset:
    pos: np.ndarray
    rot_xyzw: np.ndarray
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None


@dataclass(slots=True)
class DofStateReset:
    positions: np.ndarray
    velocities: np.ndarray | None = None


@dataclass(slots=True)
class ResetBuffer:
    root: RootStateReset | None = None
    dof: DofStateReset | None = None

    @property
    def pending(self) -> bool:
        return self.root is not None or self.dof is not None


@dataclass(slots=True)
class GpuRootStateResetBatch:
    """One object's CUDA root states for a set of environments."""

    env_ids: object
    obj_id: int
    state: object


@dataclass(slots=True)
class GpuDofStateResetBatch:
    """One articulation's CUDA DOF states for a set of environments."""

    env_ids: object
    obj_id: int
    state: object


class KangSimWorld:
    """Owns a PhysX world, registered objects, commands, and world state.

    In CPU simulation, ``state`` is the canonical Python runtime state. In GPU
    simulation, PhysX ``GpuArrayView`` buffers are canonical and ``state`` is the
    latest explicit CPU/Torch snapshot refreshed by ``refresh()`` or
    ``step(refresh=True)``.

    Args:
        sim_device: Simulation backend/device. ``"cpu"`` is the default PhysX
            CPU path. ``"cuda"`` enables the experimental PhysX GPU path.
        state_device: Torch device for ``world.state`` snapshot tensors. This
            is intentionally separate from ``sim_device``.
        device: Compatibility alias for ``state_device``.
    """

    def __init__(
        self,
        num_envs: int = 1,
        physics_config=None,
        sim_dt: float | None = None,
        add_ground: bool = False,
        sim_device="cpu",
        device=None,
        state_device=None,
    ):
        _require_physx()
        if physics_config is None:
            physics_config = _ke.physics.PhysicsConfig.z_up()
        if sim_dt is not None:
            physics_config.dt = float(sim_dt)
        sim_device = _resolve_sim_device(sim_device)
        uses_gpu_sim = sim_device.type == "cuda"
        if hasattr(physics_config, "enable_gpu"):
            physics_config.enable_gpu = uses_gpu_sim

        self.num_envs = int(num_envs)
        self._all_env_ids = tuple(range(self.num_envs))
        self.has_ground = bool(add_ground)
        self.physics = _ke.physics.PhysicsWorld(physics_config)
        if add_ground:
            self.physics.add_default_ground()

        self.sim_device = sim_device
        self.state_device = _resolve_state_device(
            state_device=state_device, device=device
        )
        self.device = self.state_device
        self.state = KangWorldState(
            num_envs=self.num_envs,
            device=self.state_device,
            canonical_source="gpu" if uses_gpu_sim else "cpu",
            snapshot=uses_gpu_sim,
            gpu_system_provider=lambda: self.gpu_system,
        )
        self.articulations: dict[tuple[int, int], SimArticulation] = {}
        self.rigids: dict[tuple[int, int], SimRigid] = {}
        self.static_rigids: dict[tuple[int, int], SimRigid] = {}
        self.commands: dict[tuple[int, int], CommandBuffer] = {}
        self.batch_commands: dict[int, BatchCommandBuffer] = {}
        self.resets: dict[tuple[int, int], ResetBuffer] = {}
        self.body_forces: dict[tuple[int, int], np.ndarray] = {}
        self.body_force_positions: dict[tuple[int, int], np.ndarray] = {}
        self._pending_reset_keys: set[tuple[int, int]] = set()
        self._gpu_root_reset_batches: list[GpuRootStateResetBatch] = []
        self._gpu_dof_reset_batches: list[GpuDofStateResetBatch] = []
        self._active_body_force_keys: set[tuple[int, int]] = set()
        self._mjcf_cache: dict[tuple[str, float, str], object] = {}
        self._mjcf_load_count = 0
        self.sim_time = 0.0
        self.sim_dt = float(physics_config.dt)
        self._advance_time_remainder = 0.0
        self._uses_gpu_sim = uses_gpu_sim
        self._gpu_system = None
        self._rigid_gpu_rows: dict[tuple[int, int], int] = {}
        self._rigid_gpu_index_tensors: dict[tuple[tuple[int, int], ...], object] = {}
        self._articulation_gpu_rows: dict[tuple[int, int], int] = {}
        self._articulation_gpu_index_tensors: dict[
            tuple[tuple[int, int], ...], object
        ] = {}
        self._articulation_gpu_row_tensors: dict[int, object] = {}
        self._rigid_gpu_row_tensors: dict[int, object] = {}
        self._articulation_gpu_effort_limit_tensors: dict[
            tuple[int, object], object
        ] = {}
        self._validated_cuda_batch_command_obj_ids: set[int] = set()
        self._dense_cuda_batch_command_obj_ids: dict[int, bool] = {}
        self.sensors: dict[str, object] = {}
        self._contact_sensor_batch = None
        self._released = False

    def add_contact_sensor(self, target, body_ids=None, *, name: str = ""):
        """Attach a GPU contact sensor to a simulation object or object view."""
        from .sensor import ContactSensor

        return self._add_contact_sensor(
            ContactSensor, target, body_ids, name=name, suffix="contact"
        )

    def add_force_sensor(self, target, body_ids=None, *, name: str = ""):
        """Attach a normal-force sensor to a simulation object or view."""
        from .sensor import ForceSensor

        return self._add_contact_sensor(
            ForceSensor, target, body_ids, name=name, suffix="force"
        )

    def _add_contact_sensor(
        self, sensor_type, target, body_ids, *, name: str, suffix: str
    ):
        from .sensor import _ContactSensorBatch

        if not name:
            target_name = target.name or f"object_{target.obj_id}"
            name = f"{target_name}_{suffix}"
        name = str(name)
        if name in self.sensors:
            raise ValueError(f"sensor name already registered: {name!r}")
        sensor = sensor_type(self, target, body_ids, name=name)
        self.sensors[name] = sensor
        if self._contact_sensor_batch is None:
            self._contact_sensor_batch = _ContactSensorBatch(self)
        self._contact_sensor_batch.mark_dirty()
        return sensor

    def refresh_sensors(self):
        """Refresh all sensor modules, sharing raw producer fetches per frame."""
        if not self.sensors:
            return {}
        contact_sensors = tuple(
            sensor for sensor in self.sensors.values() if sensor.requires_contact_data
        )
        if contact_sensors:
            self.gpu_system.fetch_contact_pairs()
            self._contact_sensor_batch.refresh(contact_sensors)
        for sensor in self.sensors.values():
            if not sensor.requires_contact_data:
                sensor.refresh()
        return {name: sensor.data for name, sensor in self.sensors.items()}

    def clear_sensor_outputs(self):
        if not self.sensors:
            return {}
        contact_sensors = tuple(
            sensor for sensor in self.sensors.values() if sensor.requires_contact_data
        )
        if contact_sensors:
            if self._contact_sensor_batch is None:
                from .sensor import _ContactSensorBatch

                self._contact_sensor_batch = _ContactSensorBatch(self)
            self._contact_sensor_batch.clear_outputs(contact_sensors)
        return {name: sensor.data for name, sensor in self.sensors.items()}

    def add_articulation(
        self,
        data,
        *,
        env_id: int = 0,
        obj_id: int = 0,
        name: str = "",
        config=None,
    ) -> SimArticulation:
        if config is None:
            config = _ke.physics.ArticulationConfig.free_base()
        restore_collision_group = None
        if hasattr(config, "collision_group") and int(config.collision_group) == 0:
            restore_collision_group = int(config.collision_group)
            config.collision_group = int(env_id) + 1
        self._require_gpu_runtime_uninitialized("add_articulation")
        key = (int(env_id), int(obj_id))
        if key in self.articulations or key in self.rigids or key in self.static_rigids:
            raise ValueError(f"object already registered at env={key[0]}, obj={key[1]}")

        try:
            if isinstance(data, _ke.physics.ArticulationTemplate):
                articulation = _ke.physics.Articulation.build_from_template(
                    self.physics, data, config
                )
            else:
                articulation = _ke.physics.Articulation.build(
                    self.physics, data, config
                )
        finally:
            if restore_collision_group is not None:
                config.collision_group = restore_collision_group
        record = SimArticulation(key[0], key[1], str(name), articulation, self)
        self.articulations[key] = record
        self.state.add_articulation(
            articulation, key[0], key[1], name, physics=self.physics
        )
        self.commands[key] = CommandBuffer(ControlMode.NONE)
        self.resets[key] = ResetBuffer()
        self.body_forces[key] = np.zeros(
            (articulation.num_links(), 3), dtype=np.float32
        )
        self.body_force_positions[key] = np.full(
            (articulation.num_links(), 3), np.nan, dtype=np.float32
        )
        return record

    def create_articulation_template(self, data, *, config=None):
        """Precompute immutable articulation resources for repeated instances."""
        if config is None:
            config = _ke.physics.ArticulationConfig.free_base()
        self._require_gpu_runtime_uninitialized("create_articulation_template")
        return _ke.physics.ArticulationTemplate.create(data, config)

    def add_articulation_batch(
        self,
        template,
        *,
        obj_id: int = 0,
        name: str = "",
        config=None,
        env_ids=None,
    ) -> SimArticulationBatch:
        """Create per-environment PhysX instances from one shared template."""
        selected = tuple(env_id_list(env_ids, self.num_envs))
        for env_id in selected:
            self.add_articulation(
                template,
                env_id=env_id,
                obj_id=obj_id,
                name=name,
                config=config,
            )
        return SimArticulationBatch(self, int(obj_id), selected, str(name))

    def add_rigid(
        self,
        data,
        *,
        env_id: int = 0,
        obj_id: int = 0,
        name: str = "",
        pos=None,
        rot_xyzw=None,
        density: float = 1.0,
        collision_group: int | None = None,
        contact_offset: float = 0.02,
        rest_offset: float = 0.0,
        kinematic: bool = False,
    ) -> SimRigid:
        self._require_gpu_runtime_uninitialized("add_rigid")
        key = (int(env_id), int(obj_id))
        if key in self.articulations or key in self.rigids or key in self.static_rigids:
            raise ValueError(f"object already registered at env={key[0]}, obj={key[1]}")
        if pos is None:
            pos = np.zeros(3, dtype=np.float32)
        if rot_xyzw is None:
            rot_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if collision_group is None:
            collision_group = key[0] + 1
        rigid = self.physics.create_dynamic_rigid(
            data,
            _as_cpu_numpy(pos).reshape(3),
            _as_cpu_numpy(rot_xyzw).reshape(4),
            float(density),
            int(collision_group),
            float(contact_offset),
            float(rest_offset),
        )
        if kinematic:
            rigid.set_kinematic(True)
        shape_specs = rigid_shape_specs(data)
        body_names = [spec.name for spec in shape_specs]
        local_pos = np.stack([spec.local_pos for spec in shape_specs], axis=0)
        local_rot = np.stack([spec.local_rot for spec in shape_specs], axis=0)
        record = SimRigid(key[0], key[1], str(name), rigid, self, data)
        self.rigids[key] = record
        self.state.add_rigid(
            rigid,
            key[0],
            key[1],
            name,
            physics=self.physics,
            body_names=body_names,
            local_pos=local_pos,
            local_rot=local_rot,
        )
        self.commands[key] = CommandBuffer(ControlMode.NONE)
        self.resets[key] = ResetBuffer()
        self.body_forces[key] = np.zeros((len(body_names), 3), dtype=np.float32)
        self.body_force_positions[key] = np.full(
            (len(body_names), 3), np.nan, dtype=np.float32
        )
        return record

    def add_static_rigid(
        self,
        data,
        *,
        env_id: int = 0,
        obj_id: int = 0,
        name: str = "",
        pos=None,
        rot_xyzw=None,
        collision_group: int | None = None,
        contact_offset: float = 0.02,
        rest_offset: float = 0.0,
    ) -> SimRigid:
        self._require_gpu_runtime_uninitialized("add_static_rigid")
        key = (int(env_id), int(obj_id))
        if key in self.articulations or key in self.rigids or key in self.static_rigids:
            raise ValueError(f"object already registered at env={key[0]}, obj={key[1]}")
        if pos is None:
            pos = np.zeros(3, dtype=np.float32)
        if rot_xyzw is None:
            rot_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if collision_group is None:
            collision_group = key[0] + 1
        rigid = self.physics.create_static_rigid(
            data,
            _as_cpu_numpy(pos).reshape(3).tolist(),
            _as_cpu_numpy(rot_xyzw).reshape(4).tolist(),
            int(collision_group),
            float(contact_offset),
            float(rest_offset),
        )
        record = SimRigid(key[0], key[1], str(name), rigid, self, data)
        self.static_rigids[key] = record
        return record

    def add_mjcf_articulation(
        self,
        mjcf_path: str,
        *,
        env_id: int = 0,
        obj_id: int = 0,
        name: str = "",
        config=None,
        order: str = "DFS",
        scale: float = 1.0,
    ) -> SimArticulation:
        data = self.load_mjcf(mjcf_path, scale=scale, order=order)
        return self.add_articulation(
            data,
            env_id=env_id,
            obj_id=obj_id,
            name=name,
            config=config,
        )

    def get_articulation(self, env_id: int = 0, obj_id: int = 0) -> SimArticulation:
        key = (int(env_id), int(obj_id))
        try:
            return self.articulations[key]
        except KeyError as exc:
            raise KeyError(
                f"no articulation registered at env={key[0]}, obj={key[1]}"
            ) from exc

    def get_rigid(self, env_id: int = 0, obj_id: int = 0) -> SimRigid:
        key = (int(env_id), int(obj_id))
        try:
            return (self.rigids | self.static_rigids)[key]
        except KeyError as exc:
            raise KeyError(
                f"no rigid registered at env={key[0]}, obj={key[1]}"
            ) from exc

    def get_object(
        self, env_id: int = 0, obj_id: int = 0
    ) -> SimArticulation | SimRigid:
        key = (int(env_id), int(obj_id))
        if key in self.articulations:
            return self.articulations[key]
        if key in self.rigids:
            return self.rigids[key]
        if key in self.static_rigids:
            return self.static_rigids[key]
        raise KeyError(f"no object registered at env={key[0]}, obj={key[1]}")

    def get_articulation_batch(
        self,
        env_ids: EnvIdLike = None,
        obj_id: int = 0,
        name: str | None = None,
    ) -> SimArticulationBatch:
        obj_id = int(obj_id)
        selected_env_ids = self._object_env_ids(self.articulations, obj_id, env_ids)
        if name is None:
            name = self.articulations[(selected_env_ids[0], obj_id)].name
        return SimArticulationBatch(self, obj_id, selected_env_ids, str(name))

    def get_rigid_batch(
        self,
        env_ids: EnvIdLike = None,
        obj_id: int = 0,
        name: str | None = None,
    ) -> SimRigidBatch:
        obj_id = int(obj_id)
        records = self.rigids | self.static_rigids
        selected_env_ids = self._object_env_ids(records, obj_id, env_ids)
        if name is None:
            name = records[(selected_env_ids[0], obj_id)].name
        return SimRigidBatch(self, obj_id, selected_env_ids, str(name))

    def get_object_batch(
        self,
        env_ids: EnvIdLike = None,
        obj_id: int = 0,
        name: str | None = None,
    ) -> SimArticulationBatch | SimRigidBatch:
        obj_id = int(obj_id)
        if any(key[1] == obj_id for key in self.articulations):
            return self.get_articulation_batch(env_ids, obj_id, name)
        if any(key[1] == obj_id for key in self.rigids) or any(
            key[1] == obj_id for key in self.static_rigids
        ):
            return self.get_rigid_batch(env_ids, obj_id, name)
        raise KeyError(f"no object registered with obj_id={obj_id}")

    def _object_env_ids(self, records, obj_id: int, env_ids: EnvIdLike):
        registered = sorted(env_id for env_id, oid in records if oid == int(obj_id))
        if not registered:
            raise KeyError(f"no object registered with obj_id={obj_id}")
        if env_ids is None:
            return tuple(registered)
        selected = tuple(env_id_list(env_ids, self.num_envs))
        missing = [eid for eid in selected if (eid, int(obj_id)) not in records]
        if missing:
            raise KeyError(
                f"object obj={obj_id} is missing env registrations: {missing}"
            )
        return selected

    def load_mjcf(self, mjcf_path: str, scale: float = 1.0, order: str = "DFS"):
        key = (str(Path(mjcf_path).expanduser().resolve()), float(scale), str(order))
        data = self._mjcf_cache.get(key)
        if data is None:
            data = _ke.asset.MJCFLoader.load(key[0], scale=key[1], order=key[2])
            self._mjcf_cache[key] = data
            self._mjcf_load_count += 1
        return data

    def get_mjcf_cache_size(self) -> int:
        return len(self._mjcf_cache)

    def get_mjcf_load_count(self) -> int:
        return self._mjcf_load_count

    def set_cmd(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        cmd,
        mode: ControlMode | str = ControlMode.POS,
        kp: object | None = 200.0,
        kd: object | None = 10.0,
    ):
        import sys

        mode = ControlMode(mode)
        # Batched views keep this canonical tuple. Preserve it instead of
        # round-tripping thousands of ids through NumPy on every substep.
        env_ids = (
            env_id
            if isinstance(env_id, tuple) and env_id == self._all_env_ids
            else env_id_list(env_id, self.num_envs)
        )
        torch = sys.modules.get("torch")
        cuda_cmd = (
            torch is not None and torch.is_tensor(cmd) and cmd.device.type == "cuda"
        )
        if cuda_cmd:
            if self._gpu_system is None:
                raise RuntimeError(
                    "CUDA command tensors require init_gpu_system(); KangSimWorld will not silently copy them to CPU"
                )
            if mode not in (
                ControlMode.POS,
                ControlMode.VEL,
                ControlMode.TORQUE,
                ControlMode.PD_EXPLICIT,
            ):
                raise ValueError(
                    f"CUDA command tensors do not support mode={mode.value!r}"
                )
            if cmd.dtype != torch.float32:
                raise TypeError("CUDA command tensor must have dtype=torch.float32")
            if not cmd.is_contiguous():
                raise ValueError("CUDA command tensor must be contiguous")
            batch = self._try_set_cuda_batch_command(
                env_ids,
                int(obj_id),
                cmd,
                mode,
                kp,
                kd,
                torch=torch,
            )
            if batch:
                return

        for env_index, eid in enumerate(env_ids):
            key = (eid, int(obj_id))
            if key in self.rigids:
                raise TypeError(
                    f"rigid body at env={eid}, obj={obj_id} does not accept commands"
                )
            record = self.articulations[key]
            expected = record.articulation.num_dofs()
            if cuda_cmd:
                gpu_device = int(self.gpu_system.articulation_joint_forces().device_id)
                cmd_device = cmd.device.index if cmd.device.index is not None else 0
                if cmd_device != gpu_device:
                    raise ValueError(
                        f"CUDA command tensor is on cuda:{cmd_device}, expected cuda:{gpu_device}"
                    )
                if cmd.ndim == 1 and tuple(cmd.shape) == (expected,):
                    arr = cmd
                elif cmd.ndim == 2 and tuple(cmd.shape) == (
                    len(env_ids),
                    expected,
                ):
                    arr = cmd[env_index]
                else:
                    raise ValueError(
                        "CUDA command tensor expected shape "
                        f"[{expected}] or [{len(env_ids)}, {expected}], got "
                        f"{list(cmd.shape)}"
                    )
            else:
                arr = select_env_value(cmd, env_index, len(env_ids), (expected,))
            value_count = int(arr.numel()) if cuda_cmd else int(arr.size)
            if (
                mode in (ControlMode.POS, ControlMode.VEL, ControlMode.PD_EXPLICIT)
                and value_count != expected
            ):
                raise ValueError(
                    f"{mode.value} command for env={eid}, obj={obj_id} expected {expected} values, got {value_count}"
                )
            if mode == ControlMode.PD_EXPLICIT:
                if cuda_cmd:
                    kp_arr = self._select_cuda_pd_gain(
                        kp,
                        env_index,
                        len(env_ids),
                        expected,
                        torch=torch,
                        device=cmd.device,
                        name="kp",
                    )
                    kd_arr = self._select_cuda_pd_gain(
                        kd,
                        env_index,
                        len(env_ids),
                        expected,
                        torch=torch,
                        device=cmd.device,
                        name="kd",
                    )
                else:
                    kp_arr = self._select_pd_gain_array(
                        kp, env_index, len(env_ids), expected, name="kp"
                    )
                    kd_arr = self._select_pd_gain_array(
                        kd, env_index, len(env_ids), expected, name="kd"
                    )
            else:
                kp_arr = _optional_drive_array(kp, expected)
                kd_arr = _optional_drive_array(kd, expected)
            stored = arr if cuda_cmd else arr.copy()
            self.commands[key] = CommandBuffer(mode, stored, kp_arr, kd_arr)

    def _try_set_cuda_batch_command(
        self,
        env_ids: list[int],
        obj_id: int,
        cmd,
        mode: ControlMode,
        kp,
        kd,
        *,
        torch,
    ) -> bool:
        if self._gpu_system is None or not torch.is_tensor(cmd):
            return False
        if cmd.device.type != "cuda" or cmd.ndim != 2:
            return False
        if tuple(env_ids) != self._all_env_ids:
            return False
        key0 = (0, int(obj_id))
        if key0 in self.rigids:
            raise TypeError(f"rigid body obj={obj_id} does not accept commands")
        if key0 not in self.articulations:
            return False
        expected = self.articulations[key0].articulation.num_dofs()
        if tuple(cmd.shape) != (self.num_envs, expected):
            return False

        gpu_device = int(self.gpu_system.articulation_joint_forces().device_id)
        cmd_device = cmd.device.index if cmd.device.index is not None else 0
        if cmd_device != gpu_device:
            raise ValueError(
                f"CUDA command tensor is on cuda:{cmd_device}, expected cuda:{gpu_device}"
            )
        if int(obj_id) not in self._validated_cuda_batch_command_obj_ids:
            for eid in env_ids:
                key = (int(eid), int(obj_id))
                if key in self.rigids:
                    raise TypeError(
                        f"rigid body at env={eid}, obj={obj_id} does not accept commands"
                    )
                if key not in self.articulations:
                    raise KeyError(
                        f"articulation env={eid}, obj={obj_id} is not registered"
                    )
                dofs = self.articulations[key].articulation.num_dofs()
                if dofs != expected:
                    raise RuntimeError(
                        f"articulation obj={obj_id} has inconsistent DOF counts: "
                        f"env 0 has {expected}, env {eid} has {dofs}"
                    )
            self._validated_cuda_batch_command_obj_ids.add(int(obj_id))

        if mode == ControlMode.PD_EXPLICIT:
            kp_arr = self._select_cuda_batch_pd_gain(
                kp, self.num_envs, expected, torch=torch, device=cmd.device, name="kp"
            )
            kd_arr = self._select_cuda_batch_pd_gain(
                kd, self.num_envs, expected, torch=torch, device=cmd.device, name="kd"
            )
        else:
            kp_arr = _optional_drive_array(kp, expected)
            kd_arr = _optional_drive_array(kd, expected)

        had_full_batch = int(obj_id) in self.batch_commands
        self.batch_commands[int(obj_id)] = BatchCommandBuffer(
            mode=mode,
            env_ids=(
                env_ids
                if isinstance(env_ids, tuple)
                else tuple(int(eid) for eid in env_ids)
            ),
            obj_id=int(obj_id),
            cmd=cmd,
            kp=kp_arr,
            kd=kd_arr,
        )
        if not had_full_batch:
            for eid in env_ids:
                self.commands[(int(eid), int(obj_id))] = CommandBuffer(ControlMode.NONE)
        return True

    def _select_cuda_batch_pd_gain(
        self,
        value,
        env_count: int,
        expected: int,
        *,
        torch,
        device,
        name: str,
    ):
        if value is None:
            raise ValueError(f"GPU PD_EXPLICIT requires per-DOF {name} array")
        if torch.is_tensor(value):
            if value.device.type != "cuda":
                raise RuntimeError(
                    f"GPU PD_EXPLICIT {name} tensor must be CUDA; mixed CPU/CUDA gains are not allowed"
                )
            value_device = value.device.index if value.device.index is not None else 0
            cmd_device = device.index if device.index is not None else 0
            if value_device != cmd_device:
                raise ValueError(
                    f"GPU PD_EXPLICIT {name} tensor is on cuda:{value_device}, expected cuda:{cmd_device}"
                )
            if value.dtype != torch.float32:
                raise TypeError(
                    f"GPU PD_EXPLICIT {name} tensor must have dtype=torch.float32"
                )
            if not value.is_contiguous():
                raise ValueError(f"GPU PD_EXPLICIT {name} tensor must be contiguous")
            if value.ndim == 1 and tuple(value.shape) == (expected,):
                return value
            if value.ndim == 2 and tuple(value.shape) == (env_count, expected):
                return value
            raise ValueError(
                f"GPU PD_EXPLICIT {name} tensor expected shape "
                f"[{expected}] or [{env_count}, {expected}], got "
                f"{list(value.shape)}"
            )
        array = _as_cpu_numpy(value).astype(np.float32, copy=False)
        if array.ndim == 0 or array.size == 1:
            raise ValueError(f"GPU PD_EXPLICIT {name} must be a per-DOF array")
        if array.shape == (expected,):
            return array.copy()
        if array.shape == (env_count, expected):
            return array.copy()
        raise ValueError(
            f"GPU PD_EXPLICIT {name} array expected shape "
            f"[{expected}] or [{env_count}, {expected}], got {list(array.shape)}"
        )

    def _select_cuda_pd_gain(
        self,
        value,
        env_index: int,
        env_count: int,
        expected: int,
        *,
        torch,
        device,
        name: str,
    ):
        if value is None:
            raise ValueError(f"GPU PD_EXPLICIT requires per-DOF {name} array")
        if torch.is_tensor(value):
            if value.device.type != "cuda":
                raise RuntimeError(
                    f"GPU PD_EXPLICIT {name} tensor must be CUDA; mixed CPU/CUDA gains are not allowed"
                )
            value_device = value.device.index if value.device.index is not None else 0
            cmd_device = device.index if device.index is not None else 0
            if value_device != cmd_device:
                raise ValueError(
                    f"GPU PD_EXPLICIT {name} tensor is on cuda:{value_device}, expected cuda:{cmd_device}"
                )
            if value.dtype != torch.float32:
                raise TypeError(
                    f"GPU PD_EXPLICIT {name} tensor must have dtype=torch.float32"
                )
            if not value.is_contiguous():
                raise ValueError(f"GPU PD_EXPLICIT {name} tensor must be contiguous")
            if value.ndim == 1 and tuple(value.shape) == (expected,):
                return value
            if value.ndim == 2 and tuple(value.shape) == (env_count, expected):
                return value[env_index]
            raise ValueError(
                f"GPU PD_EXPLICIT {name} tensor expected shape "
                f"[{expected}] or [{env_count}, {expected}], got "
                f"{list(value.shape)}"
            )

        array = _as_cpu_numpy(value).astype(np.float32, copy=False)
        if array.ndim == 0 or array.size == 1:
            raise ValueError(f"GPU PD_EXPLICIT {name} must be a per-DOF array")
        if array.shape == (expected,):
            return array.copy()
        if array.shape == (env_count, expected):
            return array[env_index].copy()
        raise ValueError(
            f"GPU PD_EXPLICIT {name} array expected shape "
            f"[{expected}] or [{env_count}, {expected}], got {list(array.shape)}"
        )

    def _select_pd_gain_array(
        self,
        value,
        env_index: int,
        env_count: int,
        expected: int,
        *,
        name: str,
    ):
        if value is None:
            raise ValueError(f"PD_EXPLICIT requires per-DOF {name} array")
        array = _as_cpu_numpy(value).astype(np.float32, copy=False)
        if array.ndim == 0 or array.size == 1:
            raise ValueError(f"PD_EXPLICIT {name} must be a per-DOF array")
        if array.shape == (expected,):
            return array.copy()
        if array.shape == (env_count, expected):
            return array[env_index].copy()
        raise ValueError(
            f"PD_EXPLICIT {name} array expected shape "
            f"[{expected}] or [{env_count}, {expected}], got {list(array.shape)}"
        )

    def clear_cmd(self, env_id: EnvIdLike = None, obj_id: int | None = None):
        keys = list(self.commands.keys())
        env_ids = None if env_id is None else set(env_id_list(env_id, self.num_envs))
        for key in keys:
            if env_ids is not None and key[0] not in env_ids:
                continue
            if obj_id is not None and key[1] != int(obj_id):
                continue
            self.commands[key] = CommandBuffer(ControlMode.NONE)
        if env_id is None:
            if obj_id is None:
                self.batch_commands.clear()
            else:
                self.batch_commands.pop(int(obj_id), None)
        else:
            for oid, batch in list(self.batch_commands.items()):
                if obj_id is not None and oid != int(obj_id):
                    continue
                if any(eid in env_ids for eid in batch.env_ids):
                    self.batch_commands.pop(oid, None)

    def apply_commands(self):
        gpu_per_env_commands = []
        gpu_full_batch_commands = []
        if self._gpu_system is not None:
            gpu_full_batch_commands = [
                batch
                for batch in self.batch_commands.values()
                if batch.mode != ControlMode.NONE and batch.cmd is not None
            ]
        for key, buffer in self.commands.items():
            if buffer.mode == ControlMode.NONE or buffer.cmd is None:
                continue
            if key[1] in self.batch_commands:
                continue
            if (
                self._gpu_system is not None
                and key in self.articulations
                and buffer.mode
                in (
                    ControlMode.POS,
                    ControlMode.VEL,
                    ControlMode.TORQUE,
                    ControlMode.PD_EXPLICIT,
                )
            ):
                gpu_per_env_commands.append((key, buffer))
                continue
            articulation = self.articulations[key].articulation
            if buffer.mode == ControlMode.POS:
                metadata_dirty = False
                if buffer.kp is not None:
                    articulation.set_kps(buffer.kp)
                    metadata_dirty = True
                if buffer.kd is not None:
                    articulation.set_kds(buffer.kd)
                    metadata_dirty = True
                if metadata_dirty:
                    self.state.record(key[0], key[1]).cache.refresh_metadata()
                articulation.set_drive_targets(buffer.cmd)
            elif buffer.mode == ControlMode.VEL:
                metadata_dirty = False
                if buffer.kd is not None:
                    articulation.set_kds(buffer.kd)
                    metadata_dirty = True
                if metadata_dirty:
                    self.state.record(key[0], key[1]).cache.refresh_metadata()
                articulation.set_drive_velocity_targets(buffer.cmd)
            elif buffer.mode == ControlMode.TORQUE:
                torque = _clip_forces(buffer.cmd, articulation.get_effort_limits())
                articulation.set_joint_forces(torque)
            elif buffer.mode == ControlMode.PD_EXPLICIT:
                kp = buffer.kp
                kd = buffer.kd
                if buffer.kp is not None:
                    articulation.set_kps(buffer.kp)
                else:
                    kp = _as_cpu_numpy(articulation.get_kps()).reshape(-1)
                if buffer.kd is not None:
                    articulation.set_kds(buffer.kd)
                else:
                    kd = _as_cpu_numpy(articulation.get_kds()).reshape(-1)
                dof_pos = _as_cpu_numpy(articulation.get_dof_positions()).reshape(-1)
                dof_vel = _as_cpu_numpy(articulation.get_dof_velocities()).reshape(-1)
                torque = kp * (buffer.cmd - dof_pos) - kd * dof_vel
                torque = _clip_forces(torque, articulation.get_effort_limits())
                articulation.set_joint_forces(torque)
            else:
                raise NotImplementedError(
                    f"control mode '{buffer.mode.value}' is not implemented yet"
                )
        self._fetch_gpu_explicit_pd_state(gpu_full_batch_commands, gpu_per_env_commands)
        if gpu_full_batch_commands:
            self._apply_gpu_articulation_full_batch_commands(gpu_full_batch_commands)
        if gpu_per_env_commands:
            self._apply_gpu_articulation_per_env_commands(gpu_per_env_commands)

    def _fetch_gpu_explicit_pd_state(
        self,
        full_batch_commands: list[BatchCommandBuffer],
        per_env_commands: list[tuple[tuple[int, int], CommandBuffer]],
    ):
        """Fetch current GPU joint state once before explicit-PD evaluation."""
        uses_explicit_pd = any(
            command.mode == ControlMode.PD_EXPLICIT for command in full_batch_commands
        ) or any(
            command.mode == ControlMode.PD_EXPLICIT for _, command in per_env_commands
        )
        if not uses_explicit_pd:
            return
        self.gpu_system.fetch_articulation_joint_positions()
        self.gpu_system.fetch_articulation_joint_velocities()

    def _apply_gpu_articulation_full_batch_commands(
        self, commands: list[BatchCommandBuffer]
    ):
        torch = _torch()
        gpu_system = self.gpu_system
        target_qpos_view = gpu_system.articulation_target_joint_positions()
        target_qvel_view = gpu_system.articulation_target_joint_velocities()
        qf_view = gpu_system.articulation_joint_forces()
        qpos_view = gpu_system.articulation_joint_positions()
        qvel_view = gpu_system.articulation_joint_velocities()
        device_id = int(target_qpos_view.device_id)
        device = torch.device(f"cuda:{device_id}")
        target_qpos = target_qpos_view.torch()
        target_qvel = target_qvel_view.torch()
        qf = qf_view.torch()
        qpos = qpos_view.torch()
        qvel = qvel_view.torch()

        for batch in commands:
            rows = self._articulation_gpu_rows_for_obj(
                batch.obj_id, batch.env_ids, device=device
            )
            dof_indices = self._articulation_gpu_dof_indices(
                (batch.env_ids[0], batch.obj_id), device=device
            )
            dof_count = int(
                self.articulations[
                    (batch.env_ids[0], batch.obj_id)
                ].articulation.num_dofs()
            )
            cmd = torch.as_tensor(batch.cmd, dtype=torch.float32, device=device)
            if cmd.ndim == 1:
                cmd = cmd.reshape(1, -1).expand(len(batch.env_ids), -1)
            cmd = cmd[:, :dof_count]
            cols = dof_indices[:dof_count]

            if batch.mode == ControlMode.POS:
                target_qpos[rows[:, None], cols[None, :]] = cmd
                gpu_system.apply_articulation_target_joint_positions(
                    self._articulation_gpu_batch_command_index_view(
                        batch,
                        device=device,
                        name="kangsimworld_articulation_pos_batch_command_indices",
                    )
                )
            elif batch.mode == ControlMode.VEL:
                target_qvel[rows[:, None], cols[None, :]] = cmd
                gpu_system.apply_articulation_target_joint_velocities(
                    self._articulation_gpu_batch_command_index_view(
                        batch,
                        device=device,
                        name="kangsimworld_articulation_vel_batch_command_indices",
                    )
                )
            elif batch.mode == ControlMode.TORQUE:
                limits = self._articulation_gpu_effort_limits(
                    batch.obj_id, device=device
                )[:dof_count]
                qf[rows[:, None], cols[None, :]] = torch.clamp(
                    cmd, -limits[None, :], limits[None, :]
                )
                gpu_system.apply_articulation_joint_forces(
                    self._articulation_gpu_batch_command_index_view(
                        batch,
                        device=device,
                        name="kangsimworld_articulation_torque_batch_command_indices",
                    )
                )
            elif batch.mode == ControlMode.PD_EXPLICIT:
                kp = torch.as_tensor(batch.kp, dtype=torch.float32, device=device)
                kd = torch.as_tensor(batch.kd, dtype=torch.float32, device=device)
                if kp.ndim == 1:
                    kp = kp.reshape(1, -1).expand(len(batch.env_ids), -1)
                if kd.ndim == 1:
                    kd = kd.reshape(1, -1).expand(len(batch.env_ids), -1)
                limits = self._articulation_gpu_effort_limits(
                    batch.obj_id, device=device
                )[:dof_count]
                current_qpos = qpos[rows[:, None], cols[None, :]]
                current_qvel = qvel[rows[:, None], cols[None, :]]
                torque = (
                    kp[:, :dof_count] * (cmd - current_qpos)
                    - kd[:, :dof_count] * current_qvel
                )
                qf[rows[:, None], cols[None, :]] = torch.clamp(
                    torque, -limits[None, :], limits[None, :]
                )
                gpu_system.apply_articulation_joint_forces(
                    self._articulation_gpu_batch_command_index_view(
                        batch,
                        device=device,
                        name="kangsimworld_articulation_pd_batch_command_indices",
                    )
                )
            else:
                raise NotImplementedError(
                    f"control mode '{batch.mode.value}' is not implemented yet"
                )

    def _apply_gpu_articulation_per_env_commands(
        self, commands: list[tuple[tuple[int, int], CommandBuffer]]
    ):
        torch = _torch()
        gpu_system = self.gpu_system
        target_qpos_view = gpu_system.articulation_target_joint_positions()
        target_qvel_view = gpu_system.articulation_target_joint_velocities()
        qf_view = gpu_system.articulation_joint_forces()
        qpos_view = gpu_system.articulation_joint_positions()
        qvel_view = gpu_system.articulation_joint_velocities()
        device_id = int(target_qpos_view.device_id)
        device = torch.device(f"cuda:{device_id}")
        target_qpos = target_qpos_view.torch()
        target_qvel = target_qvel_view.torch()
        qf = qf_view.torch()
        qpos = qpos_view.torch()
        qvel = qvel_view.torch()

        pos_keys = []
        vel_keys = []
        torque_keys = []
        for key, buffer in commands:
            row = self.articulation_gpu_row(key[0], key[1])
            dof_count = int(gpu_system.articulation_dof_count(row))
            dof_indices = self._articulation_gpu_dof_indices(key, device=device)
            cmd = torch.as_tensor(buffer.cmd, dtype=torch.float32, device=device)
            if buffer.mode == ControlMode.POS:
                target_qpos[row, dof_indices] = cmd[:dof_count]
                pos_keys.append(key)
            elif buffer.mode == ControlMode.VEL:
                target_qvel[row, dof_indices] = cmd[:dof_count]
                vel_keys.append(key)
            elif buffer.mode == ControlMode.TORQUE:
                limits = torch.as_tensor(
                    self.articulations[key].articulation.get_effort_limits(),
                    dtype=torch.float32,
                    device=device,
                )
                qf[row, dof_indices] = torch.clamp(
                    cmd[:dof_count], -limits[:dof_count], limits[:dof_count]
                )
                torque_keys.append(key)
            elif buffer.mode == ControlMode.PD_EXPLICIT:
                kp = torch.as_tensor(buffer.kp, dtype=torch.float32, device=device)
                kd = torch.as_tensor(buffer.kd, dtype=torch.float32, device=device)
                limits = torch.as_tensor(
                    self.articulations[key].articulation.get_effort_limits(),
                    dtype=torch.float32,
                    device=device,
                )
                torque = (
                    kp[:dof_count] * (cmd[:dof_count] - qpos[row, dof_indices])
                    - kd[:dof_count] * qvel[row, dof_indices]
                )
                qf[row, dof_indices] = torch.clamp(
                    torque, -limits[:dof_count], limits[:dof_count]
                )
                torque_keys.append(key)

        if pos_keys:
            gpu_system.apply_articulation_target_joint_positions(
                self._articulation_gpu_index_view_for_keys(
                    tuple(pos_keys),
                    device=device,
                    name="kangsimworld_articulation_pos_command_indices",
                )
            )
        if vel_keys:
            gpu_system.apply_articulation_target_joint_velocities(
                self._articulation_gpu_index_view_for_keys(
                    tuple(vel_keys),
                    device=device,
                    name="kangsimworld_articulation_vel_command_indices",
                )
            )
        if torque_keys:
            gpu_system.apply_articulation_joint_forces(
                self._articulation_gpu_index_view_for_keys(
                    tuple(torque_keys),
                    device=device,
                    name="kangsimworld_articulation_torque_command_indices",
                )
            )

    def apply_resets(self):
        if (
            not self._pending_reset_keys
            and not self._gpu_root_reset_batches
            and not self._gpu_dof_reset_batches
        ):
            return False
        pending_keys = list(self._pending_reset_keys)
        self._pending_reset_keys.clear()
        gpu_root_batches = self._gpu_root_reset_batches
        gpu_dof_batches = self._gpu_dof_reset_batches
        self._gpu_root_reset_batches = []
        self._gpu_dof_reset_batches = []
        gpu_rigid_root_resets: list[tuple[tuple[int, int], RootStateReset]] = []
        gpu_articulation_root_resets: list[tuple[tuple[int, int], RootStateReset]] = []
        gpu_articulation_dof_resets: list[tuple[tuple[int, int], DofStateReset]] = []
        for key in pending_keys:
            reset = self.resets[key]
            if not reset.pending:
                continue
            env_id, obj_id = key
            cache = self.state.record(env_id, obj_id).cache
            self._clear_body_force_for_key(key)
            if reset.root is not None:
                if self._gpu_system is not None and key in self.rigids:
                    gpu_rigid_root_resets.append((key, reset.root))
                elif self._gpu_system is not None and key in self.articulations:
                    gpu_articulation_root_resets.append((key, reset.root))
                else:
                    cache.set_root(
                        reset.root.pos,
                        reset.root.rot_xyzw,
                        reset.root.linear_velocity,
                        reset.root.angular_velocity,
                    )
            if reset.dof is not None:
                if self._gpu_system is not None and key in self.articulations:
                    gpu_articulation_dof_resets.append((key, reset.dof))
                else:
                    cache.set_dofs(reset.dof.positions, reset.dof.velocities)
            self.resets[key] = ResetBuffer()
        if gpu_rigid_root_resets:
            self._apply_gpu_rigid_root_resets(gpu_rigid_root_resets)
            self._clear_gpu_rigid_commands_after_reset(
                tuple(key for key, _ in gpu_rigid_root_resets)
            )
            self.state.mark_stale()
        if gpu_articulation_root_resets:
            self._apply_gpu_articulation_root_resets(gpu_articulation_root_resets)
            self.state.mark_stale()
        if gpu_articulation_dof_resets:
            self._apply_gpu_articulation_dof_resets(gpu_articulation_dof_resets)
            self.state.mark_stale()
        if gpu_root_batches:
            self._apply_gpu_root_reset_batches(gpu_root_batches)
            self.state.mark_stale()
        if gpu_dof_batches:
            self._apply_gpu_dof_reset_batches(gpu_dof_batches)
            self.state.mark_stale()
        gpu_articulation_reset_keys = tuple(
            dict.fromkeys(
                [key for key, _ in gpu_articulation_root_resets]
                + [key for key, _ in gpu_articulation_dof_resets]
            )
        )
        if gpu_articulation_reset_keys:
            self._clear_gpu_articulation_commands_after_reset(
                gpu_articulation_reset_keys
            )
        if (
            gpu_rigid_root_resets
            or gpu_articulation_root_resets
            or gpu_articulation_dof_resets
            or gpu_root_batches
            or gpu_dof_batches
        ):
            self.state.gpu.invalidate_articulation_dynamics()
            self._clear_gpu_contact_data_after_reset()
            return True
        return False

    def _apply_gpu_rigid_root_resets(
        self, resets: list[tuple[tuple[int, int], RootStateReset]]
    ):
        torch = _torch()
        gpu_system = self.gpu_system
        rigid_view = gpu_system.rigid_data()
        device_id = int(rigid_view.device_id)
        device = torch.device(f"cuda:{device_id}")
        rigid_state = rigid_view.torch()
        keys = tuple(key for key, _ in resets)

        for reset_index, (key, reset) in enumerate(resets):
            row = self.rigid_gpu_row(key[0], key[1])
            rigid_state[row, 0:3] = torch.as_tensor(
                reset.pos, dtype=torch.float32, device=device
            )
            rigid_state[row, 3:7] = torch.as_tensor(
                reset.rot_xyzw, dtype=torch.float32, device=device
            )
            linear_velocity = (
                np.zeros(3, dtype=np.float32)
                if reset.linear_velocity is None
                else reset.linear_velocity
            )
            angular_velocity = (
                np.zeros(3, dtype=np.float32)
                if reset.angular_velocity is None
                else reset.angular_velocity
            )
            rigid_state[row, 7:10] = torch.as_tensor(
                linear_velocity, dtype=torch.float32, device=device
            )
            rigid_state[row, 10:13] = torch.as_tensor(
                angular_velocity, dtype=torch.float32, device=device
            )

        index_view = self._rigid_gpu_index_view_for_keys(
            keys, device=device, name="kangsimworld_rigid_root_reset_indices"
        )
        gpu_system.apply_rigid_data(index_view)

    def _apply_gpu_articulation_root_resets(
        self, resets: list[tuple[tuple[int, int], RootStateReset]]
    ):
        torch = _torch()
        gpu_system = self.gpu_system
        link_view = gpu_system.articulation_link_data()
        device_id = int(link_view.device_id)
        device = torch.device(f"cuda:{device_id}")
        link_state = link_view.torch()
        keys = tuple(key for key, _ in resets)

        for key, reset in resets:
            row = self.articulation_gpu_row(key[0], key[1])
            root_link_index = int(
                self.articulations[key].articulation.get_link_indices()[0]
            )
            link_state[row, root_link_index, 0:3] = torch.as_tensor(
                reset.pos, dtype=torch.float32, device=device
            )
            link_state[row, root_link_index, 3:7] = torch.as_tensor(
                reset.rot_xyzw, dtype=torch.float32, device=device
            )
            linear_velocity = (
                np.zeros(3, dtype=np.float32)
                if reset.linear_velocity is None
                else reset.linear_velocity
            )
            angular_velocity = (
                np.zeros(3, dtype=np.float32)
                if reset.angular_velocity is None
                else reset.angular_velocity
            )
            link_state[row, root_link_index, 7:10] = torch.as_tensor(
                linear_velocity, dtype=torch.float32, device=device
            )
            link_state[row, root_link_index, 10:13] = torch.as_tensor(
                angular_velocity, dtype=torch.float32, device=device
            )

        index_view = self._articulation_gpu_index_view_for_keys(
            keys, device=device, name="kangsimworld_articulation_root_reset_indices"
        )
        gpu_system.apply_articulation_root_pose(index_view)
        gpu_system.apply_articulation_root_vel(index_view)
        gpu_system.update_articulation_kinematics()

    def _apply_gpu_articulation_dof_resets(
        self, resets: list[tuple[tuple[int, int], DofStateReset]]
    ):
        torch = _torch()
        gpu_system = self.gpu_system
        qpos_view = gpu_system.articulation_joint_positions()
        qvel_view = gpu_system.articulation_joint_velocities()
        device_id = int(qpos_view.device_id)
        device = torch.device(f"cuda:{device_id}")
        qpos = qpos_view.torch()
        qvel = qvel_view.torch()
        keys = tuple(key for key, _ in resets)

        for key, reset in resets:
            row = self.articulation_gpu_row(key[0], key[1])
            dof_count = int(gpu_system.articulation_dof_count(row))
            dof_indices = self._articulation_gpu_dof_indices(key, device=device)
            qpos[row, dof_indices] = torch.as_tensor(
                reset.positions, dtype=torch.float32, device=device
            )[:dof_count]
            velocity = (
                np.zeros(dof_count, dtype=np.float32)
                if reset.velocities is None
                else reset.velocities
            )
            qvel[row, dof_indices] = torch.as_tensor(
                velocity, dtype=torch.float32, device=device
            )[:dof_count]

        index_view = self._articulation_gpu_index_view_for_keys(
            keys, device=device, name="kangsimworld_articulation_dof_reset_indices"
        )
        gpu_system.apply_articulation_joint_positions(index_view)
        gpu_system.apply_articulation_joint_velocities(index_view)
        gpu_system.update_articulation_kinematics()

    def _apply_gpu_root_reset_batches(self, batches: list[GpuRootStateResetBatch]):
        gpu_system = self.gpu_system
        link_state = None
        rigid_state = None
        articulation_changed = False

        for batch in batches:
            self._clear_batch_body_forces(batch.env_ids, batch.obj_id)
            first_key = (0, batch.obj_id)
            if first_key in self.articulations:
                self.batch_commands.pop(batch.obj_id, None)
                if link_state is None:
                    link_state = gpu_system.articulation_link_data().torch()
                rows = self._articulation_gpu_rows_for_obj(
                    batch.obj_id, batch.env_ids, device=link_state.device
                )
                root_link = int(
                    self.articulations[first_key].articulation.get_link_indices()[0]
                )
                link_state[rows, root_link, :] = batch.state
                index_view = self._gpu_row_index_view(
                    rows, "kangsimworld_articulation_root_reset_batch_indices"
                )
                gpu_system.apply_articulation_root_pose(index_view)
                gpu_system.apply_articulation_root_vel(index_view)
                gpu_system.clear_articulation_commands(index_view)
                articulation_changed = True
            else:
                if rigid_state is None:
                    rigid_state = gpu_system.rigid_data().torch()
                rows = self._rigid_gpu_rows_for_obj(
                    batch.obj_id, batch.env_ids, device=rigid_state.device
                )
                rigid_state[rows, :] = batch.state
                index_view = self._gpu_row_index_view(
                    rows, "kangsimworld_rigid_root_reset_batch_indices"
                )
                gpu_system.apply_rigid_data(index_view)
                gpu_system.clear_rigid_commands(index_view)

        if articulation_changed:
            gpu_system.update_articulation_kinematics()

    def _apply_gpu_dof_reset_batches(self, batches: list[GpuDofStateResetBatch]):
        gpu_system = self.gpu_system
        qpos = gpu_system.articulation_joint_positions().torch()
        qvel = gpu_system.articulation_joint_velocities().torch()
        device = qpos.device

        for batch in batches:
            self.batch_commands.pop(batch.obj_id, None)
            self._clear_batch_body_forces(batch.env_ids, batch.obj_id)
            rows = self._articulation_gpu_rows_for_obj(
                batch.obj_id, batch.env_ids, device=device
            )
            first_key = (0, batch.obj_id)
            dof_indices = self._articulation_gpu_dof_indices(first_key, device=device)
            dof_count = batch.state.shape[1]
            cols = dof_indices[:dof_count]
            qpos[rows[:, None], cols[None, :]] = batch.state[..., 0]
            qvel[rows[:, None], cols[None, :]] = batch.state[..., 1]
            index_view = self._gpu_row_index_view(
                rows, "kangsimworld_articulation_dof_reset_batch_indices"
            )
            gpu_system.apply_articulation_joint_positions(index_view)
            gpu_system.apply_articulation_joint_velocities(index_view)
            gpu_system.clear_articulation_commands(index_view)

        if batches:
            gpu_system.update_articulation_kinematics()

    def _clear_batch_body_forces(self, env_ids, obj_id: int):
        if not self._active_body_force_keys:
            return
        torch = _torch()
        if torch.is_tensor(env_ids):
            # Body-force commands are stored in a Python key map. Only this
            # uncommon mixed path requires materializing CUDA ids on host.
            env_ids = env_ids.detach().cpu().tolist()
        reset_envs = set(env_ids)
        active = tuple(
            key
            for key in self._active_body_force_keys
            if key[1] == obj_id and key[0] in reset_envs
        )
        for key in active:
            self._clear_body_force_for_key(key)

    def _gpu_row_index_view(self, rows, name: str):
        from ..utils import to_gpu_array_view

        torch = _torch()
        indices = rows.to(dtype=torch.int32).contiguous()
        return to_gpu_array_view(indices, dtype=torch.int32, name=name)

    def _clear_gpu_rigid_commands_after_reset(self, keys: tuple[tuple[int, int], ...]):
        if not keys:
            return
        torch = _torch()
        gpu_system = self.gpu_system
        rigid_view = gpu_system.rigid_data()
        device = torch.device(f"cuda:{int(rigid_view.device_id)}")
        index_view = self._rigid_gpu_index_view_for_keys(
            keys, device=device, name="kangsimworld_rigid_reset_clear_indices"
        )
        gpu_system.clear_rigid_commands(index_view)

    def _clear_gpu_articulation_commands_after_reset(
        self, keys: tuple[tuple[int, int], ...]
    ):
        if not keys:
            return
        torch = _torch()
        gpu_system = self.gpu_system
        link_view = gpu_system.articulation_link_data()
        device = torch.device(f"cuda:{int(link_view.device_id)}")
        reset_obj_ids = {key[1] for key in keys}
        for obj_id in reset_obj_ids:
            # A dense batch command cannot represent a reset-created hole.
            # Drop it instead of silently reapplying stale control on the next
            # step; the caller can submit the next policy command explicitly.
            self.batch_commands.pop(obj_id, None)
        for key in keys:
            self.commands[key] = CommandBuffer(ControlMode.NONE)

        index_view = self._articulation_gpu_index_view_for_keys(
            keys, device=device, name="kangsimworld_articulation_reset_clear_indices"
        )
        gpu_system.clear_articulation_commands(index_view)

    def _clear_gpu_contact_data_after_reset(self):
        clear = getattr(self.gpu_system, "clear_contact_data", None)
        if clear is not None:
            clear()

    def set_body_force(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        body_id: int,
        force,
    ):
        env_ids = env_id_list(env_id, self.num_envs)
        for env_index, eid in enumerate(env_ids):
            key = (eid, int(obj_id))
            forces = self.body_forces[key]
            body_idx = int(body_id)
            if body_idx < 0 or body_idx >= forces.shape[0]:
                raise IndexError(
                    f"body_id {body_idx} out of range for env={eid}, obj={obj_id}"
                )
            forces[body_idx] = select_env_value(force, env_index, len(env_ids), (3,))
            if np.any(forces[body_idx]):
                self._active_body_force_keys.add(key)
            elif not np.any(forces):
                self._active_body_force_keys.discard(key)
            self.body_force_positions[key][body_idx].fill(np.nan)

    def set_body_force_at_position(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        body_id: int,
        force,
        position,
    ):
        env_ids = env_id_list(env_id, self.num_envs)
        for env_index, eid in enumerate(env_ids):
            key = (eid, int(obj_id))
            forces = self.body_forces[key]
            positions = self.body_force_positions[key]
            body_idx = int(body_id)
            if body_idx < 0 or body_idx >= forces.shape[0]:
                raise IndexError(
                    f"body_id {body_idx} out of range for env={eid}, obj={obj_id}"
                )
            forces[body_idx] = select_env_value(force, env_index, len(env_ids), (3,))
            positions[body_idx] = select_env_value(
                position, env_index, len(env_ids), (3,)
            )
            if np.any(forces[body_idx]):
                self._active_body_force_keys.add(key)
            elif not np.any(forces):
                self._active_body_force_keys.discard(key)
                positions.fill(np.nan)

    def apply_body_forces(self):
        if not self._active_body_force_keys:
            return
        for key in self._active_body_force_keys:
            forces = self.body_forces[key]
            active_body_ids = np.flatnonzero(np.any(forces != 0.0, axis=1))
            if active_body_ids.size == 0:
                continue
            positions = self.body_force_positions[key]
            articulation = self.articulations.get(key)
            rigid = self.rigids.get(key)
            if articulation is None and rigid is None:
                continue
            for body_id in active_body_ids:
                force = forces[body_id]
                position = positions[body_id]
                has_position = np.all(np.isfinite(position))
                if articulation is not None:
                    if has_position:
                        articulation.articulation.add_link_force_at_position(
                            int(body_id), force, position
                        )
                    else:
                        articulation.articulation.add_link_force(int(body_id), force)
                elif rigid is not None:
                    if has_position:
                        rigid.rigid.add_force_at_position(force, position)
                    else:
                        rigid.rigid.add_force(force)

    def clear_body_forces(self):
        if not self._active_body_force_keys:
            return
        for key in self._active_body_force_keys:
            self.body_forces[key].fill(0.0)
            self.body_force_positions[key].fill(np.nan)
        self._active_body_force_keys.clear()

    def _clear_body_force_for_key(self, key: tuple[int, int]):
        if key in self.body_forces:
            self.body_forces[key].fill(0.0)
        if key in self.body_force_positions:
            self.body_force_positions[key].fill(np.nan)
        self._active_body_force_keys.discard(key)

    def step(
        self, substeps: int = 1, refresh: bool = True, apply_commands: bool = True
    ):
        """Advance simulation and return ``world.state``.

        Args:
            substeps: Number of PhysX simulation steps to run. ``0`` applies
                queued resets/commands and sensor cleanup without advancing
                time; this is useful for Direct GPU reset-only frames.
            refresh: When ``True``, refresh ``world.state`` before returning.
                GPU simulation users can set this to ``False`` and read
                canonical CUDA views directly to avoid CPU readback.
            apply_commands: When ``True``, flush queued root state/control
                commands before stepping. Set to ``False`` for reset-only
                frames that should not re-apply user commands.

        For GPU simulation this returns a CPU/Torch snapshot. Use GPU view
        helpers for the latest canonical CUDA state when avoiding readback.
        """
        reset_applied = self.apply_resets()
        if apply_commands:
            self.apply_commands()
        substep_count = int(substeps)
        for _ in range(substep_count):
            self.apply_body_forces()
            self.physics.step()
            self.sim_time += self.sim_dt
        self.clear_body_forces()
        if self._uses_gpu_sim:
            if substep_count > 0:
                self.state.gpu.invalidate_articulation_dynamics()
            self.state.mark_stale()
        if reset_applied and substep_count == 0:
            self.clear_sensor_outputs()
        else:
            self.refresh_sensors()
        if refresh:
            self.state.refresh()
        return self.state

    def advance(
        self,
        duration: float,
        refresh: bool = True,
        apply_commands: bool = True,
    ):
        """Advance by a requested amount of simulation time using fixed steps.

        ``duration`` never changes the PhysX timestep. The method converts it
        into as many whole ``sim_dt`` steps as are currently due and retains
        any fractional remainder for the next call. This is useful from an app
        ``fixed_update(fixed_dt)`` callback whose control frequency may differ
        from the world's physics frequency.

        For example, a 120 Hz world advances 2 substeps for ``1 / 60`` seconds
        and 4 substeps for ``1 / 30`` seconds. Durations smaller than
        ``sim_dt`` may execute no physics until enough time accumulates.

        Args:
            duration: Requested simulation duration in seconds.
            refresh: Forwarded to :meth:`step` when at least one physics step
                is due.
            apply_commands: Forwarded to :meth:`step` when at least one physics
                step is due.

        Returns:
            The current ``world.state``.
        """
        duration_seconds = float(duration)
        if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
            raise ValueError("duration must be a finite non-negative value")
        if duration_seconds == 0.0:
            return self.state

        accumulated = self._advance_time_remainder + duration_seconds
        step_ratio = accumulated / self.sim_dt
        nearest_steps = round(step_ratio)
        if math.isclose(step_ratio, nearest_steps, rel_tol=1.0e-6, abs_tol=1.0e-9):
            substeps = int(nearest_steps)
        else:
            substeps = int(math.floor(step_ratio))
        self._advance_time_remainder = accumulated - substeps * self.sim_dt
        if self._advance_time_remainder < 0.0:
            self._advance_time_remainder = 0.0
        if substeps == 0:
            return self.state
        return self.step(
            substeps=substeps,
            refresh=refresh,
            apply_commands=apply_commands,
        )

    def refresh(self):
        """Refresh and return ``world.state``.

        In GPU simulation this performs an explicit CPU/Torch snapshot update.
        """
        return self.state.refresh()

    def articulation(self, env_id: int = 0, obj_id: int = 0):
        return self.articulations[(int(env_id), int(obj_id))].articulation

    def rigid(self, env_id: int = 0, obj_id: int = 0):
        key = (int(env_id), int(obj_id))
        return (self.rigids | self.static_rigids)[key].rigid

    def init_gpu_system(
        self, cuda_device_id: int | None = None, stream_handle: int | None = None
    ):
        """Initialize explicit PhysX GPU mirrors and cache rigid row mappings.

        Call this after registering simulation objects and before using
        low-level GPU mirror apply/fetch paths. It intentionally performs a
        visible runtime boundary instead of hiding PhysX GPU warm-up inside a
        high-level setter.
        """
        if not self._uses_gpu_sim:
            raise RuntimeError(
                "KangSimWorld.init_gpu_system() requires sim_device='cuda'"
            )
        physics = getattr(_ke, "physics", None)
        if (
            physics is None
            or not hasattr(physics, "PhysicsGpuSystem")
            or not hasattr(physics, "GpuPhysicsConfig")
        ):
            raise RuntimeError("KangEngine was built without PhysicsGpuSystem bindings")
        if self._gpu_system is not None:
            return self._gpu_system

        if cuda_device_id is None:
            cuda_device_id = (
                0 if self.sim_device.index is None else self.sim_device.index
            )
        config = _ke.physics.GpuPhysicsConfig()
        config.cuda_device_id = int(cuda_device_id)
        gpu_system = _ke.physics.PhysicsGpuSystem(self.physics, config)
        gpu_system.init()
        if stream_handle is not None:
            gpu_system.set_cuda_stream(int(stream_handle))

        self._gpu_system = gpu_system
        self._rigid_gpu_rows = {
            key: int(gpu_system.rigid_row(record.rigid))
            for key, record in self.rigids.items()
        }
        self._articulation_gpu_rows = {
            key: int(gpu_system.articulation_row(record.articulation))
            for key, record in self.articulations.items()
        }
        for key, row in self._articulation_gpu_rows.items():
            articulation = self.articulations[key].articulation
            logical_dofs = int(articulation.num_dofs())
            physx_dofs = int(gpu_system.articulation_dof_count(row))
            dof_indices = list(articulation.get_dof_gpu_indices())
            if logical_dofs != physx_dofs:
                raise RuntimeError(
                    "KangEngine/PhysX articulation DOF mismatch for "
                    f"env={key[0]}, obj={key[1]}: logical={logical_dofs}, "
                    f"PhysX GPU={physx_dofs}. Check for duplicate or "
                    "unsupported multi-axis joint mappings."
                )
            if sorted(dof_indices) != list(range(physx_dofs)):
                raise RuntimeError(
                    f"Invalid articulation GPU DOF mapping for env={key[0]}, obj={key[1]}: {dof_indices}"
                )
        self._rigid_gpu_index_tensors.clear()
        self._articulation_gpu_index_tensors.clear()
        self._articulation_gpu_row_tensors.clear()
        self._rigid_gpu_row_tensors.clear()
        return gpu_system

    @property
    def gpu_system(self):
        if self._gpu_system is None:
            raise RuntimeError(
                "GPU system is not initialized; call init_gpu_system() first"
            )
        return self._gpu_system

    def rigid_gpu_row(self, env_id: int, obj_id: int = 0) -> int:
        key = (int(env_id), int(obj_id))
        if key not in self._rigid_gpu_rows:
            raise KeyError(
                f"no cached PhysX GPU rigid row for env={key[0]}, obj={key[1]}"
            )
        return self._rigid_gpu_rows[key]

    def rigid_gpu_index_view(self, env_id: EnvIdLike, obj_id: int = 0):
        """Return a cached CUDA int32 logical-row index view for rigid batches."""
        rigid_view = self.gpu_system.rigid_data()
        keys = tuple((env, int(obj_id)) for env in env_id_list(env_id, self.num_envs))

        torch = _torch()
        device = torch.device(f"cuda:{int(rigid_view.device_id)}")
        return self._rigid_gpu_index_view_for_keys(
            keys, device=device, name="kangsimworld_rigid_indices"
        )

    def articulation_gpu_row(self, env_id: int, obj_id: int = 0) -> int:
        key = (int(env_id), int(obj_id))
        if key not in self._articulation_gpu_rows:
            raise KeyError(
                f"no cached PhysX GPU articulation row for env={key[0]}, obj={key[1]}"
            )
        return self._articulation_gpu_rows[key]

    def articulation_gpu_index_view(self, env_id: EnvIdLike, obj_id: int = 0):
        """Return a cached CUDA int32 logical-row index view for articulations."""
        link_view = self.gpu_system.articulation_link_data()
        keys = tuple((env, int(obj_id)) for env in env_id_list(env_id, self.num_envs))

        torch = _torch()
        device = torch.device(f"cuda:{int(link_view.device_id)}")
        return self._articulation_gpu_index_view_for_keys(
            keys, device=device, name="kangsimworld_articulation_indices"
        )

    def get_gpu_rigid_data(self, *, fetch: bool = True):
        """Return the full PhysX GPU rigid mirror as a Torch CUDA view."""
        return self.state.gpu.rigid_data_tensor(fetch=fetch)

    def get_gpu_rigid_accelerations(self, *, fetch: bool = True):
        """Return rigid COM accelerations as ``[lin xyz, ang xyz]``.

        Shape is ``[rigid_count, 6]``. The physics scene must be created with
        ``PhysicsConfig(enable_body_accelerations=True)``.
        """
        return self.state.gpu.rigid_accelerations_tensor(fetch=fetch)

    def get_gpu_articulation_link_data(
        self, *, fetch_pose: bool = True, fetch_velocity: bool = True
    ):
        """Return the full articulation link mirror as a Torch CUDA view.

        Layout is ``[articulation_count, max_links, 13]`` with
        ``[pos xyz, quat xyzw, linear velocity xyz, angular velocity xyz]``.
        """
        if fetch_velocity:
            self.gpu_system.fetch_articulation_link_vel()
        return self.state.gpu.articulation_link_data_tensor(fetch=fetch_pose)

    def get_gpu_articulation_joint_positions(self, *, fetch: bool = True):
        return self.state.gpu.articulation_joint_positions_tensor(fetch=fetch)

    def get_gpu_articulation_joint_velocities(self, *, fetch: bool = True):
        return self.state.gpu.articulation_joint_velocities_tensor(fetch=fetch)

    def get_gpu_articulation_joint_accelerations(self, *, fetch: bool = True):
        return self.state.gpu.articulation_joint_accelerations_tensor(fetch=fetch)

    def get_gpu_articulation_joint_forces(self, *, fetch: bool = True):
        return self.state.gpu.articulation_joint_forces_tensor(fetch=fetch)

    def get_gpu_articulation_target_joint_positions(self, *, fetch: bool = True):
        return self.state.gpu.articulation_target_joint_positions_tensor(fetch=fetch)

    def get_gpu_articulation_target_joint_velocities(self, *, fetch: bool = True):
        return self.state.gpu.articulation_target_joint_velocities_tensor(fetch=fetch)

    def get_gpu_articulation_link_incoming_joint_forces(self, *, fetch: bool = True):
        return self.state.gpu.articulation_link_incoming_joint_forces_tensor(
            fetch=fetch
        )

    def get_gpu_articulation_link_accelerations(self, *, fetch: bool = True):
        """Return link COM acceleration `[lin xyz, ang xyz]`."""
        return self.state.gpu.articulation_link_accelerations_tensor(fetch=fetch)

    def get_gpu_articulation_link_forces(self):
        """Return the writable `[articulation, max_links, 3]` force buffer."""
        return self.state.gpu.articulation_link_forces_tensor()

    def get_gpu_articulation_link_torques(self):
        """Return the writable `[articulation, max_links, 3]` torque buffer."""
        return self.state.gpu.articulation_link_torques_tensor()

    def apply_gpu_articulation_link_wrenches(self, *, forces=True, torques=True):
        """Submit the current dense CUDA link force/torque command buffers."""
        if forces:
            self.gpu_system.apply_articulation_link_forces()
        if torques:
            self.gpu_system.apply_articulation_link_torques()

    def set_gpu_articulation_link_wrenches(self, *, forces=None, torques=None):
        """Copy dense CUDA wrench tensors into the PhysX command buffers."""
        if forces is None and torques is None:
            raise ValueError("forces or torques must be provided")
        torch = _torch()
        for name, value in (("forces", forces), ("torques", torques)):
            if value is None:
                continue
            if not torch.is_tensor(value) or value.device.type != "cuda":
                raise TypeError(f"{name} must be a CUDA tensor")
            if value.dtype != torch.float32 or not value.is_contiguous():
                raise TypeError(f"{name} must be contiguous CUDA float32")
        if forces is not None:
            target = self.get_gpu_articulation_link_forces()
            if forces.device != target.device:
                raise ValueError(
                    f"forces device {forces.device} must be {target.device}"
                )
            if tuple(forces.shape) != tuple(target.shape):
                raise ValueError(
                    f"forces shape {tuple(forces.shape)} must be {tuple(target.shape)}"
                )
            target.copy_(forces)
        if torques is not None:
            target = self.get_gpu_articulation_link_torques()
            if torques.device != target.device:
                raise ValueError(
                    f"torques device {torques.device} must be {target.device}"
                )
            if tuple(torques.shape) != tuple(target.shape):
                raise ValueError(
                    f"torques shape {tuple(torques.shape)} must be {tuple(target.shape)}"
                )
            target.copy_(torques)
        self.apply_gpu_articulation_link_wrenches(
            forces=forces is not None, torques=torques is not None
        )

    def get_gpu_articulation_dense_jacobians(self, *, compute: bool = True):
        """Return lazy-computed PhysX dense-Jacobian blocks.

        Shape is ``[articulation_count, max_jacobian_rows *
        max_generalized_dofs]``. For row ``i``, reshape the leading
        ``rows * cols`` values using ``get_gpu_articulation_dynamics_shape(i)``.
        """
        return self.state.gpu.articulation_dense_jacobians_tensor(compute=compute)

    def get_gpu_articulation_mass_matrices(self, *, compute: bool = True):
        """Return lazy-computed PhysX mass-matrix blocks.

        Each row begins with a packed ``n * n`` matrix, where ``n`` is that
        articulation's generalized DOF count.
        """
        return self.state.gpu.articulation_mass_matrices_tensor(compute=compute)

    def get_gpu_articulation_gravity_forces(self, *, compute: bool = True):
        """Return lazy-computed generalized gravity-compensation forces."""
        return self.state.gpu.articulation_gravity_forces_tensor(compute=compute)

    def get_gpu_articulation_coriolis_forces(self, *, compute: bool = True):
        """Return lazy-computed generalized Coriolis/centrifugal forces."""
        return self.state.gpu.articulation_coriolis_forces_tensor(compute=compute)

    def get_gpu_articulation_com_world(self, *, compute: bool = True):
        return self.state.gpu.articulation_com_world_tensor(compute=compute)

    def get_gpu_articulation_com_root(self, *, compute: bool = True):
        return self.state.gpu.articulation_com_root_tensor(compute=compute)

    def get_gpu_articulation_centroidal_dynamics(self, *, compute: bool = True):
        """Return `(centroidal_momentum_matrix, bias_force)` CUDA blocks."""
        gpu = self.state.gpu
        return (
            gpu.articulation_centroidal_momentum_matrices_tensor(compute=compute),
            gpu.articulation_centroidal_bias_forces_tensor(compute=False),
        )

    def get_gpu_articulation_dynamics_shape(self, articulation_row: int):
        """Return ``(jacobian_rows, generalized_dofs)`` for one GPU row."""
        row = int(articulation_row)
        return (
            int(self.gpu_system.articulation_jacobian_row_count(row)),
            int(self.gpu_system.articulation_generalized_dof_count(row)),
        )

    def get_gpu_contact_pairs(self, *, fetch: bool = True):
        return self.state.gpu.contact_pairs_tensor(fetch=fetch)

    def get_gpu_contact_pair_count(self, *, fetch: bool = True):
        return self.state.gpu.contact_pair_count_tensor(fetch=fetch)

    def get_gpu_contact_pair_headers(self, *, fetch: bool = True):
        return self.state.gpu.contact_pair_headers_tensor(fetch=fetch)

    def get_gpu_contact_pair_body_refs(self, *, fetch: bool = True):
        return self.state.gpu.contact_pair_body_refs_tensor(fetch=fetch)

    def get_gpu_contact_points(self, *, fetch: bool = True):
        return self.state.gpu.contact_points_tensor(fetch=fetch)

    def get_gpu_contact_point_count(self, *, fetch: bool = True):
        return self.state.gpu.contact_point_count_tensor(fetch=fetch)

    def get_gpu_contact_point_pair_indices(self, *, fetch: bool = True):
        return self.state.gpu.contact_point_pair_indices_tensor(fetch=fetch)

    def _rigid_gpu_index_view_for_keys(self, keys, *, device, name: str):
        torch = _torch()
        from ..utils import to_gpu_array_view

        keys = tuple((int(env_id), int(obj_id)) for env_id, obj_id in keys)
        tensor = self._rigid_gpu_index_tensors.get(keys)
        if tensor is None or tensor.device != device:
            rows = [self.rigid_gpu_row(env_id, obj_id) for env_id, obj_id in keys]
            tensor = torch.tensor(rows, dtype=torch.int32, device=device)
            self._rigid_gpu_index_tensors[keys] = tensor
        return to_gpu_array_view(tensor, dtype=torch.int32, name=name)

    def _articulation_gpu_index_view_for_keys(self, keys, *, device, name: str):
        torch = _torch()
        from ..utils import to_gpu_array_view

        keys = tuple((int(env_id), int(obj_id)) for env_id, obj_id in keys)
        tensor = self._articulation_gpu_index_tensors.get(keys)
        if tensor is None or tensor.device != device:
            rows = [
                self.articulation_gpu_row(env_id, obj_id) for env_id, obj_id in keys
            ]
            tensor = torch.tensor(rows, dtype=torch.int32, device=device)
            self._articulation_gpu_index_tensors[keys] = tensor
        return to_gpu_array_view(tensor, dtype=torch.int32, name=name)

    def _articulation_gpu_batch_command_index_view(
        self, batch: BatchCommandBuffer, *, device, name: str
    ):
        """Return ``None`` when a full batch already matches PhysX row order.

        Passing sparse indices makes the native layer pack the selected rows
        into a scratch buffer. A single cloned object spanning every
        articulation is already the dense PhysX buffer, so that pack is pure
        overhead.
        """
        oid = int(batch.obj_id)
        is_dense = self._dense_cuda_batch_command_obj_ids.get(oid)
        if is_dense is None:
            is_dense = (
                batch.env_ids == self._all_env_ids
                and int(self.gpu_system.articulation_count()) == self.num_envs
                and all(
                    self.articulation_gpu_row(env_id, oid) == env_id
                    for env_id in self._all_env_ids
                )
            )
            self._dense_cuda_batch_command_obj_ids[oid] = is_dense
        if is_dense:
            return None
        return self._articulation_gpu_index_view_for_keys(
            tuple((eid, oid) for eid in batch.env_ids),
            device=device,
            name=name,
        )

    def _articulation_gpu_rows_for_obj(self, obj_id: int, env_ids, *, device):
        torch = _torch()
        oid = int(obj_id)
        all_rows = self._articulation_gpu_row_tensors.get(oid)
        if all_rows is None or all_rows.device != device:
            all_rows = torch.tensor(
                [
                    self.articulation_gpu_row(env_id, oid)
                    for env_id in range(self.num_envs)
                ],
                dtype=torch.long,
                device=device,
            )
            self._articulation_gpu_row_tensors[oid] = all_rows
        if isinstance(env_ids, tuple) and env_ids == self._all_env_ids:
            return all_rows
        index = torch.as_tensor(env_ids, dtype=torch.long, device=device)
        return all_rows.index_select(0, index)

    def _articulation_gpu_effort_limits(self, obj_id: int, *, device):
        torch = _torch()
        key = (int(obj_id), device)
        limits = self._articulation_gpu_effort_limit_tensors.get(key)
        if limits is None:
            limits = torch.as_tensor(
                self.articulations[(0, int(obj_id))].articulation.get_effort_limits(),
                dtype=torch.float32,
                device=device,
            )
            self._articulation_gpu_effort_limit_tensors[key] = limits
        return limits

    def _rigid_gpu_rows_for_obj(self, obj_id: int, env_ids, *, device):
        torch = _torch()
        oid = int(obj_id)
        all_rows = self._rigid_gpu_row_tensors.get(oid)
        if all_rows is None or all_rows.device != device:
            all_rows = torch.tensor(
                [self.rigid_gpu_row(env_id, oid) for env_id in range(self.num_envs)],
                dtype=torch.long,
                device=device,
            )
            self._rigid_gpu_row_tensors[oid] = all_rows
        index = torch.as_tensor(env_ids, dtype=torch.long, device=device)
        return all_rows.index_select(0, index)

    def _articulation_gpu_dof_indices(self, key, *, device):
        return self.state.record(key[0], key[1]).cache.dof_gpu_indices.to(device)

    def _require_gpu_runtime_uninitialized(self, operation: str):
        if self._gpu_system is not None:
            raise RuntimeError(
                f"KangSimWorld.{operation}() cannot be called after "
                "init_gpu_system(); register objects before initializing GPU "
                "mirrors"
            )

    def set_gpu_root_state_batch(self, env_ids, obj_id: int, state):
        """Queue CUDA root states without creating per-environment reset objects.

        ``env_ids`` may be a one-dimensional CUDA int32/int64 tensor. It stays
        on device through row selection and the indexed PhysX apply call.
        ``state`` must be a contiguous ``float32`` CUDA tensor shaped ``[N, 13]``
        with position, xyzw rotation, linear velocity, and angular velocity.
        """
        ids = self._gpu_reset_env_ids(env_ids)
        oid = int(obj_id)
        self._validate_gpu_reset_object(ids, oid, allow_rigid=True)
        first_key = (0, oid)
        device_id = int(
            self.gpu_system.articulation_link_data().device_id
            if first_key in self.articulations
            else self.gpu_system.rigid_data().device_id
        )
        self._validate_gpu_reset_env_ids_device(ids, device_id)
        self._validate_gpu_reset_tensor(state, (len(ids), 13), "root state", device_id)
        self._gpu_root_reset_batches.append(GpuRootStateResetBatch(ids, oid, state))

    def set_gpu_dof_state_batch(self, env_ids, obj_id: int, state):
        """Queue CUDA DOF position/velocity pairs as one batch.

        ``env_ids`` may be a one-dimensional CUDA int32/int64 tensor. It stays
        on device through row selection and the indexed PhysX apply call.
        ``state`` must be a contiguous ``float32`` CUDA tensor shaped
        ``[N, num_dofs, 2]``.
        """
        ids = self._gpu_reset_env_ids(env_ids)
        oid = int(obj_id)
        self._validate_gpu_reset_object(ids, oid, allow_rigid=False)
        dof_count = self.articulations[(0, oid)].articulation.num_dofs()
        device_id = int(self.gpu_system.articulation_joint_positions().device_id)
        self._validate_gpu_reset_env_ids_device(ids, device_id)
        self._validate_gpu_reset_tensor(
            state, (len(ids), int(dof_count), 2), "DOF state", device_id
        )
        self._gpu_dof_reset_batches.append(GpuDofStateResetBatch(ids, oid, state))

    def _gpu_reset_env_ids(self, env_ids):
        torch = _torch()
        if torch.is_tensor(env_ids):
            if env_ids.device.type != "cuda":
                raise TypeError(
                    "GPU reset environment ids must be a CUDA tensor or a host sequence"
                )
            if env_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError(
                    "GPU reset environment ids must have dtype torch.int32 or torch.int64"
                )
            if env_ids.ndim != 1:
                raise ValueError("GPU reset environment ids must be one-dimensional")
            if env_ids.numel() == 0:
                raise ValueError("GPU reset batch requires at least one environment")
            return env_ids.contiguous()

        ids = tuple(int(env_id) for env_id in env_ids)
        if not ids:
            raise ValueError("GPU reset batch requires at least one environment")
        if min(ids) < 0 or max(ids) >= self.num_envs:
            raise IndexError(
                f"GPU reset environment ids must be in [0, {self.num_envs})"
            )
        return ids

    def _validate_gpu_reset_env_ids_device(self, env_ids, device_id: int):
        torch = _torch()
        if not torch.is_tensor(env_ids):
            return
        value_device = env_ids.device.index if env_ids.device.index is not None else 0
        if value_device != device_id:
            raise ValueError(
                f"GPU reset environment ids are on cuda:{value_device}, expected cuda:{device_id}"
            )

    def _validate_gpu_reset_object(self, env_ids, obj_id: int, *, allow_rigid: bool):
        if self._gpu_system is None:
            raise RuntimeError("GPU reset batches require init_gpu_system()")
        first_key = (0, obj_id)
        valid = first_key in self.articulations or (
            allow_rigid and first_key in self.rigids
        )
        if not valid:
            kind = "articulation or rigid" if allow_rigid else "articulation"
            raise KeyError(f"no {kind} registered at env={env_ids[0]}, obj={obj_id}")

    def _validate_gpu_reset_tensor(self, value, shape, name: str, device_id: int):
        torch = _torch()
        if not torch.is_tensor(value) or value.device.type != "cuda":
            raise TypeError(f"GPU reset {name} must be a CUDA tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"GPU reset {name} must have dtype=torch.float32")
        if not value.is_contiguous():
            raise ValueError(f"GPU reset {name} must be contiguous")
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"GPU reset {name} expected shape {list(shape)}, got {list(value.shape)}"
            )
        value_device = value.device.index if value.device.index is not None else 0
        if value_device != device_id:
            raise ValueError(
                f"GPU reset {name} is on cuda:{value_device}, expected cuda:{device_id}"
            )

    def set_root_state(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        pos,
        rot_xyzw,
        linear_velocity=None,
        angular_velocity=None,
        immediate: bool = False,
    ):
        import sys

        torch = sys.modules.get("torch")
        cuda_pos = (
            torch is not None and torch.is_tensor(pos) and pos.device.type == "cuda"
        )
        cuda_rot = (
            torch is not None
            and torch.is_tensor(rot_xyzw)
            and rot_xyzw.device.type == "cuda"
        )
        cuda_lin_vel = (
            torch is not None
            and torch.is_tensor(linear_velocity)
            and linear_velocity.device.type == "cuda"
        )
        cuda_ang_vel = (
            torch is not None
            and torch.is_tensor(angular_velocity)
            and angular_velocity.device.type == "cuda"
        )
        cuda_root_reset = cuda_pos or cuda_rot or cuda_lin_vel or cuda_ang_vel
        if cuda_root_reset and not (cuda_pos and cuda_rot):
            raise RuntimeError(
                "If any root reset tensor is CUDA, the whole root reset must "
                "use CUDA tensors; position and rotation are required"
            )
        if immediate and cuda_root_reset:
            raise RuntimeError(
                "immediate=True does not support CUDA root reset tensors; use the deferred GPU reset path"
            )
        if immediate:
            self.state.set_root_state(
                env_id, obj_id, pos, rot_xyzw, linear_velocity, angular_velocity
            )
            env_ids = env_id_list(env_id, self.num_envs)
            for eid in env_ids:
                self.resets[(eid, int(obj_id))].root = None
                if not self.resets[(eid, int(obj_id))].pending:
                    self._pending_reset_keys.discard((eid, int(obj_id)))
            return

        env_ids = env_id_list(env_id, self.num_envs)
        if cuda_root_reset:
            if self._gpu_system is None:
                raise RuntimeError(
                    "CUDA root reset tensors require init_gpu_system(); KangSimWorld will not silently copy them to CPU"
                )
            first_key = (env_ids[0], int(obj_id))
            if first_key in self.articulations:
                gpu_device = int(self.gpu_system.articulation_link_data().device_id)
            else:
                gpu_device = int(self.gpu_system.rigid_data().device_id)
            for name, value, is_cuda, width in (
                ("position", pos, cuda_pos, 3),
                ("rotation", rot_xyzw, cuda_rot, 4),
                ("linear velocity", linear_velocity, cuda_lin_vel, 3),
                ("angular velocity", angular_velocity, cuda_ang_vel, 3),
            ):
                if value is None:
                    continue
                if not is_cuda:
                    raise RuntimeError(
                        f"CUDA root reset {name} must be a CUDA tensor or None; "
                        "mixed CPU/CUDA root reset tensors are not allowed"
                    )
                value_device = (
                    value.device.index if value.device.index is not None else 0
                )
                if value_device != gpu_device:
                    raise ValueError(
                        f"CUDA root reset {name} tensor is on cuda:{value_device}, expected cuda:{gpu_device}"
                    )
                if value.dtype != torch.float32:
                    raise TypeError(
                        f"CUDA root reset {name} tensor must have dtype=torch.float32"
                    )
                if not value.is_contiguous():
                    raise ValueError(
                        f"CUDA root reset {name} tensor must be contiguous"
                    )
                if value.ndim == 1 and tuple(value.shape) == (width,):
                    continue
                if value.ndim == 2 and tuple(value.shape) == (len(env_ids), width):
                    continue
                raise ValueError(
                    f"CUDA root reset {name} tensor expected shape "
                    f"[{width}] or [{len(env_ids)}, {width}], got "
                    f"{list(value.shape)}"
                )

        for env_index, eid in enumerate(env_ids):
            key = (eid, int(obj_id))
            if key not in self.articulations and key not in self.rigids:
                raise KeyError(f"no object registered at env={eid}, obj={obj_id}")
            if cuda_root_reset:
                pos_arr = pos if pos.ndim == 1 else pos[env_index]
                rot_arr = rot_xyzw if rot_xyzw.ndim == 1 else rot_xyzw[env_index]
                linear_velocity_arr = None
                if linear_velocity is not None:
                    linear_velocity_arr = (
                        linear_velocity
                        if linear_velocity.ndim == 1
                        else linear_velocity[env_index]
                    )
                angular_velocity_arr = None
                if angular_velocity is not None:
                    angular_velocity_arr = (
                        angular_velocity
                        if angular_velocity.ndim == 1
                        else angular_velocity[env_index]
                    )
                self.resets[key].root = RootStateReset(
                    pos_arr,
                    rot_arr,
                    linear_velocity_arr,
                    angular_velocity_arr,
                )
                self._pending_reset_keys.add(key)
                continue

            pos_arr = select_env_value(pos, env_index, len(env_ids), (3,))
            rot_arr = select_env_value(rot_xyzw, env_index, len(env_ids), (4,))
            linear_velocity_arr = select_optional_env_value(
                linear_velocity, env_index, len(env_ids), (3,)
            )
            angular_velocity_arr = select_optional_env_value(
                angular_velocity, env_index, len(env_ids), (3,)
            )
            self.resets[key].root = RootStateReset(
                pos_arr.copy(),
                rot_arr.copy(),
                None if linear_velocity_arr is None else linear_velocity_arr.copy(),
                None if angular_velocity_arr is None else angular_velocity_arr.copy(),
            )
            self._pending_reset_keys.add(key)

    def set_dof_state(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        positions,
        velocities=None,
        immediate: bool = False,
    ):
        import sys

        torch = sys.modules.get("torch")
        cuda_positions = (
            torch is not None
            and torch.is_tensor(positions)
            and positions.device.type == "cuda"
        )
        cuda_velocities = (
            torch is not None
            and torch.is_tensor(velocities)
            and velocities.device.type == "cuda"
        )
        if cuda_velocities and not cuda_positions:
            raise RuntimeError(
                "CUDA DOF velocity reset tensors require CUDA position tensors; "
                "KangSimWorld will not silently copy them to CPU"
            )
        if immediate and (cuda_positions or cuda_velocities):
            raise RuntimeError(
                "immediate=True does not support CUDA DOF reset tensors; use the deferred GPU reset path"
            )
        if immediate:
            self.state.set_dof_state(env_id, obj_id, positions, velocities)
            env_ids = env_id_list(env_id, self.num_envs)
            for eid in env_ids:
                self.resets[(eid, int(obj_id))].dof = None
                if not self.resets[(eid, int(obj_id))].pending:
                    self._pending_reset_keys.discard((eid, int(obj_id)))
            return

        env_ids = env_id_list(env_id, self.num_envs)
        if cuda_positions:
            if self._gpu_system is None:
                raise RuntimeError(
                    "CUDA DOF reset tensors require init_gpu_system(); KangSimWorld will not silently copy them to CPU"
                )
            gpu_device = int(self.gpu_system.articulation_joint_positions().device_id)
            pos_device = (
                positions.device.index if positions.device.index is not None else 0
            )
            if pos_device != gpu_device:
                raise ValueError(
                    f"CUDA DOF position tensor is on cuda:{pos_device}, expected cuda:{gpu_device}"
                )
            if positions.dtype != torch.float32:
                raise TypeError(
                    "CUDA DOF position tensor must have dtype=torch.float32"
                )
            if not positions.is_contiguous():
                raise ValueError("CUDA DOF position tensor must be contiguous")
            if velocities is not None:
                if not cuda_velocities:
                    raise RuntimeError(
                        "CUDA DOF position reset tensors require velocities to "
                        "be CUDA tensors or None; KangSimWorld will not silently "
                        "copy mixed reset tensors to CPU"
                    )
                vel_device = (
                    velocities.device.index
                    if velocities.device.index is not None
                    else 0
                )
                if vel_device != gpu_device:
                    raise ValueError(
                        f"CUDA DOF velocity tensor is on cuda:{vel_device}, expected cuda:{gpu_device}"
                    )
                if velocities.dtype != torch.float32:
                    raise TypeError(
                        "CUDA DOF velocity tensor must have dtype=torch.float32"
                    )
                if not velocities.is_contiguous():
                    raise ValueError("CUDA DOF velocity tensor must be contiguous")

        for env_index, eid in enumerate(env_ids):
            key = (eid, int(obj_id))
            if key in self.rigids:
                raise TypeError(
                    f"rigid body at env={eid}, obj={obj_id} does not have DOF state"
                )
            expected = self.articulations[key].articulation.num_dofs()
            if cuda_positions:
                if positions.ndim == 1 and tuple(positions.shape) == (expected,):
                    pos = positions
                elif positions.ndim == 2 and tuple(positions.shape) == (
                    len(env_ids),
                    expected,
                ):
                    pos = positions[env_index]
                else:
                    raise ValueError(
                        "CUDA DOF position tensor expected shape "
                        f"[{expected}] or [{len(env_ids)}, {expected}], got "
                        f"{list(positions.shape)}"
                    )
                vel = None
                if velocities is not None:
                    if velocities.ndim == 1 and tuple(velocities.shape) == (expected,):
                        vel = velocities
                    elif velocities.ndim == 2 and tuple(velocities.shape) == (
                        len(env_ids),
                        expected,
                    ):
                        vel = velocities[env_index]
                    else:
                        raise ValueError(
                            "CUDA DOF velocity tensor expected shape "
                            f"[{expected}] or [{len(env_ids)}, {expected}], got "
                            f"{list(velocities.shape)}"
                        )
                self.resets[key].dof = DofStateReset(pos, vel)
                self._pending_reset_keys.add(key)
                continue

            pos = select_env_value(positions, env_index, len(env_ids), (expected,))
            if pos.size != expected:
                raise ValueError(
                    f"dof reset for env={eid}, obj={obj_id} expected {expected} positions, got {pos.size}"
                )
            vel = None
            if velocities is not None:
                vel = select_env_value(velocities, env_index, len(env_ids), (expected,))
                if vel.size != expected:
                    raise ValueError(
                        f"dof reset for env={eid}, obj={obj_id} expected {expected} velocities, got {vel.size}"
                    )
            self.resets[key].dof = DofStateReset(
                pos.copy(), None if vel is None else vel.copy()
            )
            self._pending_reset_keys.add(key)

    def release(self):
        if self._released:
            return
        for sensor in self.sensors.values():
            sensor.release()
        self.sensors.clear()
        if self._contact_sensor_batch is not None:
            self._contact_sensor_batch.release()
            self._contact_sensor_batch = None
        if self._gpu_system is not None:
            self._gpu_system.invalidate()
            self._gpu_system = None
        self._rigid_gpu_rows.clear()
        self._rigid_gpu_index_tensors.clear()
        self._articulation_gpu_rows.clear()
        self._articulation_gpu_index_tensors.clear()
        self._articulation_gpu_row_tensors.clear()
        self._rigid_gpu_row_tensors.clear()
        self._articulation_gpu_effort_limit_tensors.clear()
        self._validated_cuda_batch_command_obj_ids.clear()
        self._dense_cuda_batch_command_obj_ids.clear()
        self._gpu_root_reset_batches.clear()
        self._gpu_dof_reset_batches.clear()
        self.batch_commands.clear()
        for record in self.articulations.values():
            record.articulation.release()
        for record in self.rigids.values():
            record.rigid.release()
        for record in self.static_rigids.values():
            record.rigid.release()
        self.articulations.clear()
        self.rigids.clear()
        self.static_rigids.clear()
        self.commands.clear()
        self.resets.clear()
        self.body_forces.clear()
        self.body_force_positions.clear()
        if self.state is not None and hasattr(self.state, "release"):
            self.state.release()
        self.physics = None
        self._pending_reset_keys.clear()
        self._active_body_force_keys.clear()
        self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
