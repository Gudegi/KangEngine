"""Canonical Python-side batched state cache for KangEngine simulation objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .rigid import expand_rigid_body_state
from .utils.env_utils import (
    EnvIdLike,
    env_id_list,
    select_env_value,
    select_optional_env_value,
)
from .utils.tensor import as_cpu_numpy, as_tensor, resolve_device


def _empty_state(batch_shape, num_links: int, num_dofs: int, device):
    def empty(*shape):
        return torch.empty((*batch_shape, *shape), dtype=torch.float32, device=device)

    return ArticulationState(
        root_pos=empty(3),
        root_rot=empty(4),
        root_vel=empty(3),
        root_ang_vel=empty(3),
        link_pos=empty(num_links, 3),
        link_rot=empty(num_links, 4),
        link_vel=empty(num_links, 3),
        link_ang_vel=empty(num_links, 3),
        link_contact_force=empty(num_links, 3),
        link_ground_contact_force=empty(num_links, 3),
        dof_pos=empty(num_dofs),
        dof_vel=empty(num_dofs),
        dof_force=empty(num_dofs),
    )


def _state_slice(state, index):
    return ArticulationState(
        root_pos=state.root_pos[index],
        root_rot=state.root_rot[index],
        root_vel=state.root_vel[index],
        root_ang_vel=state.root_ang_vel[index],
        link_pos=state.link_pos[index],
        link_rot=state.link_rot[index],
        link_vel=state.link_vel[index],
        link_ang_vel=state.link_ang_vel[index],
        link_contact_force=state.link_contact_force[index],
        link_ground_contact_force=state.link_ground_contact_force[index],
        dof_pos=state.dof_pos[index],
        dof_vel=state.dof_vel[index],
        dof_force=state.dof_force[index],
    )


@dataclass(slots=True)
class ArticulationState:
    """Tensor views for one PhysX articulation or an env batch.

    Quaternion arrays use xyzw order. Link arrays include the root link at
    index 0, matching ``Articulation`` link indices.
    """

    root_pos: torch.Tensor
    root_rot: torch.Tensor
    root_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    link_pos: torch.Tensor
    link_rot: torch.Tensor
    link_vel: torch.Tensor
    link_ang_vel: torch.Tensor
    link_contact_force: torch.Tensor
    link_ground_contact_force: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    dof_force: torch.Tensor


class ArticulationStateCache:
    """Torch cache around ``ke.Articulation`` flat state getters."""

    def __init__(self, articulation, physics=None, device=None):
        self.articulation = articulation
        self.physics = physics
        self.device = resolve_device(device)
        self.num_links = int(articulation.num_links())
        self.num_dofs = int(articulation.num_dofs())
        self.refresh_metadata()

    def refresh_metadata(self):
        self.dof_names = list(self.articulation.get_dof_names())
        self.dof_limits = as_tensor(
            self.articulation.get_dof_limits(),
            shape=(self.num_dofs, 2),
            device=self.device,
        )
        self.dof_kps = as_tensor(
            self.articulation.get_kps(), shape=(self.num_dofs,), device=self.device
        )
        self.dof_kds = as_tensor(
            self.articulation.get_kds(), shape=(self.num_dofs,), device=self.device
        )
        self.dof_effort_limits = as_tensor(
            self.articulation.get_effort_limits(),
            shape=(self.num_dofs,),
            device=self.device,
        )
        self.link_masses = as_tensor(
            self.articulation.get_link_masses(),
            shape=(self.num_links,),
            device=self.device,
        )

    def refresh_into(self, state: ArticulationState) -> ArticulationState:
        a = self.articulation
        state.root_pos.copy_(as_tensor(a.get_root_position(), shape=(3,), device=self.device))
        state.root_rot.copy_(as_tensor(a.get_root_rotation(), shape=(4,), device=self.device))
        state.root_vel.copy_(
            as_tensor(a.get_root_linear_velocity(), shape=(3,), device=self.device)
        )
        state.root_ang_vel.copy_(
            as_tensor(a.get_root_angular_velocity(), shape=(3,), device=self.device)
        )
        state.link_pos.copy_(
            as_tensor(
                a.get_link_positions(), shape=(self.num_links, 3), device=self.device
            )
        )
        state.link_rot.copy_(
            as_tensor(
                a.get_link_rotations(), shape=(self.num_links, 4), device=self.device
            )
        )
        state.link_vel.copy_(
            as_tensor(
                a.get_link_linear_velocities(),
                shape=(self.num_links, 3),
                device=self.device,
            )
        )
        state.link_ang_vel.copy_(
            as_tensor(
                a.get_link_angular_velocities(),
                shape=(self.num_links, 3),
                device=self.device,
            )
        )
        self._copy_contact_forces(state.link_contact_force, False)
        self._copy_contact_forces(state.link_ground_contact_force, True)
        state.dof_pos.copy_(
            as_tensor(a.get_dof_positions(), shape=(self.num_dofs,), device=self.device)
        )
        state.dof_vel.copy_(
            as_tensor(a.get_dof_velocities(), shape=(self.num_dofs,), device=self.device)
        )
        state.dof_force.copy_(
            as_tensor(a.get_dof_forces(), shape=(self.num_dofs,), device=self.device)
        )
        return state

    def _copy_contact_forces(self, out, ground_only: bool):
        if self.physics is None:
            out.zero_()
            return
        if ground_only:
            values = self.physics.get_ground_contact_forces(self.articulation)
        else:
            values = self.physics.get_contact_forces(self.articulation)
        out.copy_(as_tensor(values, shape=(self.num_links, 3), device=self.device))

    def set_root(self, pos, rot_xyzw, linear_velocity=None, angular_velocity=None):
        linear_velocity = [0.0, 0.0, 0.0] if linear_velocity is None else linear_velocity
        angular_velocity = (
            [0.0, 0.0, 0.0] if angular_velocity is None else angular_velocity
        )
        self.articulation.set_root_state(
            as_cpu_numpy(pos, shape=(3,)),
            as_cpu_numpy(rot_xyzw, shape=(4,)),
            as_cpu_numpy(linear_velocity, shape=(3,)),
            as_cpu_numpy(angular_velocity, shape=(3,)),
        )

    def set_dofs(self, positions, velocities=None):
        if velocities is None:
            velocities = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
        self.articulation.set_dof_state(
            as_cpu_numpy(positions, shape=(self.num_dofs,)),
            as_cpu_numpy(velocities, shape=(self.num_dofs,)),
        )


class RigidStateCache:
    """State cache for one dynamic rigid body.

    Compound rigid bodies expose one body slot per collision shape for
    MimicKit-style body APIs, while PhysX still simulates one rigid actor.
    """

    def __init__(
        self,
        rigid,
        physics=None,
        body_names=None,
        local_pos=None,
        local_rot=None,
        device=None,
    ):
        self.rigid = rigid
        self.physics = physics
        self.device = resolve_device(device)
        self.body_names = list(body_names) if body_names is not None else ["rigid"]
        self.local_pos = (
            np.zeros((len(self.body_names), 3), dtype=np.float32)
            if local_pos is None
            else np.asarray(local_pos, dtype=np.float32).reshape(-1, 3)
        )
        self.local_rot = (
            np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                (len(self.body_names), 1),
            )
            if local_rot is None
            else np.asarray(local_rot, dtype=np.float32).reshape(-1, 4)
        )
        if len(self.body_names) != self.local_pos.shape[0]:
            raise ValueError(
                "rigid body names and local transforms must have matching lengths"
            )
        self.num_links = len(self.body_names)
        self.num_dofs = 0
        self.refresh_metadata()

    def refresh_metadata(self):
        self.dof_names = []
        self.dof_limits = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        self.dof_kps = torch.zeros((0,), dtype=torch.float32, device=self.device)
        self.dof_kds = torch.zeros((0,), dtype=torch.float32, device=self.device)
        self.dof_effort_limits = torch.zeros((0,), dtype=torch.float32, device=self.device)
        self.link_masses = torch.zeros((self.num_links,), dtype=torch.float32, device=self.device)
        if self.num_links:
            self.link_masses[0] = float(self.rigid.get_mass())

    def refresh_into(self, state: ArticulationState) -> ArticulationState:
        pos = as_cpu_numpy(self.rigid.get_root_position(), shape=(3,))
        rot = as_cpu_numpy(self.rigid.get_root_rotation(), shape=(4,))
        vel = as_cpu_numpy(self.rigid.get_root_linear_velocity(), shape=(3,))
        ang_vel = as_cpu_numpy(self.rigid.get_root_angular_velocity(), shape=(3,))
        body_pos, body_rot = expand_rigid_body_state(
            pos, rot, self.local_pos, self.local_rot
        )
        body_vel = np.repeat(vel.reshape(1, 3), self.num_links, axis=0)
        body_ang_vel = np.repeat(ang_vel.reshape(1, 3), self.num_links, axis=0)
        state.root_pos.copy_(as_tensor(pos, device=self.device))
        state.root_rot.copy_(as_tensor(rot, device=self.device))
        state.root_vel.copy_(as_tensor(vel, device=self.device))
        state.root_ang_vel.copy_(as_tensor(ang_vel, device=self.device))
        state.link_pos.copy_(as_tensor(body_pos, device=self.device))
        state.link_rot.copy_(as_tensor(body_rot, device=self.device))
        state.link_vel.copy_(as_tensor(body_vel, device=self.device))
        state.link_ang_vel.copy_(as_tensor(body_ang_vel, device=self.device))
        self._copy_contact_forces(state.link_contact_force, False)
        self._copy_contact_forces(state.link_ground_contact_force, True)
        return state

    def _copy_contact_forces(self, out, ground_only: bool):
        out.zero_()
        if self.physics is None:
            return
        if ground_only:
            values = self.physics.get_rigid_ground_contact_force(self.rigid)
        else:
            values = self.physics.get_rigid_contact_force(self.rigid)
        if self.num_links:
            out[0] = as_tensor(values, shape=(3,), device=self.device)

    def set_root(self, pos, rot_xyzw, linear_velocity=None, angular_velocity=None):
        linear_velocity = [0.0, 0.0, 0.0] if linear_velocity is None else linear_velocity
        angular_velocity = (
            [0.0, 0.0, 0.0] if angular_velocity is None else angular_velocity
        )
        self.rigid.set_root_state(
            as_cpu_numpy(pos, shape=(3,)),
            as_cpu_numpy(rot_xyzw, shape=(4,)),
            as_cpu_numpy(linear_velocity, shape=(3,)),
            as_cpu_numpy(angular_velocity, shape=(3,)),
        )

    def set_dofs(self, positions, velocities=None):
        raise TypeError("rigid bodies do not have DOF state")


@dataclass(slots=True)
class ArticulationRecord:
    env_id: int
    obj_id: int
    articulation: object
    name: str
    cache: ArticulationStateCache | RigidStateCache


class KangStateCache:
    """Canonical runtime state cache keyed by ``(env_id, obj_id)``.

    This is the Python-side bridge toward MimicKit/Newton/Isaac-style APIs:
    object getters return preallocated tensors batched over envs, e.g.
    ``get_root_pos(obj_id) -> [num_envs, 3]``.
    """

    def __init__(self, num_envs: int = 1, device=None):
        self.num_envs = int(num_envs)
        self.device = resolve_device(device)
        self._records: dict[tuple[int, int], ArticulationRecord] = {}
        self._states: dict[int, ArticulationState] = {}
        self._registered_env_ids: dict[int, set[int]] = {}
        self._complete_obj_ids: set[int] = set()

    def _register_record(self, record: ArticulationRecord) -> ArticulationRecord:
        cache = record.cache
        state = self._states.get(record.obj_id)
        if state is None:
            state = _empty_state(
                (self.num_envs,), cache.num_links, cache.num_dofs, self.device
            )
            self._states[record.obj_id] = state
        else:
            expected_links = state.link_pos.shape[1]
            expected_dofs = state.dof_pos.shape[1]
            if cache.num_links != expected_links or cache.num_dofs != expected_dofs:
                raise ValueError(
                    f"object topology mismatch for obj={record.obj_id}: expected "
                    f"{expected_links} links/{expected_dofs} dofs, got "
                    f"{cache.num_links} links/{cache.num_dofs} dofs"
                )
        cache.refresh_into(_state_slice(state, record.env_id))
        env_ids = self._registered_env_ids.setdefault(record.obj_id, set())
        env_ids.add(record.env_id)
        if len(env_ids) == self.num_envs:
            self._complete_obj_ids.add(record.obj_id)
        return record

    def add_articulation(
        self,
        articulation,
        env_id: int = 0,
        obj_id: int = 0,
        name="",
        physics=None,
    ):
        env_id = int(env_id)
        obj_id = int(obj_id)
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} out of range for num_envs={self.num_envs}")
        key = (env_id, obj_id)
        if key in self._records:
            raise ValueError(f"object already registered at env={env_id}, obj={obj_id}")
        cache = ArticulationStateCache(articulation, physics=physics, device=self.device)
        record = ArticulationRecord(env_id, obj_id, articulation, str(name), cache)
        self._register_record(record)
        self._records[key] = record
        return record

    def add_rigid(
        self,
        rigid,
        env_id: int = 0,
        obj_id: int = 0,
        name="",
        physics=None,
        body_names=None,
        local_pos=None,
        local_rot=None,
    ):
        env_id = int(env_id)
        obj_id = int(obj_id)
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} out of range for num_envs={self.num_envs}")
        key = (env_id, obj_id)
        if key in self._records:
            raise ValueError(f"object already registered at env={env_id}, obj={obj_id}")
        cache = RigidStateCache(
            rigid,
            physics=physics,
            body_names=body_names,
            local_pos=local_pos,
            local_rot=local_rot,
            device=self.device,
        )
        record = ArticulationRecord(env_id, obj_id, rigid, str(name), cache)
        self._register_record(record)
        self._records[key] = record
        return record

    def refresh(self):
        for record in self._records.values():
            record.cache.refresh_into(
                _state_slice(self._states[record.obj_id], record.env_id)
            )
        return self

    def record(self, env_id: int, obj_id: int) -> ArticulationRecord:
        key = (int(env_id), int(obj_id))
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"no object registered at env={key[0]}, obj={key[1]}") from exc

    def articulation_state(self, env_id: int, obj_id: int) -> ArticulationState:
        record = self.record(env_id, obj_id)
        return _state_slice(self._states[record.obj_id], record.env_id)

    def _batched_state(self, obj_id: int) -> ArticulationState:
        obj_id = int(obj_id)
        if obj_id not in self._complete_obj_ids:
            registered = self._registered_env_ids.get(obj_id, set())
            missing = [env_id for env_id in range(self.num_envs) if env_id not in registered]
            raise KeyError(f"object obj={obj_id} is missing env registrations: {missing}")
        return self._states[obj_id]

    def get_root_pos(self, obj_id: int):
        return self._batched_state(obj_id).root_pos

    def get_root_rot(self, obj_id: int):
        return self._batched_state(obj_id).root_rot

    def get_root_vel(self, obj_id: int):
        return self._batched_state(obj_id).root_vel

    def get_root_ang_vel(self, obj_id: int):
        return self._batched_state(obj_id).root_ang_vel

    def get_body_pos(self, obj_id: int):
        return self._batched_state(obj_id).link_pos

    def get_body_rot(self, obj_id: int):
        return self._batched_state(obj_id).link_rot

    def get_body_vel(self, obj_id: int):
        return self._batched_state(obj_id).link_vel

    def get_body_ang_vel(self, obj_id: int):
        return self._batched_state(obj_id).link_ang_vel

    def get_contact_forces(self, obj_id: int):
        return self._batched_state(obj_id).link_contact_force

    def get_ground_contact_forces(self, obj_id: int):
        return self._batched_state(obj_id).link_ground_contact_force

    def get_dof_pos(self, obj_id: int):
        return self._batched_state(obj_id).dof_pos

    def get_dof_vel(self, obj_id: int):
        return self._batched_state(obj_id).dof_vel

    def get_dof_forces(self, obj_id: int):
        return self._batched_state(obj_id).dof_force

    def get_obj_num_bodies(self, obj_id: int) -> int:
        return self.record(0, obj_id).cache.num_links

    def get_obj_num_dofs(self, obj_id: int) -> int:
        return self.record(0, obj_id).cache.num_dofs

    def get_obj_dof_names(self, obj_id: int) -> list[str]:
        return list(self.record(0, obj_id).cache.dof_names)

    def get_obj_dof_limits(self, obj_id: int):
        return self.record(0, obj_id).cache.dof_limits.clone()

    def get_obj_pd_gains(self, obj_id: int):
        cache = self.record(0, obj_id).cache
        return cache.dof_kps.clone(), cache.dof_kds.clone()

    def get_obj_effort_limits(self, obj_id: int):
        return self.record(0, obj_id).cache.dof_effort_limits.clone()

    def get_obj_link_masses(self, obj_id: int):
        return self.record(0, obj_id).cache.link_masses.clone()

    def calc_obj_mass(self, env_id: int, obj_id: int) -> float:
        obj = self.record(env_id, obj_id).articulation
        if hasattr(obj, "calc_mass"):
            return float(obj.calc_mass())
        if hasattr(obj, "get_mass"):
            return float(obj.get_mass())
        return 0.0

    def set_root_state(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        pos,
        rot_xyzw,
        linear_velocity=None,
        angular_velocity=None,
    ):
        env_ids = env_id_list(env_id, self.num_envs)
        for env_index, eid in enumerate(env_ids):
            self.record(eid, obj_id).cache.set_root(
                select_env_value(pos, env_index, len(env_ids), (3,)),
                select_env_value(rot_xyzw, env_index, len(env_ids), (4,)),
                select_optional_env_value(
                    linear_velocity, env_index, len(env_ids), (3,)
                ),
                select_optional_env_value(
                    angular_velocity, env_index, len(env_ids), (3,)
                ),
            )

    def set_dof_state(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        positions,
        velocities=None,
    ):
        env_ids = env_id_list(env_id, self.num_envs)
        for env_index, eid in enumerate(env_ids):
            expected = self.record(eid, obj_id).cache.num_dofs
            self.record(eid, obj_id).cache.set_dofs(
                select_env_value(positions, env_index, len(env_ids), (expected,)),
                select_optional_env_value(
                    velocities, env_index, len(env_ids), (expected,)
                ),
            )
