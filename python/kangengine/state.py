"""Python-side world state and snapshots for KangEngine simulation objects."""

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


def _empty_state(batch_shape, num_bodies: int, num_dofs: int, device):
    def empty(*shape):
        return torch.empty((*batch_shape, *shape), dtype=torch.float32, device=device)

    return SimObjectState(
        root_pos=empty(3),
        root_rot=empty(4),
        root_vel=empty(3),
        root_ang_vel=empty(3),
        body_pos=empty(num_bodies, 3),
        body_rot=empty(num_bodies, 4),
        body_vel=empty(num_bodies, 3),
        body_ang_vel=empty(num_bodies, 3),
        body_contact_force=empty(num_bodies, 3),
        body_ground_contact_force=empty(num_bodies, 3),
        dof_pos=empty(num_dofs),
        dof_vel=empty(num_dofs),
        dof_force=empty(num_dofs),
    )


def _state_slice(state, index):
    return SimObjectState(
        root_pos=state.root_pos[index],
        root_rot=state.root_rot[index],
        root_vel=state.root_vel[index],
        root_ang_vel=state.root_ang_vel[index],
        body_pos=state.body_pos[index],
        body_rot=state.body_rot[index],
        body_vel=state.body_vel[index],
        body_ang_vel=state.body_ang_vel[index],
        body_contact_force=state.body_contact_force[index],
        body_ground_contact_force=state.body_ground_contact_force[index],
        dof_pos=state.dof_pos[index],
        dof_vel=state.dof_vel[index],
        dof_force=state.dof_force[index],
    )


@dataclass(slots=True)
class SimObjectState:
    """Tensor views for one simulation object or an env batch.

    Quaternion arrays use xyzw order. Body arrays include the root body at
    index 0.
    """

    root_pos: torch.Tensor
    root_rot: torch.Tensor
    root_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    body_pos: torch.Tensor
    body_rot: torch.Tensor
    body_vel: torch.Tensor
    body_ang_vel: torch.Tensor
    body_contact_force: torch.Tensor
    body_ground_contact_force: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    dof_force: torch.Tensor


class ArticulationStateCache:
    """Torch cache around ``ke.physics.Articulation`` flat state getters."""

    def __init__(self, articulation, physics=None, device=None):
        self.articulation = articulation
        self.physics = physics
        self.device = resolve_device(device)
        self.num_bodies = int(articulation.num_links())
        self.num_dofs = int(articulation.num_dofs())
        self.refresh_metadata()

    def refresh_metadata(self):
        self.dof_names = list(self.articulation.get_dof_names())
        self.dof_gpu_indices = as_tensor(
            self.articulation.get_dof_gpu_indices(),
            shape=(self.num_dofs,),
            device=self.device,
        ).long()
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
        self.body_masses = as_tensor(
            self.articulation.get_link_masses(),
            shape=(self.num_bodies,),
            device=self.device,
        )

    def refresh_into(self, state: SimObjectState) -> SimObjectState:
        a = self.articulation
        state.root_pos.copy_(as_tensor(a.get_root_position(), shape=(3,), device=self.device))
        state.root_rot.copy_(as_tensor(a.get_root_rotation(), shape=(4,), device=self.device))
        state.root_vel.copy_(
            as_tensor(a.get_root_linear_velocity(), shape=(3,), device=self.device)
        )
        state.root_ang_vel.copy_(
            as_tensor(a.get_root_angular_velocity(), shape=(3,), device=self.device)
        )
        state.body_pos.copy_(
            as_tensor(
                a.get_link_positions(), shape=(self.num_bodies, 3), device=self.device
            )
        )
        state.body_rot.copy_(
            as_tensor(
                a.get_link_rotations(), shape=(self.num_bodies, 4), device=self.device
            )
        )
        state.body_vel.copy_(
            as_tensor(
                a.get_link_linear_velocities(),
                shape=(self.num_bodies, 3),
                device=self.device,
            )
        )
        state.body_ang_vel.copy_(
            as_tensor(
                a.get_link_angular_velocities(),
                shape=(self.num_bodies, 3),
                device=self.device,
            )
        )
        self._copy_contact_forces(state.body_contact_force, False)
        self._copy_contact_forces(state.body_ground_contact_force, True)
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
        out.copy_(as_tensor(values, shape=(self.num_bodies, 3), device=self.device))

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
        self.num_bodies = len(self.body_names)
        self.num_dofs = 0
        self.refresh_metadata()

    def refresh_metadata(self):
        self.dof_names = []
        self.dof_limits = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        self.dof_kps = torch.zeros((0,), dtype=torch.float32, device=self.device)
        self.dof_kds = torch.zeros((0,), dtype=torch.float32, device=self.device)
        self.dof_effort_limits = torch.zeros((0,), dtype=torch.float32, device=self.device)
        self.body_masses = torch.zeros((self.num_bodies,), dtype=torch.float32, device=self.device)
        if self.num_bodies:
            self.body_masses[0] = float(self.rigid.get_mass())

    def refresh_into(self, state: SimObjectState) -> SimObjectState:
        pos = as_cpu_numpy(self.rigid.get_root_position(), shape=(3,))
        rot = as_cpu_numpy(self.rigid.get_root_rotation(), shape=(4,))
        vel = as_cpu_numpy(self.rigid.get_root_linear_velocity(), shape=(3,))
        ang_vel = as_cpu_numpy(self.rigid.get_root_angular_velocity(), shape=(3,))
        body_pos, body_rot = expand_rigid_body_state(
            pos, rot, self.local_pos, self.local_rot
        )
        body_vel = np.repeat(vel.reshape(1, 3), self.num_bodies, axis=0)
        body_ang_vel = np.repeat(ang_vel.reshape(1, 3), self.num_bodies, axis=0)
        state.root_pos.copy_(as_tensor(pos, device=self.device))
        state.root_rot.copy_(as_tensor(rot, device=self.device))
        state.root_vel.copy_(as_tensor(vel, device=self.device))
        state.root_ang_vel.copy_(as_tensor(ang_vel, device=self.device))
        state.body_pos.copy_(as_tensor(body_pos, device=self.device))
        state.body_rot.copy_(as_tensor(body_rot, device=self.device))
        state.body_vel.copy_(as_tensor(body_vel, device=self.device))
        state.body_ang_vel.copy_(as_tensor(body_ang_vel, device=self.device))
        self._copy_contact_forces(state.body_contact_force, False)
        self._copy_contact_forces(state.body_ground_contact_force, True)
        return state

    def _copy_contact_forces(self, out, ground_only: bool):
        out.zero_()
        if self.physics is None:
            return
        if ground_only:
            values = self.physics.get_rigid_ground_contact_force(self.rigid)
        else:
            values = self.physics.get_rigid_contact_force(self.rigid)
        if self.num_bodies:
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
class ObjectRecord:
    env_id: int
    obj_id: int
    articulation: object
    name: str
    cache: ArticulationStateCache | RigidStateCache


ArticulationState = SimObjectState
ArticulationRecord = ObjectRecord


class CPUStateBackend:
    """CPU-backed object state table keyed by ``(env_id, obj_id)``.

    CPU simulation uses this cache as canonical runtime state. GPU simulation
    uses the same storage layout for explicit CPU/Torch snapshots.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device=None,
        *,
        canonical_source: str = "cpu",
        snapshot: bool = False,
    ):
        self.num_envs = int(num_envs)
        self.device = resolve_device(device)
        self.canonical_source = str(canonical_source)
        self.is_snapshot = bool(snapshot)
        self.version = 0
        self.stale = False
        self.strict_snapshot_reads = False
        self._records: dict[tuple[int, int], ObjectRecord] = {}
        self._states: dict[int, SimObjectState] = {}
        self._registered_env_ids: dict[int, set[int]] = {}
        self._complete_obj_ids: set[int] = set()

    def set_strict_snapshot_reads(self, enabled: bool = True):
        """Raise on stale GPU snapshot reads when debugging policy data flow."""
        self.strict_snapshot_reads = bool(enabled)
        return self

    def mark_stale(self):
        if self.is_snapshot:
            self.stale = True
        return self

    def mark_fresh(self):
        self.stale = False
        self.version += 1
        return self

    def _check_snapshot_read(self):
        if self.is_snapshot and self.stale and self.strict_snapshot_reads:
            raise RuntimeError(
                "world.state is a stale CPU/Torch snapshot of canonical GPU "
                "state; call world.refresh(), world.step(refresh=True), or use "
                "world.get_gpu_*() for the latest CUDA view"
            )

    def _register_record(self, record: ObjectRecord) -> ObjectRecord:
        cache = record.cache
        state = self._states.get(record.obj_id)
        if state is None:
            state = _empty_state(
                (self.num_envs,), cache.num_bodies, cache.num_dofs, self.device
            )
            self._states[record.obj_id] = state
        else:
            expected_bodies = state.body_pos.shape[1]
            expected_dofs = state.dof_pos.shape[1]
            if cache.num_bodies != expected_bodies or cache.num_dofs != expected_dofs:
                raise ValueError(
                    f"object topology mismatch for obj={record.obj_id}: expected "
                    f"{expected_bodies} bodies/{expected_dofs} dofs, got "
                    f"{cache.num_bodies} bodies/{cache.num_dofs} dofs"
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
        record = ObjectRecord(env_id, obj_id, articulation, str(name), cache)
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
        record = ObjectRecord(env_id, obj_id, rigid, str(name), cache)
        self._register_record(record)
        self._records[key] = record
        return record

    def refresh(self):
        for record in self._records.values():
            record.cache.refresh_into(
                _state_slice(self._states[record.obj_id], record.env_id)
            )
        self.mark_fresh()
        return self

    def record(self, env_id: int, obj_id: int) -> ObjectRecord:
        key = (int(env_id), int(obj_id))
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"no object registered at env={key[0]}, obj={key[1]}") from exc

    def object_state(self, env_id: int, obj_id: int) -> SimObjectState:
        self._check_snapshot_read()
        record = self.record(env_id, obj_id)
        return _state_slice(self._states[record.obj_id], record.env_id)

    def object_states(self, obj_id: int, env_ids: EnvIdLike = None) -> SimObjectState:
        self._check_snapshot_read()
        env_ids = env_id_list(env_ids, self.num_envs)
        for eid in env_ids:
            self.record(eid, obj_id)
        return _state_slice(self._states[int(obj_id)], env_ids)

    def articulation_state(self, env_id: int, obj_id: int) -> SimObjectState:
        return self.object_state(env_id, obj_id)

    def articulation_states(self, obj_id: int, env_ids: EnvIdLike = None) -> SimObjectState:
        return self.object_states(obj_id, env_ids)

    def _batched_state(self, obj_id: int) -> SimObjectState:
        self._check_snapshot_read()
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
        return self._batched_state(obj_id).body_pos

    def get_body_rot(self, obj_id: int):
        return self._batched_state(obj_id).body_rot

    def get_body_vel(self, obj_id: int):
        return self._batched_state(obj_id).body_vel

    def get_body_ang_vel(self, obj_id: int):
        return self._batched_state(obj_id).body_ang_vel

    def get_contact_forces(self, obj_id: int):
        return self._batched_state(obj_id).body_contact_force

    def get_ground_contact_forces(self, obj_id: int):
        return self._batched_state(obj_id).body_ground_contact_force

    def get_dof_pos(self, obj_id: int):
        return self._batched_state(obj_id).dof_pos

    def get_dof_vel(self, obj_id: int):
        return self._batched_state(obj_id).dof_vel

    def get_dof_forces(self, obj_id: int):
        return self._batched_state(obj_id).dof_force

    def get_obj_num_bodies(self, obj_id: int) -> int:
        return self.record(0, obj_id).cache.num_bodies

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

    def get_obj_body_masses(self, obj_id: int):
        return self.record(0, obj_id).cache.body_masses.clone()

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

    def release(self):
        self._records.clear()
        self._states.clear()
        self._registered_env_ids.clear()
        self._complete_obj_ids.clear()
        return self


class StateSnapshotBackend(CPUStateBackend):
    """CPU/Torch readback snapshot backend for GPU simulation."""

    def __init__(
        self,
        num_envs: int = 1,
        device=None,
        *,
        canonical_source: str = "cpu",
        snapshot: bool = False,
        gpu_system_provider=None,
    ):
        super().__init__(
            num_envs,
            device,
            canonical_source=canonical_source,
            snapshot=snapshot,
        )
        self._gpu_system_provider = gpu_system_provider

    def refresh(self):
        if self.canonical_source != "gpu" or self._gpu_system_provider is None:
            return super().refresh()
        gpu_system = self._gpu_system_provider()
        if gpu_system is None:
            return super().refresh()

        has_articulations = any(
            isinstance(record.cache, ArticulationStateCache)
            for record in self._records.values()
        )
        has_rigids = any(
            isinstance(record.cache, RigidStateCache)
            for record in self._records.values()
        )

        articulation_link_data = None
        articulation_qpos = None
        articulation_qvel = None
        articulation_qf = None
        if has_articulations:
            gpu_system.fetch_articulation_link_pose()
            gpu_system.fetch_articulation_link_vel()
            gpu_system.fetch_articulation_joint_positions()
            gpu_system.fetch_articulation_joint_velocities()
            gpu_system.fetch_articulation_joint_forces()
            articulation_link_data = gpu_system.articulation_link_data().torch()
            articulation_qpos = gpu_system.articulation_joint_positions().torch()
            articulation_qvel = gpu_system.articulation_joint_velocities().torch()
            articulation_qf = gpu_system.articulation_joint_forces().torch()

        rigid_data = None
        if has_rigids:
            gpu_system.fetch_rigid_data()
            rigid_data = gpu_system.rigid_data().torch()

        for record in self._records.values():
            state = _state_slice(self._states[record.obj_id], record.env_id)
            cache = record.cache
            if isinstance(cache, ArticulationStateCache):
                row = int(gpu_system.articulation_row(cache.articulation))
                num_bodies = int(cache.num_bodies)
                num_dofs = int(cache.num_dofs)
                link_indices = torch.as_tensor(
                    cache.articulation.get_link_indices(),
                    dtype=torch.long,
                    device=articulation_link_data.device,
                )
                link_state = articulation_link_data[row, link_indices[:num_bodies]]
                state.root_pos.copy_(link_state[0, 0:3].to(self.device))
                state.root_rot.copy_(link_state[0, 3:7].to(self.device))
                state.root_vel.copy_(link_state[0, 7:10].to(self.device))
                state.root_ang_vel.copy_(link_state[0, 10:13].to(self.device))
                state.body_pos.copy_(link_state[:, 0:3].to(self.device))
                state.body_rot.copy_(link_state[:, 3:7].to(self.device))
                state.body_vel.copy_(link_state[:, 7:10].to(self.device))
                state.body_ang_vel.copy_(link_state[:, 10:13].to(self.device))
                dof_indices = cache.dof_gpu_indices.to(articulation_qpos.device)
                state.dof_pos.copy_(articulation_qpos[row, dof_indices].to(self.device))
                state.dof_vel.copy_(articulation_qvel[row, dof_indices].to(self.device))
                state.dof_force.copy_(articulation_qf[row, dof_indices].to(self.device))
                state.body_contact_force.zero_()
                state.body_ground_contact_force.zero_()
                continue

            if isinstance(cache, RigidStateCache):
                row = int(gpu_system.rigid_row(cache.rigid))
                rigid_state = rigid_data[row]
                state.root_pos.copy_(rigid_state[0:3].to(self.device))
                state.root_rot.copy_(rigid_state[3:7].to(self.device))
                state.root_vel.copy_(rigid_state[7:10].to(self.device))
                state.root_ang_vel.copy_(rigid_state[10:13].to(self.device))
                state.body_pos.copy_(
                    rigid_state[0:3].to(self.device).reshape(1, 3).expand_as(
                        state.body_pos
                    )
                )
                state.body_rot.copy_(
                    rigid_state[3:7].to(self.device).reshape(1, 4).expand_as(
                        state.body_rot
                    )
                )
                state.body_vel.copy_(
                    rigid_state[7:10].to(self.device).reshape(1, 3).expand_as(
                        state.body_vel
                    )
                )
                state.body_ang_vel.copy_(
                    rigid_state[10:13].to(self.device).reshape(1, 3).expand_as(
                        state.body_ang_vel
                    )
                )
                state.body_contact_force.zero_()
                state.body_ground_contact_force.zero_()

        self.mark_fresh()
        return self


class GPUStateBackend:
    """Canonical PhysX GPU state with optional logical CUDA views."""

    def __init__(self, num_envs: int = 1, device=None, gpu_system_provider=None):
        self.num_envs = int(num_envs)
        self.device = resolve_device(device)
        self._gpu_system_provider = gpu_system_provider
        self._metadata_backend = None
        self._row_index_tensors = {}
        self._link_index_tensors = {}
        self._dof_index_tensors = {}
        self._frame_cache = {}

    def set_metadata_backend(self, backend):
        self._metadata_backend = backend
        self._row_index_tensors.clear()
        self._link_index_tensors.clear()
        self._dof_index_tensors.clear()
        self.clear_frame_cache()
        return self

    def clear_frame_cache(self):
        self._frame_cache.clear()

    def refresh_frame_cache(
        self,
        *,
        articulation_pose: bool = True,
        articulation_velocity: bool = True,
        dof_pos: bool = True,
        dof_vel: bool = True,
        dof_force: bool = False,
        rigid_data: bool = False,
    ):
        """Fetch GPU state once and reuse tensor views for the current frame."""
        gpu_system = self._require_gpu_system()
        self.clear_frame_cache()
        if articulation_pose:
            gpu_system.fetch_articulation_link_pose()
        if articulation_velocity:
            gpu_system.fetch_articulation_link_vel()
        if articulation_pose or articulation_velocity:
            self._frame_cache["articulation_link_data"] = (
                gpu_system.articulation_link_data().torch()
            )
        if dof_pos:
            gpu_system.fetch_articulation_joint_positions()
            self._frame_cache["articulation_joint_positions"] = (
                gpu_system.articulation_joint_positions().torch()
            )
        if dof_vel:
            gpu_system.fetch_articulation_joint_velocities()
            self._frame_cache["articulation_joint_velocities"] = (
                gpu_system.articulation_joint_velocities().torch()
            )
        if dof_force:
            gpu_system.fetch_articulation_joint_forces()
            self._frame_cache["articulation_joint_forces"] = (
                gpu_system.articulation_joint_forces().torch()
            )
        if rigid_data:
            gpu_system.fetch_rigid_data()
            self._frame_cache["rigid_data"] = gpu_system.rigid_data().torch()
        return self

    @property
    def gpu_system(self):
        if self._gpu_system_provider is None:
            return None
        return self._gpu_system_provider()

    def _require_gpu_system(self):
        gpu_system = self.gpu_system
        if gpu_system is None:
            raise RuntimeError("GPU state backend is not attached to PhysicsGpuSystem")
        return gpu_system

    def _require_metadata_backend(self):
        if self._metadata_backend is None:
            raise RuntimeError("GPU state backend has no object metadata backend")
        return self._metadata_backend

    def _records_for_obj(self, obj_id: int):
        backend = self._require_metadata_backend()
        obj_id = int(obj_id)
        if obj_id not in backend._complete_obj_ids:
            registered = backend._registered_env_ids.get(obj_id, set())
            missing = [env_id for env_id in range(self.num_envs) if env_id not in registered]
            raise KeyError(f"object obj={obj_id} is missing env registrations: {missing}")
        return [backend.record(env_id, obj_id) for env_id in range(self.num_envs)]

    def _object_kind(self, obj_id: int):
        record = self._records_for_obj(obj_id)[0]
        if isinstance(record.cache, ArticulationStateCache):
            return "articulation"
        if isinstance(record.cache, RigidStateCache):
            return "rigid"
        raise TypeError(f"unsupported object cache type for obj={obj_id}")

    def _row_indices(self, obj_id: int, kind: str, *, device):
        key = (kind, int(obj_id), str(device))
        tensor = self._row_index_tensors.get(key)
        if tensor is not None:
            return tensor
        gpu_system = self._require_gpu_system()
        records = self._records_for_obj(obj_id)
        if kind == "articulation":
            rows = [int(gpu_system.articulation_row(r.cache.articulation)) for r in records]
        elif kind == "rigid":
            rows = [int(gpu_system.rigid_row(r.cache.rigid)) for r in records]
        else:
            raise TypeError(f"unsupported GPU state object kind: {kind}")
        tensor = torch.tensor(rows, dtype=torch.long, device=device)
        self._row_index_tensors[key] = tensor
        return tensor

    def _articulation_link_indices(self, obj_id: int, *, device):
        key = (int(obj_id), str(device))
        tensor = self._link_index_tensors.get(key)
        if tensor is not None:
            return tensor
        records = self._records_for_obj(obj_id)
        first = list(records[0].cache.articulation.get_link_indices())
        for record in records[1:]:
            other = list(record.cache.articulation.get_link_indices())
            if other != first:
                raise RuntimeError(
                    f"articulation obj={obj_id} has inconsistent link index maps"
                )
        tensor = torch.tensor(first, dtype=torch.long, device=device)
        self._link_index_tensors[key] = tensor
        return tensor

    def _articulation_dof_indices(self, obj_id: int, *, device):
        key = (int(obj_id), str(device))
        tensor = self._dof_index_tensors.get(key)
        if tensor is not None:
            return tensor
        records = self._records_for_obj(obj_id)
        first = records[0].cache.dof_gpu_indices.to(device)
        for record in records[1:]:
            other = record.cache.dof_gpu_indices.to(device)
            if not torch.equal(other, first):
                raise RuntimeError(
                    f"articulation obj={obj_id} has inconsistent DOF GPU maps"
                )
        self._dof_index_tensors[key] = first
        return first

    def _articulation_link_tensor(self, obj_id: int, *, fetch_pose=True, fetch_velocity=True):
        gpu_system = self._require_gpu_system()
        cache_key = ("articulation_link", int(obj_id))
        if not fetch_pose and not fetch_velocity and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]
        if fetch_pose:
            gpu_system.fetch_articulation_link_pose()
        if fetch_velocity:
            gpu_system.fetch_articulation_link_vel()
        raw = self._frame_cache.get("articulation_link_data")
        if raw is None or fetch_pose or fetch_velocity:
            raw = gpu_system.articulation_link_data().torch()
            if not fetch_pose and not fetch_velocity:
                self._frame_cache["articulation_link_data"] = raw
        rows = self._row_indices(obj_id, "articulation", device=raw.device)
        links = self._articulation_link_indices(obj_id, device=raw.device)
        value = raw[rows][:, links]
        if not fetch_pose and not fetch_velocity:
            self._frame_cache[cache_key] = value
        return value

    def _articulation_dof_tensor(self, obj_id: int, source, *, fetch=True, cache_name=None):
        cache_key = None if cache_name is None else (cache_name, int(obj_id))
        if not fetch and cache_key is not None and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]
        raw = None if cache_name is None else self._frame_cache.get(cache_name)
        if raw is None or fetch:
            raw = source(fetch=fetch).torch()
            if not fetch and cache_name is not None:
                self._frame_cache[cache_name] = raw
        rows = self._row_indices(obj_id, "articulation", device=raw.device)
        dofs = self._articulation_dof_indices(obj_id, device=raw.device)
        value = raw[rows][:, dofs]
        if not fetch and cache_key is not None:
            self._frame_cache[cache_key] = value
        return value

    def _rigid_tensor(self, obj_id: int, *, fetch=True):
        cache_key = ("rigid", int(obj_id))
        if not fetch and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]
        raw = self.rigid_data_tensor(fetch=fetch)
        rows = self._row_indices(obj_id, "rigid", device=raw.device)
        value = raw[rows]
        if not fetch:
            self._frame_cache[cache_key] = value
        return value

    def get_root_pos(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=fetch, fetch_velocity=False)[:, 0, 0:3]
        return self._rigid_tensor(obj_id, fetch=fetch)[:, 0:3]

    def get_root_rot(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=fetch, fetch_velocity=False)[:, 0, 3:7]
        return self._rigid_tensor(obj_id, fetch=fetch)[:, 3:7]

    def get_root_vel(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=False, fetch_velocity=fetch)[:, 0, 7:10]
        return self._rigid_tensor(obj_id, fetch=fetch)[:, 7:10]

    def get_root_ang_vel(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=False, fetch_velocity=fetch)[:, 0, 10:13]
        return self._rigid_tensor(obj_id, fetch=fetch)[:, 10:13]

    def get_body_pos(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=fetch, fetch_velocity=False)[:, :, 0:3]
        rigid = self._rigid_tensor(obj_id, fetch=fetch)
        num_bodies = self._records_for_obj(obj_id)[0].cache.num_bodies
        return rigid[:, 0:3].reshape(self.num_envs, 1, 3).expand(-1, num_bodies, -1)

    def get_body_rot(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=fetch, fetch_velocity=False)[:, :, 3:7]
        rigid = self._rigid_tensor(obj_id, fetch=fetch)
        num_bodies = self._records_for_obj(obj_id)[0].cache.num_bodies
        return rigid[:, 3:7].reshape(self.num_envs, 1, 4).expand(-1, num_bodies, -1)

    def get_body_vel(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=False, fetch_velocity=fetch)[:, :, 7:10]
        rigid = self._rigid_tensor(obj_id, fetch=fetch)
        num_bodies = self._records_for_obj(obj_id)[0].cache.num_bodies
        return rigid[:, 7:10].reshape(self.num_envs, 1, 3).expand(-1, num_bodies, -1)

    def get_body_ang_vel(self, obj_id: int, *, fetch: bool = True):
        kind = self._object_kind(obj_id)
        if kind == "articulation":
            return self._articulation_link_tensor(obj_id, fetch_pose=False, fetch_velocity=fetch)[:, :, 10:13]
        rigid = self._rigid_tensor(obj_id, fetch=fetch)
        num_bodies = self._records_for_obj(obj_id)[0].cache.num_bodies
        return rigid[:, 10:13].reshape(self.num_envs, 1, 3).expand(-1, num_bodies, -1)

    def get_dof_pos(self, obj_id: int, *, fetch: bool = True):
        if self._object_kind(obj_id) != "articulation":
            return torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)
        return self._articulation_dof_tensor(
            obj_id,
            self.articulation_joint_positions,
            fetch=fetch,
            cache_name="articulation_joint_positions",
        )

    def get_dof_vel(self, obj_id: int, *, fetch: bool = True):
        if self._object_kind(obj_id) != "articulation":
            return torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)
        return self._articulation_dof_tensor(
            obj_id,
            self.articulation_joint_velocities,
            fetch=fetch,
            cache_name="articulation_joint_velocities",
        )

    def get_dof_forces(self, obj_id: int, *, fetch: bool = True):
        if self._object_kind(obj_id) != "articulation":
            return torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)
        return self._articulation_dof_tensor(
            obj_id,
            self.articulation_joint_forces,
            fetch=fetch,
            cache_name="articulation_joint_forces",
        )

    def rigid_data(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_rigid_data()
        return gpu_system.rigid_data()

    def rigid_data_tensor(self, *, fetch: bool = True):
        raw = self._frame_cache.get("rigid_data")
        if raw is not None and not fetch:
            return raw
        raw = self.rigid_data(fetch=fetch).torch()
        if not fetch:
            self._frame_cache["rigid_data"] = raw
        return raw

    def articulation_link_data(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_link_pose()
        return gpu_system.articulation_link_data()

    def articulation_link_data_tensor(self, *, fetch: bool = True):
        return self.articulation_link_data(fetch=fetch).torch()

    def articulation_joint_positions(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_joint_positions()
        return gpu_system.articulation_joint_positions()

    def articulation_joint_positions_tensor(self, *, fetch: bool = True):
        return self.articulation_joint_positions(fetch=fetch).torch()

    def articulation_joint_velocities(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_joint_velocities()
        return gpu_system.articulation_joint_velocities()

    def articulation_joint_velocities_tensor(self, *, fetch: bool = True):
        return self.articulation_joint_velocities(fetch=fetch).torch()

    def articulation_joint_accelerations(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_joint_accelerations()
        return gpu_system.articulation_joint_accelerations()

    def articulation_joint_accelerations_tensor(self, *, fetch: bool = True):
        return self.articulation_joint_accelerations(fetch=fetch).torch()

    def articulation_joint_forces(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_joint_forces()
        return gpu_system.articulation_joint_forces()

    def articulation_joint_forces_tensor(self, *, fetch: bool = True):
        return self.articulation_joint_forces(fetch=fetch).torch()

    def articulation_target_joint_positions(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_target_joint_positions()
        return gpu_system.articulation_target_joint_positions()

    def articulation_target_joint_positions_tensor(self, *, fetch: bool = True):
        return self.articulation_target_joint_positions(fetch=fetch).torch()

    def articulation_target_joint_velocities(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_target_joint_velocities()
        return gpu_system.articulation_target_joint_velocities()

    def articulation_target_joint_velocities_tensor(self, *, fetch: bool = True):
        return self.articulation_target_joint_velocities(fetch=fetch).torch()

    def articulation_link_incoming_joint_forces(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_articulation_link_incoming_joint_force()
        return gpu_system.articulation_link_incoming_joint_forces()

    def articulation_link_incoming_joint_forces_tensor(self, *, fetch: bool = True):
        return self.articulation_link_incoming_joint_forces(fetch=fetch).torch()

    def contact_pairs(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_contact_pairs()
        return gpu_system.contact_pairs()

    def contact_pairs_tensor(self, *, fetch: bool = True):
        return self.contact_pairs(fetch=fetch).torch()

    def contact_pair_count(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_contact_pairs()
        return gpu_system.contact_pair_count()

    def contact_pair_count_tensor(self, *, fetch: bool = True):
        return self.contact_pair_count(fetch=fetch).torch()

    def contact_pair_headers(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_contact_pairs()
        return gpu_system.contact_pair_headers()

    def contact_pair_headers_tensor(self, *, fetch: bool = True):
        return self.contact_pair_headers(fetch=fetch).torch()

    def contact_pair_body_refs(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_contact_pairs()
        return gpu_system.contact_pair_body_refs()

    def contact_pair_body_refs_tensor(self, *, fetch: bool = True):
        return self.contact_pair_body_refs(fetch=fetch).torch()

    def contact_points(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_contact_pairs()
        return gpu_system.contact_points()

    def contact_points_tensor(self, *, fetch: bool = True):
        return self.contact_points(fetch=fetch).torch()

    def contact_point_count(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_contact_pairs()
        return gpu_system.contact_point_count()

    def contact_point_count_tensor(self, *, fetch: bool = True):
        return self.contact_point_count(fetch=fetch).torch()

    def contact_point_pair_indices(self, *, fetch: bool = True):
        gpu_system = self._require_gpu_system()
        if fetch:
            gpu_system.fetch_contact_pairs()
        return gpu_system.contact_point_pair_indices()

    def contact_point_pair_indices_tensor(self, *, fetch: bool = True):
        return self.contact_point_pair_indices(fetch=fetch).torch()

    def release(self):
        self._gpu_system_provider = None
        return self


class KangWorldState:
    """Public world state handle.

    CPU simulation delegates reads to ``CPUStateBackend``. GPU simulation keeps
    canonical state in ``GPUStateBackend`` and delegates ``get_*`` reads to the
    explicit ``StateSnapshotBackend`` updated by ``refresh()``.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device=None,
        *,
        canonical_source: str = "cpu",
        snapshot: bool = False,
        gpu_system_provider=None,
    ):
        self.num_envs = int(num_envs)
        self.device = resolve_device(device)
        self.canonical_source = str(canonical_source)
        self.is_snapshot = bool(snapshot)
        self.version = 0
        self.stale = False
        self.strict_snapshot_reads = False
        if self.canonical_source == "gpu":
            self.backend = GPUStateBackend(
                self.num_envs, self.device, gpu_system_provider
            )
            # TODO: Remove this compatibility snapshot, use only GPU-direct state access.
            self.snapshot = StateSnapshotBackend(
                self.num_envs,
                self.device,
                canonical_source="gpu",
                snapshot=True,
                gpu_system_provider=gpu_system_provider,
            )
            self.backend.set_metadata_backend(self.snapshot)
        else:
            self.backend = CPUStateBackend(
                self.num_envs,
                self.device,
                canonical_source="cpu",
                snapshot=False,
            )
            self.snapshot = self.backend

    @property
    def gpu(self):
        if self.canonical_source != "gpu":
            raise RuntimeError("world.state.gpu is only available for GPU simulation")
        return self.backend

    def set_strict_snapshot_reads(self, enabled: bool = True):
        """Raise on stale GPU snapshot reads when debugging policy data flow."""
        self.strict_snapshot_reads = bool(enabled)
        return self

    def mark_stale(self):
        if self.is_snapshot:
            self.stale = True
            if hasattr(self.backend, "clear_frame_cache"):
                self.backend.clear_frame_cache()
            if hasattr(self.snapshot, "mark_stale"):
                self.snapshot.mark_stale()
        return self

    def mark_fresh(self):
        self.stale = False
        self.version += 1
        return self

    def _check_snapshot_read(self):
        if self.is_snapshot and self.stale and self.strict_snapshot_reads:
            raise RuntimeError(
                "world.state is a stale CPU/Torch snapshot of canonical GPU "
                "state; call world.refresh(), world.step(refresh=True), or use "
                "world.get_gpu_*() for the latest CUDA view"
            )

    def _read_backend(self):
        self._check_snapshot_read()
        return self.snapshot

    def add_articulation(self, *args, **kwargs):
        return self.snapshot.add_articulation(*args, **kwargs)

    def add_rigid(self, *args, **kwargs):
        return self.snapshot.add_rigid(*args, **kwargs)

    def refresh(self):
        self.snapshot.refresh()
        self.mark_fresh()
        return self

    def record(self, *args, **kwargs):
        return self.snapshot.record(*args, **kwargs)

    def object_state(self, *args, **kwargs):
        return self._read_backend().object_state(*args, **kwargs)

    def object_states(self, *args, **kwargs):
        return self._read_backend().object_states(*args, **kwargs)

    def articulation_state(self, *args, **kwargs):
        return self.object_state(*args, **kwargs)

    def articulation_states(self, *args, **kwargs):
        return self.object_states(*args, **kwargs)

    def get_root_pos(self, obj_id: int):
        return self._read_backend().get_root_pos(obj_id)

    def get_root_rot(self, obj_id: int):
        return self._read_backend().get_root_rot(obj_id)

    def get_root_vel(self, obj_id: int):
        return self._read_backend().get_root_vel(obj_id)

    def get_root_ang_vel(self, obj_id: int):
        return self._read_backend().get_root_ang_vel(obj_id)

    def get_body_pos(self, obj_id: int):
        return self._read_backend().get_body_pos(obj_id)

    def get_body_rot(self, obj_id: int):
        return self._read_backend().get_body_rot(obj_id)

    def get_body_vel(self, obj_id: int):
        return self._read_backend().get_body_vel(obj_id)

    def get_body_ang_vel(self, obj_id: int):
        return self._read_backend().get_body_ang_vel(obj_id)

    def get_contact_forces(self, obj_id: int):
        return self._read_backend().get_contact_forces(obj_id)

    def get_ground_contact_forces(self, obj_id: int):
        return self._read_backend().get_ground_contact_forces(obj_id)

    def get_dof_pos(self, obj_id: int):
        return self._read_backend().get_dof_pos(obj_id)

    def get_dof_vel(self, obj_id: int):
        return self._read_backend().get_dof_vel(obj_id)

    def get_dof_forces(self, obj_id: int):
        return self._read_backend().get_dof_forces(obj_id)

    def get_obj_num_bodies(self, obj_id: int) -> int:
        return self.snapshot.get_obj_num_bodies(obj_id)

    def get_obj_num_dofs(self, obj_id: int) -> int:
        return self.snapshot.get_obj_num_dofs(obj_id)

    def get_obj_dof_names(self, obj_id: int) -> list[str]:
        return self.snapshot.get_obj_dof_names(obj_id)

    def get_obj_dof_limits(self, obj_id: int):
        return self.snapshot.get_obj_dof_limits(obj_id)

    def get_obj_pd_gains(self, obj_id: int):
        return self.snapshot.get_obj_pd_gains(obj_id)

    def get_obj_effort_limits(self, obj_id: int):
        return self.snapshot.get_obj_effort_limits(obj_id)

    def get_obj_body_masses(self, obj_id: int):
        return self.snapshot.get_obj_body_masses(obj_id)

    def calc_obj_mass(self, *args, **kwargs):
        return self.snapshot.calc_obj_mass(*args, **kwargs)

    def set_root_state(self, *args, **kwargs):
        return self.snapshot.set_root_state(*args, **kwargs)

    def set_dof_state(self, *args, **kwargs):
        return self.snapshot.set_dof_state(*args, **kwargs)

    def release(self):
        if self.snapshot is not self.backend and hasattr(self.snapshot, "release"):
            self.snapshot.release()
        if hasattr(self.backend, "release"):
            self.backend.release()
        self.snapshot = None
        self.backend = None
        return self
