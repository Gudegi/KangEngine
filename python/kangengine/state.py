"""Python state helpers for KangEngine simulation objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rigid import expand_rigid_body_state


def _array(values, shape):
    return np.asarray(values, dtype=np.float32).reshape(shape)


@dataclass(slots=True)
class ArticulationState:
    """Snapshot of one PhysX articulation.

    Quaternion arrays use xyzw order. Link arrays include the root link at
    index 0, matching ``Articulation`` link indices.
    """

    root_pos: np.ndarray
    root_rot: np.ndarray
    root_vel: np.ndarray
    root_ang_vel: np.ndarray
    link_pos: np.ndarray
    link_rot: np.ndarray
    link_vel: np.ndarray
    link_ang_vel: np.ndarray
    link_contact_force: np.ndarray
    link_ground_contact_force: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    dof_force: np.ndarray


class ArticulationStateCache:
    """Numpy-shaped cache around ``ke.Articulation`` flat state getters."""

    def __init__(self, articulation, physics=None):
        self.articulation = articulation
        self.physics = physics
        self.num_links = int(articulation.num_links())
        self.num_dofs = int(articulation.num_dofs())
        self.refresh_metadata()
        self.state: ArticulationState | None = None

    def refresh_metadata(self):
        self.dof_names = list(self.articulation.get_dof_names())
        self.dof_limits = _array(
            self.articulation.get_dof_limits(), (self.num_dofs, 2)
        )
        self.dof_kps = _array(self.articulation.get_kps(), (self.num_dofs,))
        self.dof_kds = _array(self.articulation.get_kds(), (self.num_dofs,))
        self.dof_effort_limits = _array(
            self.articulation.get_effort_limits(), (self.num_dofs,)
        )
        self.link_masses = _array(
            self.articulation.get_link_masses(), (self.num_links,)
        )

    def refresh(self) -> ArticulationState:
        a = self.articulation
        state = ArticulationState(
            root_pos=_array(a.get_root_position(), (3,)),
            root_rot=_array(a.get_root_rotation(), (4,)),
            root_vel=_array(a.get_root_linear_velocity(), (3,)),
            root_ang_vel=_array(a.get_root_angular_velocity(), (3,)),
            link_pos=_array(a.get_link_positions(), (self.num_links, 3)),
            link_rot=_array(a.get_link_rotations(), (self.num_links, 4)),
            link_vel=_array(a.get_link_linear_velocities(), (self.num_links, 3)),
            link_ang_vel=_array(
                a.get_link_angular_velocities(), (self.num_links, 3)
            ),
            link_contact_force=self._contact_forces(False),
            link_ground_contact_force=self._contact_forces(True),
            dof_pos=_array(a.get_dof_positions(), (self.num_dofs,)),
            dof_vel=_array(a.get_dof_velocities(), (self.num_dofs,)),
            dof_force=_array(a.get_dof_forces(), (self.num_dofs,)),
        )
        self.state = state
        return state

    def _contact_forces(self, ground_only: bool):
        if self.physics is None:
            return np.zeros((self.num_links, 3), dtype=np.float32)
        if ground_only:
            values = self.physics.get_ground_contact_forces(self.articulation)
        else:
            values = self.physics.get_contact_forces(self.articulation)
        return _array(values, (self.num_links, 3))

    def set_root(self, pos, rot_xyzw, linear_velocity=None, angular_velocity=None):
        linear_velocity = [0.0, 0.0, 0.0] if linear_velocity is None else linear_velocity
        angular_velocity = (
            [0.0, 0.0, 0.0] if angular_velocity is None else angular_velocity
        )
        self.articulation.set_root_state(
            list(pos), list(rot_xyzw), list(linear_velocity), list(angular_velocity)
        )

    def set_dofs(self, positions, velocities=None):
        if velocities is None:
            velocities = np.zeros(self.num_dofs, dtype=np.float32)
        self.articulation.set_dof_state(list(positions), list(velocities))


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
    ):
        self.rigid = rigid
        self.physics = physics
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
        self.state: ArticulationState | None = None

    def refresh_metadata(self):
        self.dof_names = []
        self.dof_limits = np.zeros((0, 2), dtype=np.float32)
        self.dof_kps = np.zeros((0,), dtype=np.float32)
        self.dof_kds = np.zeros((0,), dtype=np.float32)
        self.dof_effort_limits = np.zeros((0,), dtype=np.float32)
        self.link_masses = np.zeros((self.num_links,), dtype=np.float32)
        if self.num_links:
            self.link_masses[0] = float(self.rigid.get_mass())

    def refresh(self) -> ArticulationState:
        pos = _array(self.rigid.get_root_position(), (3,))
        rot = _array(self.rigid.get_root_rotation(), (4,))
        vel = _array(self.rigid.get_root_linear_velocity(), (3,))
        ang_vel = _array(self.rigid.get_root_angular_velocity(), (3,))
        body_pos, body_rot = expand_rigid_body_state(
            pos, rot, self.local_pos, self.local_rot
        )
        body_vel = np.repeat(vel.reshape(1, 3), self.num_links, axis=0)
        body_ang_vel = np.repeat(ang_vel.reshape(1, 3), self.num_links, axis=0)
        state = ArticulationState(
            root_pos=pos,
            root_rot=rot,
            root_vel=vel,
            root_ang_vel=ang_vel,
            link_pos=body_pos,
            link_rot=body_rot,
            link_vel=body_vel,
            link_ang_vel=body_ang_vel,
            link_contact_force=self._contact_forces(False),
            link_ground_contact_force=self._contact_forces(True),
            dof_pos=np.zeros((0,), dtype=np.float32),
            dof_vel=np.zeros((0,), dtype=np.float32),
            dof_force=np.zeros((0,), dtype=np.float32),
        )
        self.state = state
        return state

    def _contact_forces(self, ground_only: bool):
        if self.physics is None:
            return np.zeros((self.num_links, 3), dtype=np.float32)
        if ground_only:
            values = self.physics.get_rigid_ground_contact_force(self.rigid)
        else:
            values = self.physics.get_rigid_contact_force(self.rigid)
        out = np.zeros((self.num_links, 3), dtype=np.float32)
        if self.num_links:
            out[0] = _array(values, (3,))
        return out

    def set_root(self, pos, rot_xyzw, linear_velocity=None, angular_velocity=None):
        linear_velocity = [0.0, 0.0, 0.0] if linear_velocity is None else linear_velocity
        angular_velocity = (
            [0.0, 0.0, 0.0] if angular_velocity is None else angular_velocity
        )
        self.rigid.set_root_state(
            list(pos), list(rot_xyzw), list(linear_velocity), list(angular_velocity)
        )

    def set_dofs(self, positions, velocities=None):
        return


@dataclass(slots=True)
class ArticulationRecord:
    env_id: int
    obj_id: int
    articulation: object
    name: str
    cache: ArticulationStateCache


class KangStateCache:
    """Batched state cache keyed by ``(env_id, obj_id)``.

    This is the Python-side bridge toward MimicKit/Newton/Isaac-style APIs:
    object getters return arrays stacked over envs, e.g.
    ``get_root_pos(obj_id) -> [num_envs, 3]``.
    """

    def __init__(self, num_envs: int = 1):
        self.num_envs = int(num_envs)
        self._records: dict[tuple[int, int], ArticulationRecord] = {}

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
        cache = ArticulationStateCache(articulation, physics=physics)
        record = ArticulationRecord(env_id, obj_id, articulation, str(name), cache)
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
        )
        record = ArticulationRecord(env_id, obj_id, rigid, str(name), cache)
        self._records[key] = record
        return record

    def refresh(self):
        for record in self._records.values():
            record.cache.refresh()
        return self

    def record(self, env_id: int, obj_id: int) -> ArticulationRecord:
        key = (int(env_id), int(obj_id))
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"no object registered at env={key[0]}, obj={key[1]}") from exc

    def articulation_state(self, env_id: int, obj_id: int) -> ArticulationState:
        cache = self.record(env_id, obj_id).cache
        return cache.state if cache.state is not None else cache.refresh()

    def _env_records(self, obj_id: int) -> list[ArticulationRecord]:
        obj_id = int(obj_id)
        records = []
        for env_id in range(self.num_envs):
            records.append(self.record(env_id, obj_id))
        return records

    def _stack(self, obj_id: int, attr: str):
        values = []
        for record in self._env_records(obj_id):
            state = record.cache.state if record.cache.state is not None else record.cache.refresh()
            values.append(getattr(state, attr))
        return np.stack(values, axis=0)

    def get_root_pos(self, obj_id: int):
        return self._stack(obj_id, "root_pos")

    def get_root_rot(self, obj_id: int):
        return self._stack(obj_id, "root_rot")

    def get_root_vel(self, obj_id: int):
        return self._stack(obj_id, "root_vel")

    def get_root_ang_vel(self, obj_id: int):
        return self._stack(obj_id, "root_ang_vel")

    def get_body_pos(self, obj_id: int):
        return self._stack(obj_id, "link_pos")

    def get_body_rot(self, obj_id: int):
        return self._stack(obj_id, "link_rot")

    def get_body_vel(self, obj_id: int):
        return self._stack(obj_id, "link_vel")

    def get_body_ang_vel(self, obj_id: int):
        return self._stack(obj_id, "link_ang_vel")

    def get_contact_forces(self, obj_id: int):
        return self._stack(obj_id, "link_contact_force")

    def get_ground_contact_forces(self, obj_id: int):
        return self._stack(obj_id, "link_ground_contact_force")

    def get_dof_pos(self, obj_id: int):
        return self._stack(obj_id, "dof_pos")

    def get_dof_vel(self, obj_id: int):
        return self._stack(obj_id, "dof_vel")

    def get_dof_forces(self, obj_id: int):
        return self._stack(obj_id, "dof_force")

    def get_obj_num_bodies(self, obj_id: int) -> int:
        return self.record(0, obj_id).cache.num_links

    def get_obj_num_dofs(self, obj_id: int) -> int:
        return self.record(0, obj_id).cache.num_dofs

    def get_obj_dof_names(self, obj_id: int) -> list[str]:
        return list(self.record(0, obj_id).cache.dof_names)

    def get_obj_dof_limits(self, obj_id: int):
        return self.record(0, obj_id).cache.dof_limits.copy()

    def get_obj_pd_gains(self, obj_id: int):
        cache = self.record(0, obj_id).cache
        return cache.dof_kps.copy(), cache.dof_kds.copy()

    def get_obj_effort_limits(self, obj_id: int):
        return self.record(0, obj_id).cache.dof_effort_limits.copy()

    def get_obj_link_masses(self, obj_id: int):
        return self.record(0, obj_id).cache.link_masses.copy()

    def calc_obj_mass(self, env_id: int, obj_id: int) -> float:
        obj = self.record(env_id, obj_id).articulation
        if hasattr(obj, "calc_mass"):
            return float(obj.calc_mass())
        if hasattr(obj, "get_mass"):
            return float(obj.get_mass())
        return 0.0

    def set_root_state(
        self,
        env_id: int | None,
        obj_id: int,
        pos,
        rot_xyzw,
        linear_velocity=None,
        angular_velocity=None,
    ):
        env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
        for eid in env_ids:
            self.record(eid, obj_id).cache.set_root(
                pos, rot_xyzw, linear_velocity, angular_velocity
            )

    def set_dof_state(self, env_id: int | None, obj_id: int, positions, velocities=None):
        env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
        for eid in env_ids:
            self.record(eid, obj_id).cache.set_dofs(positions, velocities)
