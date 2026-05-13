"""Small Python simulation host built on KangEngine PhysX bindings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from ._core import _ke
from .rigid import rigid_shape_specs
from .state import KangStateCache

try:
    import torch as _torch
except Exception:
    _torch = None


def _as_float_array(value):
    if _torch is not None and _torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _optional_drive_array(value, size: int):
    if value is None:
        return None
    arr = _as_float_array(value).reshape(-1)
    if arr.size == 1:
        arr = np.full(size, float(arr[0]), dtype=np.float32)
    if arr.size != size:
        raise ValueError(f"drive parameter expected {size} values, got {arr.size}")
    return arr.astype(np.float32, copy=True)


def _clip_forces(forces, limits):
    limits = _as_float_array(limits).reshape(-1)
    if limits.size == 1:
        limits = np.full_like(forces, float(limits[0]), dtype=np.float32)
    return np.clip(forces, -limits, limits).astype(np.float32, copy=False)


def _require_physx():
    missing = [
        name
        for name in (
            "PhysicsConfig",
            "PhysicsWorld",
            "ArticulationConfig",
            "Articulation",
        )
        if not hasattr(_ke, name)
    ]
    if missing:
        raise RuntimeError(
            "KangSimWorld requires KangEngine PhysX bindings. "
            f"Missing: {', '.join(missing)}"
        )


@dataclass(slots=True)
class SimArticulation:
    env_id: int
    obj_id: int
    name: str
    articulation: object


@dataclass(slots=True)
class SimRigid:
    env_id: int
    obj_id: int
    name: str
    rigid: object


class ControlMode(str, Enum):
    NONE = "none"
    POS = "pos"
    VEL = "vel"
    TORQUE = "torque"
    PD_EXPLICIT = "pd_explicit"


@dataclass(slots=True)
class CommandBuffer:
    mode: ControlMode
    cmd: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None


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


class KangSimWorld:
    """Owns a PhysX world, registered articulations, and batched state cache."""

    def __init__(
        self,
        num_envs: int = 1,
        physics_config=None,
        sim_dt: float | None = None,
        add_ground: bool = False,
    ):
        _require_physx()
        if physics_config is None:
            physics_config = _ke.PhysicsConfig.z_up()
        if sim_dt is not None:
            physics_config.dt = float(sim_dt)

        self.num_envs = int(num_envs)
        self.physics = _ke.PhysicsWorld(physics_config)
        if sim_dt is not None:
            self.physics.set_dt(float(sim_dt))
        if add_ground:
            self.physics.add_default_ground()

        self.state = KangStateCache(num_envs=self.num_envs)
        self.articulations: dict[tuple[int, int], SimArticulation] = {}
        self.rigids: dict[tuple[int, int], SimRigid] = {}
        self.commands: dict[tuple[int, int], CommandBuffer] = {}
        self.resets: dict[tuple[int, int], ResetBuffer] = {}
        self.body_forces: dict[tuple[int, int], np.ndarray] = {}
        self._mjcf_cache: dict[tuple[str, float, str], object] = {}
        self._mjcf_load_count = 0
        self.sim_time = 0.0
        self.sim_dt = float(physics_config.dt)
        self._released = False

    def add_articulation(
        self,
        data,
        env_id: int = 0,
        obj_id: int = 0,
        name: str = "",
        config=None,
    ) -> SimArticulation:
        if config is None:
            config = _ke.ArticulationConfig.free_base()
        if hasattr(config, "collision_group") and int(config.collision_group) == 0:
            config.collision_group = int(env_id) + 1
        key = (int(env_id), int(obj_id))
        if key in self.articulations or key in self.rigids:
            raise ValueError(f"object already registered at env={key[0]}, obj={key[1]}")

        articulation = _ke.Articulation.build(self.physics, data, config)
        record = SimArticulation(key[0], key[1], str(name), articulation)
        self.articulations[key] = record
        self.state.add_articulation(
            articulation, key[0], key[1], name, physics=self.physics
        )
        self.commands[key] = CommandBuffer(ControlMode.NONE)
        self.resets[key] = ResetBuffer()
        self.body_forces[key] = np.zeros(
            (articulation.num_links(), 3), dtype=np.float32
        )
        return record

    def add_rigid(
        self,
        data,
        env_id: int = 0,
        obj_id: int = 0,
        name: str = "",
        pos=None,
        rot_xyzw=None,
        density: float = 1.0,
        collision_group: int | None = None,
        contact_offset: float = 0.02,
        rest_offset: float = 0.0,
    ) -> SimRigid:
        key = (int(env_id), int(obj_id))
        if key in self.articulations or key in self.rigids:
            raise ValueError(f"object already registered at env={key[0]}, obj={key[1]}")
        if pos is None:
            pos = np.zeros(3, dtype=np.float32)
        if rot_xyzw is None:
            rot_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if collision_group is None:
            collision_group = key[0] + 1
        rigid = self.physics.create_dynamic_rigid(
            data,
            _as_float_array(pos).reshape(3),
            _as_float_array(rot_xyzw).reshape(4),
            float(density),
            int(collision_group),
            float(contact_offset),
            float(rest_offset),
        )
        shape_specs = rigid_shape_specs(data)
        body_names = [spec.name for spec in shape_specs]
        local_pos = np.stack([spec.local_pos for spec in shape_specs], axis=0)
        local_rot = np.stack([spec.local_rot for spec in shape_specs], axis=0)
        record = SimRigid(key[0], key[1], str(name), rigid)
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
        return record

    def add_mjcf_articulation(
        self,
        mjcf_path: str,
        env_id: int = 0,
        obj_id: int = 0,
        name: str = "",
        config=None,
        order: str = "DFS",
        scale: float = 1.0,
    ) -> SimArticulation:
        data = self.load_mjcf(mjcf_path, scale=scale, order=order)
        return self.add_articulation(data, env_id, obj_id, name, config)

    def load_mjcf(self, mjcf_path: str, scale: float = 1.0, order: str = "DFS"):
        key = (str(Path(mjcf_path).expanduser().resolve()), float(scale), str(order))
        data = self._mjcf_cache.get(key)
        if data is None:
            data = _ke.animation.MJCFLoader.load(key[0], scale=key[1], order=key[2])
            self._mjcf_cache[key] = data
            self._mjcf_load_count += 1
        return data

    def get_mjcf_cache_size(self) -> int:
        return len(self._mjcf_cache)

    def get_mjcf_load_count(self) -> int:
        return self._mjcf_load_count

    def set_cmd(
        self,
        env_id: int | None,
        obj_id: int,
        cmd,
        mode: ControlMode | str = ControlMode.POS,
        kp: float | None = 200.0,
        kd: float | None = 10.0,
    ):
        mode = ControlMode(mode)
        env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
        source = _as_float_array(cmd)
        for eid in env_ids:
            key = (eid, int(obj_id))
            if key in self.rigids:
                self.commands[key] = CommandBuffer(ControlMode.NONE)
                continue
            record = self.articulations[key]
            expected = record.articulation.num_dofs()
            if env_id is None and source.ndim >= 2 and source.shape[0] == self.num_envs:
                arr = source[eid].reshape(-1)
            else:
                arr = source.reshape(-1)
            if mode in (ControlMode.POS, ControlMode.VEL) and arr.size != expected:
                raise ValueError(
                    f"{mode.value} command for env={eid}, obj={obj_id} expected "
                    f"{expected} values, got {arr.size}"
                )
            kp_arr = _optional_drive_array(kp, expected)
            kd_arr = _optional_drive_array(kd, expected)
            self.commands[key] = CommandBuffer(mode, arr.copy(), kp_arr, kd_arr)

    def clear_cmd(self, env_id: int | None = None, obj_id: int | None = None):
        keys = list(self.commands.keys())
        for key in keys:
            if env_id is not None and key[0] != int(env_id):
                continue
            if obj_id is not None and key[1] != int(obj_id):
                continue
            self.commands[key] = CommandBuffer(ControlMode.NONE)

    def apply_commands(self):
        for key, buffer in self.commands.items():
            if buffer.mode == ControlMode.NONE or buffer.cmd is None:
                continue
            if key in self.rigids:
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
                if buffer.kp is not None:
                    articulation.set_kps(buffer.kp)
                if buffer.kd is not None:
                    articulation.set_kds(buffer.kd)
                kp = _as_float_array(articulation.get_kps()).reshape(-1)
                kd = _as_float_array(articulation.get_kds()).reshape(-1)
                dof_pos = _as_float_array(articulation.get_dof_positions()).reshape(-1)
                dof_vel = _as_float_array(articulation.get_dof_velocities()).reshape(-1)
                torque = kp * (buffer.cmd - dof_pos) - kd * dof_vel
                torque = _clip_forces(torque, articulation.get_effort_limits())
                articulation.set_joint_forces(torque)
            else:
                raise NotImplementedError(
                    f"control mode '{buffer.mode.value}' is not implemented yet"
                )

    def apply_resets(self):
        for key, reset in self.resets.items():
            if not reset.pending:
                continue
            env_id, obj_id = key
            cache = self.state.record(env_id, obj_id).cache
            if reset.root is not None:
                cache.set_root(
                    reset.root.pos,
                    reset.root.rot_xyzw,
                    reset.root.linear_velocity,
                    reset.root.angular_velocity,
                )
            if reset.dof is not None:
                cache.set_dofs(reset.dof.positions, reset.dof.velocities)
            self.resets[key] = ResetBuffer()

    def set_body_force(
        self,
        env_id: int | None,
        obj_id: int,
        body_id: int,
        force,
    ):
        env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
        for eid in env_ids:
            key = (eid, int(obj_id))
            forces = self.body_forces[key]
            body_idx = int(body_id)
            if body_idx < 0 or body_idx >= forces.shape[0]:
                raise IndexError(
                    f"body_id {body_idx} out of range for env={eid}, obj={obj_id}"
                )
            forces[body_idx] = _as_float_array(force).reshape(3)

    def apply_body_forces(self):
        for key, forces in self.body_forces.items():
            if not np.any(forces):
                continue
            for body_id, force in enumerate(forces):
                if np.any(force):
                    if key in self.articulations:
                        self.articulations[key].articulation.add_link_force(
                            int(body_id), force
                        )
                    elif key in self.rigids:
                        self.rigids[key].rigid.add_force(force)

    def clear_body_forces(self):
        for forces in self.body_forces.values():
            forces.fill(0.0)

    def step(self, substeps: int = 1, refresh: bool = True, apply_commands: bool = True):
        self.apply_resets()
        if apply_commands:
            self.apply_commands()
        for _ in range(int(substeps)):
            self.apply_body_forces()
            self.physics.step()
            self.sim_time += self.sim_dt
        self.clear_body_forces()
        if refresh:
            self.state.refresh()
        return self.state

    def refresh(self):
        return self.state.refresh()

    def articulation(self, env_id: int = 0, obj_id: int = 0):
        return self.articulations[(int(env_id), int(obj_id))].articulation

    def rigid(self, env_id: int = 0, obj_id: int = 0):
        return self.rigids[(int(env_id), int(obj_id))].rigid

    def set_root_state(
        self,
        env_id: int | None,
        obj_id: int,
        pos,
        rot_xyzw,
        linear_velocity=None,
        angular_velocity=None,
        immediate: bool = False,
    ):
        if immediate:
            self.state.set_root_state(
                env_id, obj_id, pos, rot_xyzw, linear_velocity, angular_velocity
            )
            env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
            for eid in env_ids:
                self.resets[(eid, int(obj_id))].root = None
            return

        env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
        root = RootStateReset(
            _as_float_array(pos).reshape(3),
            _as_float_array(rot_xyzw).reshape(4),
            None
            if linear_velocity is None
            else _as_float_array(linear_velocity).reshape(3),
            None
            if angular_velocity is None
            else _as_float_array(angular_velocity).reshape(3),
        )
        for eid in env_ids:
            key = (eid, int(obj_id))
            if key not in self.articulations and key not in self.rigids:
                raise KeyError(f"no object registered at env={eid}, obj={obj_id}")
            self.resets[key].root = root

    def set_dof_state(
        self,
        env_id: int | None,
        obj_id: int,
        positions,
        velocities=None,
        immediate: bool = False,
    ):
        if immediate:
            self.state.set_dof_state(env_id, obj_id, positions, velocities)
            env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
            for eid in env_ids:
                self.resets[(eid, int(obj_id))].dof = None
            return

        env_ids = range(self.num_envs) if env_id is None else [int(env_id)]
        for eid in env_ids:
            key = (eid, int(obj_id))
            if key in self.rigids:
                continue
            expected = self.articulations[key].articulation.num_dofs()
            pos = _as_float_array(positions).reshape(-1)
            if pos.size != expected:
                raise ValueError(
                    f"dof reset for env={eid}, obj={obj_id} expected "
                    f"{expected} positions, got {pos.size}"
                )
            vel = None
            if velocities is not None:
                vel = _as_float_array(velocities).reshape(-1)
                if vel.size != expected:
                    raise ValueError(
                        f"dof reset for env={eid}, obj={obj_id} expected "
                        f"{expected} velocities, got {vel.size}"
                    )
            self.resets[key].dof = DofStateReset(
                pos.copy(), None if vel is None else vel.copy()
            )

    def release(self):
        if self._released:
            return
        for record in self.articulations.values():
            record.articulation.release()
        self.articulations.clear()
        self.rigids.clear()
        self.commands.clear()
        self.resets.clear()
        self.physics = None
        self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
