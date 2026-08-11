"""MimicKit-style KangEngine adapter.

This module intentionally stays thin: it maps MimicKit's engine method names to
``KangSimWorld`` while KangSimWorld owns PhysX stepping, command buffers, reset
buffers, and state refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import sys
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

from ._core import _ke
from .app import App
from .recording import VideoRecorder
from .rigid import expand_rigid_body_state, rigid_body_names, rigid_shape_specs
from .sim import ControlMode, KangSimWorld
from .utils import preset_rgba
from .utils.env_utils import EnvIdLike, env_id_list
from .utils.tensor import as_cpu_numpy, as_tensor, resolve_device
from .visual import sim as visual_sim

try:
    import engines.engine as _mk_engine
except ImportError:
    _mk_engine = None

try:
    _ArticulationConfig = _ke.physics.ArticulationConfig
except AttributeError:
    _ArticulationConfig = None

if _mk_engine is not None:
    _BaseEngine = _mk_engine.Engine
    MimicControlMode = _mk_engine.ControlMode
    MimicObjType = _mk_engine.ObjType
else:

    class _BaseEngine:
        def __init__(self, visualize=False):
            self._prev_frame_time = 0.0

    class MimicControlMode(Enum):
        none = 0
        pos = 1
        vel = 2
        torque = 3
        pd_explicit = 4

    class MimicObjType(Enum):
        rigid = 0
        articulated = 1


_CONTROL_MODE_MAP = {
    "none": ControlMode.NONE,
    "pos": ControlMode.POS,
    "vel": ControlMode.VEL,
    "torque": ControlMode.TORQUE,
    "pd_explicit": ControlMode.PD_EXPLICIT,
}

ArrayLike: TypeAlias = Any
RootComponent: TypeAlias = Literal["pos", "rot", "vel", "ang_vel"]
DofComponent: TypeAlias = Literal["pos", "vel"]
BodyVelocityKind: TypeAlias = Literal["vel", "ang_vel"]


@dataclass(slots=True)
class KangEngineObject:
    env_id: int
    obj_id: int
    obj_type: object
    asset_file: str
    name: str
    data: object
    body_names: list[str]
    is_visual: bool
    num_dofs: int
    fixed_base: bool
    is_static: bool
    start_pos: np.ndarray
    start_rot: np.ndarray
    color: object | None = None


@dataclass(slots=True)
class _BodyVelocityOverrideBuffers:
    num_bodies: np.ndarray
    visual_mask: np.ndarray
    vel_values: torch.Tensor
    vel_masks: np.ndarray
    ang_vel_values: torch.Tensor
    ang_vel_masks: np.ndarray


@dataclass(slots=True)
class _PendingStateBuffers:
    root_state: torch.Tensor
    root_dirty: np.ndarray
    dof_state: torch.Tensor
    dof_dirty: np.ndarray
    dof_widths: np.ndarray


def _asset_path(*parts):
    from pathlib import Path

    return str(Path(__file__).resolve().parent / "assets" / Path(*parts))


def _parse_drive_string(value):
    text = str(value).strip()
    if "," not in text:
        return float(text)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class _KangEngineViewer(App):
    def __init__(self, world, headless=False):
        super().__init__()
        self.world = world
        self.headless = bool(headless)
        self.visual_bridge = None
        self.robot_material = None
        self.ground_material = None
        self._setup_done = False
        self._debug_line_views = {}

    def setup_viewer(self, width=1920, height=1080, headless=None):
        if self._setup_done:
            return
        if headless is None:
            headless = self.headless
        self.initialize(width, height, False, _ke.UpAxis.Z, headless=bool(headless))
        materials = self.create_standard_materials()
        self.robot_material = materials.common
        self.ground_material = materials.ground
        self.scene.add_ground(scale=100.0, material=self.ground_material)
        self.visual_bridge = visual_sim.SimWorldVisualizer(self, self.world)
        self._setup_done = True

    def add_articulation_scene_graph(
        self, env_id, obj_id, asset_file, order, color=None
    ):
        self.setup_viewer()
        self.visual_bridge.add_articulation_scene_graph(
            env_id,
            obj_id,
            asset_file,
            path=f"/env_{env_id}/obj_{obj_id}",
            order=order,
            material=self.robot_material,
            color=color,
        )

    def add_articulation_skin(self, env_id, obj_id, asset_file, order, color=None):
        self.setup_viewer()
        self.visual_bridge.add_articulation_skin(
            env_id,
            obj_id,
            asset_file,
            path=f"/env_{env_id}/visual_obj_{obj_id}",
            order=order,
            material=self.robot_material,
            color=color,
        )

    def add_rigid_scene_graph(
        self,
        env_id,
        obj_id,
        asset_file,
        path,
        order,
        material=None,
        color=None,
    ):
        self.setup_viewer()
        self.visual_bridge.add_rigid_scene_graph(
            env_id,
            obj_id,
            asset_file,
            path=path,
            order=order,
            material=self.robot_material if material is None else material,
            color=color,
        )

    def sync(self):
        if self.visual_bridge is not None:
            self.visual_bridge.sync()

    def render_once(self):
        self.render_frame_once()

    def draw_lines(self, slot, starts, ends, colors, line_width):
        self.setup_viewer()
        radius = max(0.0025, float(line_width) * 0.0025)
        path = f"/debug/mimickit_lines_{int(slot)}"
        view = self._debug_line_views.get(int(slot))
        if view is None:
            view = self.scene.log_lines(
                path,
                self.robot_material,
                starts,
                ends,
                colors,
                radius,
                8,
            )
            self._debug_line_views[int(slot)] = view
        else:
            view.update_lines(starts, ends, colors)

    def clear_unused_debug_lines(self, active_slots):
        for slot, view in self._debug_line_views.items():
            if slot >= active_slots:
                view.update_lines([], [], [])

    def render(self):
        pass


class KangEngineEngine(_BaseEngine):
    """MVP KangEngine backend with MimicKit-like method names."""

    def __init__(
        self, config, num_envs, device=None, visualize=False, record_video=False
    ):
        super().__init__(visualize)
        self._config = dict(config)
        self._num_envs = int(num_envs)
        self._device = resolve_device(device)
        self._visualize = bool(visualize)
        self._record_video = bool(record_video)

        sim_freq = float(self._config.get("sim_freq", 60))
        control_freq = float(self._config.get("control_freq", sim_freq))
        self._sim_timestep = 1.0 / sim_freq
        self._timestep = 1.0 / control_freq
        self._sim_steps = max(1, int(round(sim_freq / control_freq)))

        mode_name = self._config.get("control_mode", "none")
        self._control_mode = MimicControlMode[mode_name]
        self._kang_control_mode = _CONTROL_MODE_MAP[mode_name]
        self._env_spacing = float(self._config.get("env_spacing", 5.0))
        self._env_offsets = np.zeros((self._num_envs, 3), dtype=np.float32)
        self._env_offsets[:, 0] = (
            np.arange(self._num_envs, dtype=np.float32) * self._env_spacing
        )
        self._env_offsets_torch = torch.as_tensor(
            self._env_offsets,
            dtype=torch.float32,
            device=self._device,
        )

        sim_device = self._config.get("sim_device")
        if sim_device is None:
            sim_device = str(self._device) if self._device.type == "cuda" else "cpu"

        physics_config = _ke.physics.PhysicsConfig.z_up()
        found_lost_pairs_capacity = self._config.get(
            "found_lost_pairs_capacity",
            self._config.get("gpu_found_lost_pairs_capacity", None),
        )
        if found_lost_pairs_capacity is not None:
            physics_config.gpu_dynamics.found_lost_pairs_capacity = int(
                found_lost_pairs_capacity
            )
        self._world = KangSimWorld(
            num_envs=self._num_envs,
            physics_config=physics_config,
            sim_dt=self._sim_timestep,
            add_ground=bool(self._config.get("add_ground", True)),
            sim_device=sim_device,
            device=self._device,
        )
        self._created_envs: list[int] = []
        self._objects: list[list[KangEngineObject]] = [
            [] for _ in range(self._num_envs)
        ]
        self._gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        self._initialized = False
        self._released = False
        self._headless = _parse_bool(self._config.get("headless", False))
        self._viewer = (
            _KangEngineViewer(self._world, headless=self._headless)
            if self._visualize or self._record_video
            else None
        )
        record_dir = self._config.get("record_dir", None)
        record_fps = int(float(self._config.get("record_fps", control_freq)))
        self._record_video_file = self._config.get(
            "record_video_file", self._config.get("video_file", None)
        )
        self._record_video_saved = False
        self._recorder = (
            VideoRecorder(
                self._record_video_file,
                fps=record_fps,
                retain_frames=True,
                frame_dir=record_dir,
            )
            if self._record_video
            else None
        )
        self._key_callbacks = []
        self._draw_line_count = 0
        self._visual_body_pos = {}
        self._visual_body_rot = {}
        self._visual_root_pos = {}
        self._visual_root_rot = {}
        self._body_velocity_overrides: _BodyVelocityOverrideBuffers | None = None
        self._pending_state: _PendingStateBuffers | None = None
        self._contact_sensors = {}
        self._contact_body_indices = {}
        self._debug_visual_pose = os.environ.get(
            "KANGENGINE_DEBUG_VISUAL_POSE"
        ) == "1" or _parse_bool(self._config.get("debug_visual_pose", False))
        self._debug_visual_pose_stats = {}
        self._sim_color_override = self._color_override("sim_color")
        self._ref_color_override = self._color_override("ref_color")
        self._auto_init_gpu_system = self._world.sim_device.type == "cuda"

    def get_name(self):
        return "kangengine"

    def create_env(self, env_id=None):
        if env_id is None:
            env_id = len(self._created_envs)
        env_id = int(env_id)
        if env_id < 0 or env_id >= self._num_envs:
            raise ValueError(
                f"env_id {env_id} out of range for num_envs={self._num_envs}"
            )
        if env_id in self._created_envs:
            raise ValueError(f"env {env_id} already exists")
        self._created_envs.append(env_id)
        return env_id

    def create_obj(
        self,
        env_id,
        obj_type,
        asset_file,
        name,
        is_visual=False,
        enable_self_collisions=True,
        fix_root=False,
        start_pos=None,
        start_rot=None,
        color=None,
        disable_motors=False,
    ):
        obj_type_name = str(getattr(obj_type, "name", obj_type)).lower()
        is_articulated = obj_type_name == "articulated"
        is_rigid = obj_type_name == "rigid"
        if not is_articulated and not is_rigid:
            raise NotImplementedError(
                "only articulated and rigid MJCF objects are supported yet"
            )
        fixed_base = is_articulated and bool(fix_root)
        is_static = is_rigid and bool(fix_root)

        env_id = int(env_id)
        obj_id = len(self._objects[env_id])
        start_pos = (
            np.zeros(3, dtype=np.float32)
            if start_pos is None
            else self._as_numpy(start_pos).reshape(3)
        )
        start_rot = (
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if start_rot is None
            else self._as_numpy(start_rot).reshape(4)
        )

        mjcf_order = self._config.get("mjcf_order", "DFS")
        data = self._world.load_mjcf(asset_file, order=mjcf_order)
        body_names = list(data.skeleton_tree.node_names())
        if is_rigid and not is_visual:
            body_names = rigid_body_names(data)
        num_dofs = (
            0 if is_rigid else sum(len(joints) for joints in data.joints.values())
        )

        if not is_visual and is_articulated:
            if _ArticulationConfig is None:
                raise RuntimeError(
                    "ArticulationConfig not available: build with USE_PHYSX=ON"
                )
            cfg = (
                _ke.physics.ArticulationConfig.fixed_base()
                if fixed_base
                else _ke.physics.ArticulationConfig.free_base()
            )
            if "enable_self_collisions" in self._config:
                enable_self_collisions = _parse_bool(
                    self._config["enable_self_collisions"]
                )
            cfg.disable_self_collision = not bool(enable_self_collisions)
            cfg.contact_offset = float(self._config.get("contact_offset", 0.02))
            cfg.rest_offset = float(self._config.get("rest_offset", 0.0))
            self._world.add_articulation(
                data,
                env_id=env_id,
                obj_id=obj_id,
                name=name,
                config=cfg,
            )
            self._apply_drive_config(env_id, obj_id)
        elif not is_visual and is_rigid:
            rigid_kwargs = {
                "pos": self._world_pos(env_id, start_pos),
                "rot_xyzw": start_rot,
                "contact_offset": float(self._config.get("contact_offset", 0.02)),
                "rest_offset": float(self._config.get("rest_offset", 0.0)),
            }
            if is_static:
                self._world.add_static_rigid(
                    data,
                    env_id=env_id,
                    obj_id=obj_id,
                    name=name,
                    **rigid_kwargs,
                )
            else:
                self._world.add_rigid(
                    data,
                    env_id=env_id,
                    obj_id=obj_id,
                    name=name,
                    density=float(self._config.get("rigid_density", 1.0)),
                    kinematic=_parse_bool(self._config.get("rigid_kinematic", False)),
                    **rigid_kwargs,
                )
        obj = KangEngineObject(
            env_id,
            obj_id,
            obj_type,
            str(asset_file),
            str(name),
            data,
            body_names,
            bool(is_visual),
            int(num_dofs),
            fixed_base,
            is_static,
            start_pos,
            start_rot,
        )
        self._objects[env_id].append(obj)
        if not is_visual and not is_static:
            # Separate envs before the first GPU step to avoid pair explosion.
            self._world.set_root_state(
                env_id,
                obj_id,
                self._world_pos(env_id, start_pos),
                start_rot,
                immediate=True,
            )
        viewer_color = self._viewer_color(is_visual, color)
        obj.color = viewer_color
        if self._viewer is not None:
            if is_visual:
                self._viewer.add_articulation_skin(
                    env_id, obj_id, asset_file, mjcf_order, color=viewer_color
                )
        if is_visual:
            self.set_root_pos(env_id, obj_id, start_pos)
            self.set_root_rot(env_id, obj_id, start_rot)
        return obj_id

    def initialize_sim(self):
        self._validate_envs()
        for env_objs in self._objects:
            for obj in env_objs:
                if obj.is_visual or self._is_static_obj(obj.obj_id):
                    continue
                # Apply env offsets before GPU broadphase initialization.
                self._world.set_root_state(
                    obj.env_id,
                    obj.obj_id,
                    self._world_pos(obj.env_id, obj.start_pos),
                    obj.start_rot,
                    immediate=True,
                )
        self._init_gpu_system_if_needed()
        self._register_contact_sensors()
        self._world.step(substeps=0, apply_commands=False)
        self._register_sim_visual_batches()
        self._init_body_velocity_override_buffers()
        self._init_state_pending_buffers()
        self._initialized = True

    def _register_contact_sensors(self):
        if not self._use_gpu_logical_state():
            return
        for obj_id in range(self.get_objs_per_env()):
            first = self._objects[0][obj_id]
            if first.is_visual or self._is_static_obj(obj_id):
                continue
            obj_type_name = str(getattr(first.obj_type, "name", first.obj_type)).lower()
            if obj_type_name == "articulated":
                target = self._world.get_articulation_batch(obj_id=obj_id)
                body_indices = tuple(
                    int(value)
                    for value in self._world.articulation(
                        env_id=0, obj_id=obj_id
                    ).get_link_indices()
                )
            elif obj_type_name == "rigid":
                target = self._world.get_rigid_batch(obj_id=obj_id)
                body_indices = tuple(range(int(target.num_bodies)))
            else:
                continue
            self._contact_sensors[obj_id] = target.add_force_sensor(
                body_ids=None,
                name=f"mimickit_obj_{obj_id}_contact_force",
            )
            self._contact_body_indices[obj_id] = body_indices

    def _init_gpu_system_if_needed(self):
        if not self._auto_init_gpu_system:
            return
        if getattr(self._world, "_gpu_system", None) is not None:
            return
        self._world.init_gpu_system()

    def _register_sim_visual_batches(self):
        if self._viewer is None:
            return
        self._viewer.setup_viewer(headless=self._headless)
        objs_per_env = self.get_objs_per_env()
        mjcf_order = self._config.get("mjcf_order", "DFS")
        for obj_id in range(objs_per_env):
            first = self._objects[0][obj_id]
            if first.is_visual:
                continue
            if self._viewer.visual_bridge.get_visual_batch(obj_id) is not None:
                continue
            self._validate_visual_batch_slot(obj_id)
            obj_type_name = str(getattr(first.obj_type, "name", first.obj_type)).lower()
            if obj_type_name == "articulated":
                sim_handle = self._world.get_articulation_batch(obj_id=obj_id)
            elif obj_type_name == "rigid":
                sim_handle = self._world.get_rigid_batch(obj_id=obj_id)
            else:
                continue
            self._viewer.visual_bridge.add(
                sim_handle,
                first.asset_file,
                path=f"/sim_obj_{obj_id}",
                order=mjcf_order,
                material=self._viewer.robot_material,
                color=self._visual_batch_color(obj_id),
            )

    def set_cmd(self, obj_id, cmd):
        if self._kang_control_mode == ControlMode.NONE:
            return
        self._world.set_cmd(
            None,
            obj_id,
            cmd,
            self._kang_control_mode,
            kp=None,
            kd=None,
        )

    def step(self):
        self._flush_staged_state()
        self._world.step(
            substeps=self._sim_steps,
            refresh=not self._use_gpu_logical_state(),
        )
        if self._use_gpu_logical_state():
            self._world.state.gpu.refresh_frame_cache()
        self._clear_state_pending_overrides()
        self._clear_dynamic_body_velocity_overrides()
        self._clear_dynamic_visual_body_overrides()

    def render(self):
        if self._viewer is None:
            return
        base_render = getattr(_BaseEngine, "render", None)
        if base_render is not None:
            base_render(self)
        if self._viewer.should_close():
            sys.exit(0)
        self._flush_staged_state()
        self._world.step(substeps=0, apply_commands=False)
        self._clear_state_pending_overrides()
        self._viewer.sync()
        # ViewMotionEnv sends FK body transforms directly through set_body_pos/rot.
        # Apply them after PhysicsBridge sync so the viewer shows MimicKit's
        # reference motion pose, not the current PhysX articulation pose.
        self._apply_visual_body_overrides()
        for key, callback in self._key_callbacks:
            if self._viewer.was_key_pressed(key):
                callback()
        self._viewer.clear_unused_debug_lines(self._draw_line_count)
        self._draw_line_count = 0
        self._viewer.render_once()
        if self._viewer.should_close():
            sys.exit(0)
        if self._recorder is not None:
            self._recorder.write(self.get_rgb_pixels())

    def set_camera_pose(self, pos, look_at):
        if self._viewer is None:
            return
        camera = self._viewer.get_camera()
        p = self._as_numpy(pos).reshape(3)
        t = self._as_numpy(look_at).reshape(3)
        camera.set_camera_pos(_ke.Vec3(float(p[0]), float(p[1]), float(p[2])))
        camera.set_target_pos(_ke.Vec3(float(t[0]), float(t[1]), float(t[2])))

    def get_camera_pos(self):
        if self._viewer is None:
            return np.zeros(3, dtype=np.float32)
        p = self._viewer.get_camera().get_camera_pos()
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    def get_camera_dir(self):
        if self._viewer is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d = self._viewer.get_camera().get_camera_look_dir()
        return np.array([d.x, d.y, d.z], dtype=np.float32)

    def get_rgb_pixels(self):
        if self._viewer is None:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        self._viewer.setup_viewer(headless=self._headless)
        return np.asarray(self._viewer.read_rgb_pixels(True), dtype=np.uint8)

    def enabled_record_video(self):
        return self._recorder is not None

    def start_video_recording(self):
        if self._recorder is not None:
            self._record_video_saved = False
            self._recorder.start()

    def stop_video_recording(self):
        if self._recorder is not None:
            self._recorder.stop()
            self._save_video_recording()

    def get_video_recording(self):
        self._save_video_recording()
        return self._recorder

    def get_record_dir(self):
        if self._recorder is None or self._recorder.out_dir is None:
            return ""
        return str(self._recorder.out_dir)

    def _save_video_recording(self):
        if (
            self._recorder is None
            or self._record_video_saved
            or not self._record_video_file
            or self._recorder.get_num_frames() == 0
        ):
            return
        self._recorder.save(self._record_video_file)
        self._record_video_saved = True

    def get_timestep(self):
        return self._timestep

    def get_sim_timestep(self):
        return self._sim_timestep

    def get_sim_time(self):
        return self._world.sim_time

    def get_num_envs(self):
        return self._num_envs

    def get_gravity(self):
        return self._out(self._gravity)

    def get_objs_per_env(self):
        return len(self._objects[0]) if self._objects else 0

    def get_root_pos(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(self._visual_root_batch(self._visual_root_pos, obj_id, 3))
        values = self._pending_root_batch(obj_id, "pos", self._state_root_pos(obj_id))
        return self._out(self._local_pos_batch(values))

    def get_root_rot(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(self._visual_root_batch(self._visual_root_rot, obj_id, 4))
        return self._out(
            self._pending_root_batch(obj_id, "rot", self._state_root_rot(obj_id))
        )

    def get_root_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(np.zeros((self._num_envs, 3), dtype=np.float32))
        return self._out(
            self._pending_root_batch(obj_id, "vel", self._state_root_vel(obj_id))
        )

    def get_root_ang_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(np.zeros((self._num_envs, 3), dtype=np.float32))
        return self._out(
            self._pending_root_batch(
                obj_id, "ang_vel", self._state_root_ang_vel(obj_id)
            )
        )

    def get_dof_pos(self, obj_id):
        return self._out(
            self._pending_dof_batch(obj_id, "pos", self._state_dof_pos(obj_id))
        )

    def get_dof_vel(self, obj_id):
        return self._out(
            self._pending_dof_batch(obj_id, "vel", self._state_dof_vel(obj_id))
        )

    def get_dof_forces(self, obj_id):
        return self._out(self._state_dof_forces(obj_id))

    def get_body_pos(self, obj_id):
        if self._is_visual_obj(obj_id):
            value = self._visual_body_pose_batch(self._visual_body_pos, obj_id, 3)
            return self._out(self._local_pos_batch(value))
        value = self._pending_body_pose_batch(
            self._visual_body_pos, obj_id, self._state_body_pos(obj_id)
        )
        return self._out(self._local_pos_batch(value))

    def get_body_rot(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(
                self._visual_body_pose_batch(self._visual_body_rot, obj_id, 4)
            )
        return self._out(
            self._pending_body_pose_batch(
                self._visual_body_rot, obj_id, self._state_body_rot(obj_id)
            )
        )

    def get_body_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(self._body_velocity_override_batch("vel", obj_id))
        values = self._state_body_vel(obj_id)
        return self._out(self._body_velocity_override_batch("vel", obj_id, values))

    def get_body_ang_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(self._body_velocity_override_batch("ang_vel", obj_id))
        values = self._state_body_ang_vel(obj_id)
        return self._out(self._body_velocity_override_batch("ang_vel", obj_id, values))

    def get_contact_forces(self, obj_id):
        if self._is_static_obj(obj_id):
            shape = (self._num_envs, self.get_obj_num_bodies(obj_id), 3)
            return self._out(torch.zeros(shape, device=self._device))
        return self._out(self._world.state.get_contact_forces(obj_id))

    def get_ground_contact_forces(self, obj_id):
        sensor = self._contact_sensors.get(int(obj_id))
        if sensor is not None:
            # MimicKit environments separate simulated objects spatially and
            # disable articulation self-collision. Under that contract the
            # per-body normal contact force is the ground-contact force.
            forces = sensor.force
            body_indices = self._contact_body_indices[int(obj_id)]
            index = torch.as_tensor(
                body_indices, dtype=torch.long, device=forces.device
            )
            return self._out(forces.index_select(1, index))
        if self._is_static_obj(obj_id):
            shape = (self._num_envs, self.get_obj_num_bodies(obj_id), 3)
            return self._out(torch.zeros(shape, device=self._device))
        return self._out(self._world.state.get_ground_contact_forces(obj_id))

    def set_root_pos(self, env_id: EnvIdLike, obj_id: int, root_pos: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            self._set_visual_root_override(
                self._visual_root_pos, env_id, obj_id, root_pos, 3
            )
            self._apply_visual_root_overrides(obj_id)
            return
        if self._is_static_obj(obj_id):
            return
        self._stage_root_component(env_id, obj_id, "pos", root_pos)

    def set_root_rot(self, env_id: EnvIdLike, obj_id: int, root_rot: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            self._set_visual_root_override(
                self._visual_root_rot, env_id, obj_id, root_rot, 4
            )
            self._apply_visual_root_overrides(obj_id)
            return
        if self._is_static_obj(obj_id):
            return
        self._stage_root_component(env_id, obj_id, "rot", root_rot)

    def set_root_vel(self, env_id: EnvIdLike, obj_id: int, root_vel: ArrayLike) -> None:
        if self._is_visual_obj(obj_id) or self._is_static_obj(obj_id):
            return
        self._stage_root_component(env_id, obj_id, "vel", root_vel)

    def set_root_ang_vel(
        self, env_id: EnvIdLike, obj_id: int, root_ang_vel: ArrayLike
    ) -> None:
        if self._is_visual_obj(obj_id) or self._is_static_obj(obj_id):
            return
        self._stage_root_component(env_id, obj_id, "ang_vel", root_ang_vel)

    def set_dof_pos(self, env_id: EnvIdLike, obj_id: int, dof_pos: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            return
        self._stage_dof_component(env_id, obj_id, "pos", dof_pos)

    def set_dof_vel(self, env_id: EnvIdLike, obj_id: int, dof_vel: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            return
        self._stage_dof_component(env_id, obj_id, "vel", dof_vel)

    def set_body_vel(self, env_id: EnvIdLike, obj_id: int, body_vel: ArrayLike) -> None:
        self._set_body_velocity_override(
            "vel",
            env_id,
            obj_id,
            body_vel,
        )
        return

    def set_body_ang_vel(
        self, env_id: EnvIdLike, obj_id: int, body_ang_vel: ArrayLike
    ) -> None:
        self._set_body_velocity_override(
            "ang_vel",
            env_id,
            obj_id,
            body_ang_vel,
        )
        return

    def set_body_pos(self, env_id: EnvIdLike, obj_id: int, body_pos: ArrayLike) -> None:
        # MimicKit uses this in view_motion to publish global FK body positions
        # and during deferred resets. Physics state is still owned by root/DOF
        # setters, but queries must expose this pose until PhysX applies them.
        self._set_visual_body_override(
            self._visual_body_pos, env_id, obj_id, body_pos, 3
        )
        if self._viewer is not None:
            self._debug_visual_body_pose(obj_id)
            self._apply_visual_body_overrides()
        return

    def set_body_rot(self, env_id: EnvIdLike, obj_id: int, body_rot: ArrayLike) -> None:
        # MimicKit body rotations are xyzw quaternions in world space. Cache them
        # for reset-time state reads and the viewer path; the physics backend
        # does not accept direct body poses.
        self._set_visual_body_override(
            self._visual_body_rot, env_id, obj_id, body_rot, 4
        )
        if self._viewer is not None:
            self._apply_visual_body_overrides()
        return

    def set_body_forces(
        self, env_id: EnvIdLike, obj_id: int, body_id: int, forces: ArrayLike
    ) -> None:
        if self._is_visual_obj(obj_id):
            return
        env_ids = self._env_ids(env_id)
        arr = self._as_numpy(forces)
        if arr.ndim == 0:
            arr = np.full(3, float(arr), dtype=np.float32)
        for local_id, eid in enumerate(env_ids):
            force = self._select_flat_value(arr, eid, 3, local_id, len(env_ids))
            self._world.set_body_force(eid, obj_id, int(body_id), force)
        return

    def register_keyboard_callback(self, key_str, callback_func):
        key_name = str(key_str)
        if key_name == "ESC":
            key_name = "ESCAPE"
        key = getattr(_ke.keys, key_name, None)
        if key is not None:
            self._key_callbacks.append((key, callback_func))

    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        if self._viewer is None:
            return
        starts_np = self._as_numpy(start_verts).reshape(-1, 3)
        ends_np = self._as_numpy(end_verts).reshape(-1, 3)
        env_ids = self._env_ids(env_id)
        if len(env_ids) == 1:
            starts_np = self._world_pos(env_ids[0], starts_np)
            ends_np = self._world_pos(env_ids[0], ends_np)
        elif starts_np.shape[0] == len(env_ids) and ends_np.shape[0] == len(env_ids):
            offsets = np.asarray(
                [self._env_offset(eid) for eid in env_ids], dtype=np.float32
            )
            starts_np = starts_np + offsets
            ends_np = ends_np + offsets
        colors_np = self._as_numpy(cols)
        if colors_np.size == 0:
            colors_np = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        colors_np = colors_np.reshape(-1, colors_np.shape[-1])
        if colors_np.shape[-1] == 3:
            colors_np = np.concatenate(
                [colors_np, np.ones((colors_np.shape[0], 1), dtype=np.float32)],
                axis=-1,
            )
        starts = [_ke.Vec3(float(v[0]), float(v[1]), float(v[2])) for v in starts_np]
        ends = [_ke.Vec3(float(v[0]), float(v[1]), float(v[2])) for v in ends_np]
        colors = [
            _ke.Vec4(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            for c in colors_np
        ]
        self._viewer.draw_lines(
            self._draw_line_count,
            starts,
            ends,
            colors,
            line_width,
        )
        self._draw_line_count += 1
        return

    def get_obj_type(self, obj_id):
        return self._objects[0][obj_id].obj_type

    def get_obj_num_dofs(self, obj_id):
        if self._is_visual_obj(obj_id) or self._is_static_obj(obj_id):
            return self._objects[0][obj_id].num_dofs
        return self._world.state.get_obj_num_dofs(obj_id)

    def get_obj_num_bodies(self, obj_id):
        if self._is_visual_obj(obj_id) or self._is_static_obj(obj_id):
            return len(self._objects[0][obj_id].body_names)
        return self._world.state.get_obj_num_bodies(obj_id)

    def get_obj_body_names(self, obj_id):
        return list(self._objects[0][obj_id].body_names)

    def find_obj_body_id(self, obj_id, body_name):
        return self.get_obj_body_names(obj_id).index(body_name)

    def get_obj_torque_limits(self, env_id, obj_id):
        if self._is_visual_obj(obj_id) or self._is_static_obj(obj_id):
            return np.full(self.get_obj_num_dofs(obj_id), np.inf, dtype=np.float32)
        return self._as_numpy(self._world.state.get_obj_effort_limits(obj_id))

    def get_obj_dof_limits(self, env_id, obj_id):
        if self._is_visual_obj(obj_id) or self._is_static_obj(obj_id):
            n = self.get_obj_num_dofs(obj_id)
            return -np.full(n, np.inf, dtype=np.float32), np.full(
                n, np.inf, dtype=np.float32
            )
        limits = self._world.state.get_obj_dof_limits(obj_id)
        limits = self._as_numpy(limits)
        return limits[:, 0], limits[:, 1]

    def get_obj_pd_gains(self, env_id, obj_id):
        if self._is_visual_obj(obj_id) or self._is_static_obj(obj_id):
            n = self.get_obj_num_dofs(obj_id)
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        kp, kd = self._world.state.get_obj_pd_gains(obj_id)
        return self._as_numpy(kp), self._as_numpy(kd)

    def calc_obj_mass(self, env_id, obj_id):
        if self._is_static_obj(obj_id):
            return float("inf")
        return self._world.state.calc_obj_mass(int(env_id), int(obj_id))

    def get_control_mode(self):
        return self._control_mode

    def release(self):
        if self._released:
            return
        self._save_video_recording()
        self._viewer = None
        self._world.release()
        self._released = True

    def close(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def _validate_envs(self):
        if len(self._created_envs) != self._num_envs:
            missing = sorted(set(range(self._num_envs)) - set(self._created_envs))
            raise RuntimeError(f"not all envs were created; missing {missing}")
        counts = {len(env_objs) for env_objs in self._objects}
        if len(counts) != 1:
            raise RuntimeError("all envs must contain the same number of objects")

    def _validate_visual_batch_slot(self, obj_id: int):
        first = self._objects[0][int(obj_id)]
        first_type = str(getattr(first.obj_type, "name", first.obj_type)).lower()
        for env_id, env_objs in enumerate(self._objects):
            obj = env_objs[int(obj_id)]
            obj_type = str(getattr(obj.obj_type, "name", obj.obj_type)).lower()
            if obj.is_visual != first.is_visual:
                raise RuntimeError(
                    f"object obj={obj_id} mixes visual and simulated registrations"
                )
            if obj_type != first_type:
                raise RuntimeError(
                    f"object obj={obj_id} mixes object types: env 0 has {first_type!r}, env {env_id} has {obj_type!r}"
                )
            if obj.asset_file != first.asset_file:
                raise RuntimeError(
                    f"object obj={obj_id} mixes assets: env 0 has "
                    f"{first.asset_file!r}, env {env_id} has {obj.asset_file!r}"
                )

    def _visual_batch_color(self, obj_id: int):
        colors = [obj[int(obj_id)].color for obj in self._objects]
        if all(color is None for color in colors):
            return None
        rows = []
        for color in colors:
            if color is None:
                rows.append(np.asarray([0.15, 0.15, 0.15, 1.0], dtype=np.float32))
                continue
            arr = np.asarray(color, dtype=np.float32).reshape(-1)
            if arr.size == 1:
                arr = np.repeat(arr, 3)
            if arr.size == 3:
                arr = np.concatenate([arr, np.ones(1, dtype=np.float32)])
            if arr.size < 4:
                raise ValueError(f"color must have 1, 3, or 4 values; got {arr.size}")
            rows.append(np.clip(arr[:4], 0.0, 1.0).astype(np.float32, copy=False))
        return np.stack(rows, axis=0)

    def _apply_drive_config(self, env_id, obj_id):
        record = self._world.state.record(env_id, obj_id)
        cache = record.cache
        articulation = record.articulation
        dof_names = cache.dof_names
        kp = self._drive_param("kp", cache.dof_kps, dof_names, 200.0)
        kd = self._drive_param("kd", cache.dof_kds, dof_names, 10.0)
        effort = self._drive_param("effort_limit", None, dof_names, np.inf)
        articulation.set_kps(kp.tolist())
        articulation.set_kds(kd.tolist())
        articulation.set_effort_limits(effort.tolist())
        cache.refresh_metadata()

    def _drive_param(self, key, base, dof_names, fallback_scalar):
        value = self._config.get(key)
        if base is None:
            base = np.full(len(dof_names), fallback_scalar, dtype=np.float32)
        else:
            base = self._as_numpy(base).reshape(-1)
        if value is None:
            if np.any(np.abs(base) > 0):
                return base.copy()
            return np.full(base.shape, fallback_scalar, dtype=np.float32)
        return self._drive_param_array(value, dof_names, base)

    def _drive_param_array(self, value, dof_names, base):
        out = self._as_numpy(base).copy()
        if isinstance(value, str):
            value = _parse_drive_string(value)
        if isinstance(value, dict):
            for name, v in value.items():
                try:
                    idx = dof_names.index(str(name))
                except ValueError as exc:
                    raise KeyError(f"unknown DOF name in drive config: {name}") from exc
                out[idx] = float(v)
            return out
        arr = self._as_numpy(value).reshape(-1)
        if arr.size == 1:
            out.fill(float(arr[0]))
            return out
        if arr.size != len(dof_names):
            raise ValueError(
                f"drive config expected scalar, {len(dof_names)} values, or name->value dict; got {arr.size} values"
            )
        return arr.astype(np.float32, copy=True)

    def _use_gpu_logical_state(self) -> bool:
        return getattr(self._world.state, "canonical_source", "cpu") == "gpu"

    def _state_root_pos(self, obj_id: int):
        if self._is_static_obj(obj_id):
            return self._static_root_batch(obj_id, rotation=False)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_root_pos(obj_id, fetch=False)
        return self._world.state.get_root_pos(obj_id)

    def _state_root_rot(self, obj_id: int):
        if self._is_static_obj(obj_id):
            return self._static_root_batch(obj_id, rotation=True)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_root_rot(obj_id, fetch=False)
        return self._world.state.get_root_rot(obj_id)

    def _state_root_vel(self, obj_id: int):
        if self._is_static_obj(obj_id):
            return torch.zeros((self._num_envs, 3), device=self._device)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_root_vel(obj_id, fetch=False)
        return self._world.state.get_root_vel(obj_id)

    def _state_root_ang_vel(self, obj_id: int):
        if self._is_static_obj(obj_id):
            return torch.zeros((self._num_envs, 3), device=self._device)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_root_ang_vel(obj_id, fetch=False)
        return self._world.state.get_root_ang_vel(obj_id)

    def _state_body_pos(self, obj_id: int):
        if self._is_static_obj(obj_id):
            return self._static_body_pose_batch(obj_id, rotation=False)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_body_pos(obj_id, fetch=False)
        return self._world.state.get_body_pos(obj_id)

    def _state_body_rot(self, obj_id: int):
        if self._is_static_obj(obj_id):
            return self._static_body_pose_batch(obj_id, rotation=True)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_body_rot(obj_id, fetch=False)
        return self._world.state.get_body_rot(obj_id)

    def _state_body_vel(self, obj_id: int):
        if self._is_static_obj(obj_id):
            shape = (self._num_envs, self.get_obj_num_bodies(obj_id), 3)
            return torch.zeros(shape, device=self._device)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_body_vel(obj_id, fetch=False)
        return self._world.state.get_body_vel(obj_id)

    def _state_body_ang_vel(self, obj_id: int):
        if self._is_static_obj(obj_id):
            shape = (self._num_envs, self.get_obj_num_bodies(obj_id), 3)
            return torch.zeros(shape, device=self._device)
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_body_ang_vel(obj_id, fetch=False)
        return self._world.state.get_body_ang_vel(obj_id)

    def _state_dof_pos(self, obj_id: int):
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_dof_pos(obj_id, fetch=False)
        return self._world.state.get_dof_pos(obj_id)

    def _state_dof_vel(self, obj_id: int):
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_dof_vel(obj_id, fetch=False)
        return self._world.state.get_dof_vel(obj_id)

    def _state_dof_forces(self, obj_id: int):
        if self._use_gpu_logical_state():
            return self._world.state.gpu.get_dof_forces(obj_id, fetch=False)
        return self._world.state.get_dof_forces(obj_id)

    def _pending_root_batch(
        self,
        obj_id: int,
        component: RootComponent,
        values: ArrayLike,
    ) -> ArrayLike:
        buffers = self._require_pending_state()
        pending_values = self._root_pending_storage(component)
        oid = int(obj_id)
        mask = buffers.root_dirty[:, oid]
        if not bool(np.any(mask)):
            return values
        out = self._to_tensor(values).clone()
        tensor_mask = torch.as_tensor(mask, dtype=torch.bool, device=out.device)
        pending = pending_values[:, oid].to(device=out.device, dtype=out.dtype)
        out[tensor_mask] = pending[tensor_mask]
        return out

    def _pending_dof_batch(
        self,
        obj_id: int,
        component: DofComponent,
        values: ArrayLike,
    ) -> ArrayLike:
        buffers = self._require_pending_state()
        pending_values = self._dof_pending_storage(component)
        oid = int(obj_id)
        width = self.get_obj_num_dofs(obj_id)
        mask = buffers.dof_dirty[:, oid]
        if not bool(np.any(mask)):
            return values
        out = self._to_tensor(values).clone()
        tensor_mask = torch.as_tensor(mask, dtype=torch.bool, device=out.device)
        pending = pending_values[:, oid, :width].to(device=out.device, dtype=out.dtype)
        out[tensor_mask] = pending[tensor_mask]
        return out

    def _set_visual_body_override(
        self,
        storage: dict[tuple[int, int], np.ndarray],
        env_id: EnvIdLike,
        obj_id: int,
        value: ArrayLike,
        width: int,
    ) -> None:
        # Normalize MimicKit scalar, per-env, or all-env body tensors into a
        # single [num_bodies, width] array per env/object for render-time use.
        num_bodies = self.get_obj_num_bodies(obj_id)
        env_ids = self._env_ids(env_id)
        for local_id, eid in enumerate(env_ids):
            selected = self._select_body_vector_value(
                value, eid, num_bodies, width, local_id, len(env_ids)
            )
            if width == 3:
                selected = self._world_pos(eid, selected)
            storage[(eid, int(obj_id))] = np.asarray(
                selected,
                dtype=np.float32,
            ).reshape(num_bodies, width)

    def _set_body_velocity_override(
        self, kind: BodyVelocityKind, env_id: EnvIdLike, obj_id: int, value: ArrayLike
    ) -> None:
        # PhysX reduced articulations do not support arbitrary per-link velocity
        # assignment. MimicKit still expects this setter to update the engine
        # state tensor around resets/reference characters, so expose it as a
        # MimicKit-facing transient cache. The cache is cleared after a real
        # physics step for simulated objects.
        obj_id = int(obj_id)
        num_bodies = self.get_obj_num_bodies(obj_id)
        values, mask = self._body_velocity_override_storage(kind)
        env_ids = self._env_ids(env_id)
        env_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=values.device)
        selected = self._select_body_vector_batch(value, env_ids, num_bodies, 3)
        selected = selected.to(device=values.device, dtype=values.dtype)
        values[env_tensor, obj_id, :num_bodies] = selected
        mask[np.asarray(env_ids, dtype=np.int64), obj_id] = True

    def _init_body_velocity_override_buffers(self):
        objs_per_env = self.get_objs_per_env()
        if objs_per_env == 0:
            max_bodies = 0
            body_counts = np.zeros(0, dtype=np.int32)
            visual_mask = np.zeros(0, dtype=bool)
        else:
            body_counts = np.asarray(
                [self.get_obj_num_bodies(obj_id) for obj_id in range(objs_per_env)],
                dtype=np.int32,
            )
            visual_mask = np.asarray(
                [self._is_visual_obj(obj_id) for obj_id in range(objs_per_env)],
                dtype=bool,
            )
            max_bodies = int(np.max(body_counts))
        shape = (self._num_envs, objs_per_env, max_bodies, 3)
        mask_shape = (self._num_envs, objs_per_env)
        self._body_velocity_overrides = _BodyVelocityOverrideBuffers(
            num_bodies=body_counts,
            visual_mask=visual_mask,
            vel_values=torch.zeros(shape, dtype=torch.float32, device=self._device),
            vel_masks=np.zeros(mask_shape, dtype=bool),
            ang_vel_values=torch.zeros(shape, dtype=torch.float32, device=self._device),
            ang_vel_masks=np.zeros(mask_shape, dtype=bool),
        )

    def _body_velocity_override_storage(
        self, kind: BodyVelocityKind
    ) -> tuple[torch.Tensor, np.ndarray]:
        buffers = self._require_body_velocity_overrides()
        if kind == "vel":
            return buffers.vel_values, buffers.vel_masks
        if kind == "ang_vel":
            return buffers.ang_vel_values, buffers.ang_vel_masks
        raise ValueError(f"unsupported body velocity override kind: {kind}")

    def _require_body_velocity_overrides(self) -> _BodyVelocityOverrideBuffers:
        if self._body_velocity_overrides is None:
            raise RuntimeError(
                "body velocity override buffers are not initialized; "
                "call initialize_sim() before querying or mutating simulated state"
            )
        return self._body_velocity_overrides

    def _body_velocity_override_batch(
        self, kind: BodyVelocityKind, obj_id: int, base: ArrayLike | None = None
    ) -> np.ndarray:
        obj_id = int(obj_id)
        num_bodies = self.get_obj_num_bodies(obj_id)
        values, masks = self._body_velocity_override_storage(kind)
        mask = masks[:, obj_id]
        if not bool(np.any(mask)):
            if base is None:
                return torch.zeros(
                    (self._num_envs, num_bodies, 3),
                    dtype=torch.float32,
                    device=self._device,
                )
            return base
        if base is None:
            out = torch.zeros(
                (self._num_envs, num_bodies, 3),
                dtype=torch.float32,
                device=self._device,
            )
        else:
            out = self._to_tensor(base).clone()
        tensor_mask = torch.as_tensor(mask, dtype=torch.bool, device=out.device)
        pending = values[:, obj_id, :num_bodies].to(device=out.device, dtype=out.dtype)
        out[tensor_mask] = pending[tensor_mask]
        return out

    def _init_state_pending_buffers(self):
        objs_per_env = self.get_objs_per_env()
        if objs_per_env == 0:
            max_dofs = 0
            dof_widths = np.zeros(0, dtype=np.int32)
        else:
            dof_widths = np.asarray(
                [self.get_obj_num_dofs(obj_id) for obj_id in range(objs_per_env)],
                dtype=np.int32,
            )
            max_dofs = int(np.max(dof_widths)) if dof_widths.size else 0

        root_mask_shape = (self._num_envs, objs_per_env)
        self._pending_state = _PendingStateBuffers(
            root_state=torch.zeros(
                (self._num_envs, objs_per_env, 13),
                dtype=torch.float32,
                device=self._device,
            ),
            root_dirty=np.zeros(root_mask_shape, dtype=bool),
            dof_state=torch.zeros(
                (self._num_envs, objs_per_env, max_dofs, 2),
                dtype=torch.float32,
                device=self._device,
            ),
            dof_dirty=np.zeros(root_mask_shape, dtype=bool),
            dof_widths=dof_widths,
        )

    def _root_pending_storage(self, component: RootComponent) -> torch.Tensor:
        state = self._require_pending_state().root_state
        slices = {
            "pos": slice(0, 3),
            "rot": slice(3, 7),
            "vel": slice(7, 10),
            "ang_vel": slice(10, 13),
        }
        return state[..., slices[component]]

    def _dof_pending_storage(self, component: DofComponent) -> torch.Tensor:
        state = self._require_pending_state().dof_state
        return state[..., 0 if component == "pos" else 1]

    def _require_pending_state(self) -> _PendingStateBuffers:
        if self._pending_state is None:
            raise RuntimeError(
                "pending state buffers are not initialized; call initialize_sim() "
                "before querying or mutating simulated state"
            )
        return self._pending_state

    def _stage_root_component(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        component: RootComponent,
        value: ArrayLike,
    ) -> None:
        buffers = self._require_pending_state()
        env_ids = self._env_ids(env_id)
        oid = int(obj_id)
        self._seed_staged_root(env_ids, oid)

        width = 4 if component == "rot" else 3
        selected = (
            self._world_pos_batch_for_env_ids(value, env_ids)
            if component == "pos"
            else self._select_flat_batch_value(value, env_ids, width)
        )
        target = self._root_pending_storage(component)
        env_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=target.device)
        target[env_tensor, oid] = torch.as_tensor(
            selected, dtype=target.dtype, device=target.device
        ).reshape(len(env_ids), width)
        buffers.root_dirty[np.asarray(env_ids, dtype=np.int64), oid] = True

    def _stage_dof_component(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        component: DofComponent,
        value: ArrayLike,
    ) -> None:
        buffers = self._require_pending_state()
        env_ids = self._env_ids(env_id)
        oid = int(obj_id)
        width = int(buffers.dof_widths[oid])
        self._seed_staged_dof(env_ids, oid, width)

        target = self._dof_pending_storage(component)
        selected = self._select_flat_batch_value(value, env_ids, width)
        env_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=target.device)
        target[env_tensor, oid, :width] = torch.as_tensor(
            selected, dtype=target.dtype, device=target.device
        ).reshape(len(env_ids), width)
        buffers.dof_dirty[np.asarray(env_ids, dtype=np.int64), oid] = True

    def _seed_staged_root(self, env_ids: list[int], obj_id: int) -> None:
        buffers = self._require_pending_state()
        missing = [eid for eid in env_ids if not buffers.root_dirty[eid, obj_id]]
        if not missing:
            return
        dst_index = torch.as_tensor(
            missing, dtype=torch.long, device=buffers.root_state.device
        )
        for target, current in (
            (self._root_pending_storage("pos"), self._state_root_pos(obj_id)),
            (self._root_pending_storage("rot"), self._state_root_rot(obj_id)),
            (self._root_pending_storage("vel"), self._state_root_vel(obj_id)),
            (
                self._root_pending_storage("ang_vel"),
                self._state_root_ang_vel(obj_id),
            ),
        ):
            source = torch.as_tensor(current, dtype=target.dtype, device=target.device)
            target[dst_index, obj_id] = source[dst_index]

    def _seed_staged_dof(self, env_ids: list[int], obj_id: int, width: int) -> None:
        buffers = self._require_pending_state()
        missing = [eid for eid in env_ids if not buffers.dof_dirty[eid, obj_id]]
        if not missing:
            return
        dst_index = torch.as_tensor(
            missing, dtype=torch.long, device=buffers.dof_state.device
        )
        for target, current in (
            (self._dof_pending_storage("pos"), self._state_dof_pos(obj_id)),
            (self._dof_pending_storage("vel"), self._state_dof_vel(obj_id)),
        ):
            source = torch.as_tensor(current, dtype=target.dtype, device=target.device)
            target[dst_index, obj_id, :width] = source[dst_index, :width]

    def _flush_staged_state(self) -> None:
        """Queue completed root/DOF resets once, just before the physics step.

        Staging combines those writes so each dirty object is submitted as one CPU or GPU batch.
        """
        buffers = self._pending_state
        if buffers is None:
            return
        for obj_id in np.flatnonzero(np.any(buffers.root_dirty, axis=0)):
            env_ids = np.flatnonzero(buffers.root_dirty[:, obj_id]).tolist()
            index = torch.as_tensor(
                env_ids, dtype=torch.long, device=buffers.root_state.device
            )
            state = buffers.root_state[index, obj_id].contiguous()
            if state.device.type == "cuda":
                self._world.set_gpu_root_state_batch(env_ids, int(obj_id), state)
            else:
                self._world.set_root_state(
                    env_ids,
                    int(obj_id),
                    state[:, 0:3].contiguous(),
                    state[:, 3:7].contiguous(),
                    state[:, 7:10].contiguous(),
                    state[:, 10:13].contiguous(),
                )

        for obj_id in np.flatnonzero(np.any(buffers.dof_dirty, axis=0)):
            env_ids = np.flatnonzero(buffers.dof_dirty[:, obj_id]).tolist()
            width = int(buffers.dof_widths[obj_id])
            index = torch.as_tensor(
                env_ids, dtype=torch.long, device=buffers.dof_state.device
            )
            state = buffers.dof_state[index, obj_id, :width].contiguous()
            if state.device.type == "cuda":
                self._world.set_gpu_dof_state_batch(env_ids, int(obj_id), state)
            else:
                self._world.set_dof_state(
                    env_ids,
                    int(obj_id),
                    state[..., 0].contiguous(),
                    state[..., 1].contiguous(),
                )

    def _clear_state_pending_overrides(self) -> None:
        buffers = self._pending_state
        if buffers is None:
            return
        buffers.root_dirty[:, :] = False
        buffers.dof_dirty[:, :] = False

    def _visual_root_batch(self, storage, obj_id, width):
        default = (
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if width == 4
            else np.zeros(width, dtype=np.float32)
        )
        out = np.repeat(default.reshape(1, width), self._num_envs, axis=0)
        for env_id in range(self._num_envs):
            value = storage.get((env_id, int(obj_id)))
            if value is not None:
                out[env_id] = value
        if width == 3:
            out = self._local_pos_batch(out)
        return out

    def _visual_body_pose_batch(self, storage, obj_id, width):
        num_bodies = self.get_obj_num_bodies(obj_id)
        if width == 3:
            root_pos = self._visual_root_batch(self._visual_root_pos, obj_id, 3)
            out = np.repeat(root_pos[:, None, :], num_bodies, axis=1)
            out = self._world_pos_batch(out)
            for env_id in range(self._num_envs):
                value = storage.get((env_id, int(obj_id)))
                if value is not None:
                    out[env_id] = value
            return out
        else:
            default = np.zeros((num_bodies, width), dtype=np.float32)
            default[:, 3] = 1.0
        out = np.repeat(default[None, :, :], self._num_envs, axis=0)
        for env_id in range(self._num_envs):
            value = storage.get((env_id, int(obj_id)))
            if value is not None:
                out[env_id] = value
        return out

    def _pending_body_pose_batch(self, storage, obj_id, values):
        keys = [
            (env_id, int(obj_id))
            for env_id in range(self._num_envs)
            if (env_id, int(obj_id)) in storage
        ]
        if not keys:
            return values
        if torch.is_tensor(values):
            out = values.clone()
            for env_id, key_obj_id in keys:
                out[env_id] = torch.as_tensor(
                    storage[(env_id, key_obj_id)],
                    dtype=out.dtype,
                    device=out.device,
                )
            return out
        out = np.asarray(values, dtype=np.float32).copy()
        for env_id, key_obj_id in keys:
            out[env_id] = storage[(env_id, key_obj_id)]
        return out

    def _clear_dynamic_body_velocity_overrides(self):
        # Keep visual/reference object overrides, but let simulated objects report
        # PhysX velocities after the next physics step.
        buffers = self._body_velocity_overrides
        if buffers is None:
            return
        sim_obj_mask = ~buffers.visual_mask
        buffers.vel_masks[:, sim_obj_mask] = False
        buffers.ang_vel_masks[:, sim_obj_mask] = False

    def _clear_dynamic_visual_body_overrides(self):
        # Simulated objects may receive set_body_pos/rot during reset or
        # view_motion playback. After a real physics step, their rendered pose
        # must go back to PhysicsBridge unless MimicKit publishes a fresh body
        # pose later in the same step. Visual/reference objects keep overrides.
        for storage in (self._visual_body_pos, self._visual_body_rot):
            for key in list(storage):
                _, obj_id = key
                if not self._is_visual_obj(obj_id):
                    del storage[key]

    def _set_visual_root_override(
        self,
        storage: dict[tuple[int, int], np.ndarray],
        env_id: EnvIdLike,
        obj_id: int,
        value: ArrayLike,
        width: int,
    ) -> None:
        # Visual-only reference characters do not exist in PhysX. Keep their
        # root values only as a fallback until MimicKit publishes body poses.
        arr = self._as_numpy(value)
        if arr.ndim == 0:
            arr = np.full(width, float(arr), dtype=np.float32)
        env_ids = self._env_ids(env_id)
        for local_id, eid in enumerate(env_ids):
            selected = self._select_flat_value(arr, eid, width, local_id, len(env_ids))
            if width == 3:
                selected = self._world_pos(eid, selected)
            storage[(eid, int(obj_id))] = np.asarray(
                selected,
                dtype=np.float32,
            ).reshape(width)

    def _apply_visual_body_overrides(self):
        # Push cached body transforms into scene prims. This is intentionally
        # viewer-only and is used to match MimicKit reference motion playback.
        if self._viewer is None or self._viewer.visual_bridge is None:
            return
        keys = set(self._visual_body_pos) | set(self._visual_body_rot)
        for env_id, obj_id in keys:
            self._viewer.visual_bridge.set_body_transforms_scene_graph(
                env_id,
                obj_id,
                self._visual_body_pos.get((env_id, obj_id)),
                self._visual_body_rot.get((env_id, obj_id)),
            )

    def _apply_visual_root_overrides(self, obj_id=None):
        # Visual-only objects are not simulated, so root setters must drive the
        # viewer directly when MimicKit has not provided explicit body poses.
        if self._viewer is None or self._viewer.visual_bridge is None:
            return
        keys = set(self._visual_root_pos) | set(self._visual_root_rot)
        if obj_id is not None:
            keys = {key for key in keys if key[1] == int(obj_id)}
        for env_id, oid in keys:
            if (env_id, oid) in self._visual_body_pos or (
                env_id,
                oid,
            ) in self._visual_body_rot:
                continue
            self._viewer.visual_bridge.set_root_transform_scene_graph(
                env_id,
                oid,
                self._visual_root_pos.get((env_id, oid)),
                self._visual_root_rot.get((env_id, oid)),
            )

    def _debug_visual_body_pose(self, obj_id):
        if not self._debug_visual_pose:
            return
        obj_id = int(obj_id)
        for env_id in range(self._num_envs):
            key = (env_id, obj_id)
            pos = self._visual_body_pos.get(key)
            if pos is None:
                continue
            root = np.asarray(pos[0], dtype=np.float32).copy()
            stat = self._debug_visual_pose_stats.get(key)
            if stat is None:
                self._debug_visual_pose_stats[key] = {
                    "count": 1,
                    "last": root,
                    "max_delta": 0.0,
                }
                print(
                    f"[kangengine visual pose] env={env_id} obj={obj_id} "
                    f"kind={self._visual_debug_kind(obj_id)} "
                    f"count=1 root={root.tolist()} max_delta=0.000000"
                )
                continue
            delta = float(np.linalg.norm(root - stat["last"]))
            stat["count"] += 1
            stat["max_delta"] = max(float(stat["max_delta"]), delta)
            stat["last"] = root
            if stat["count"] <= 5 or stat["count"] % 30 == 0:
                print(
                    f"[kangengine visual pose] env={env_id} obj={obj_id} "
                    f"kind={self._visual_debug_kind(obj_id)} "
                    f"count={stat['count']} root={root.tolist()} "
                    f"delta={delta:.6f} max_delta={stat['max_delta']:.6f}"
                )

    def _color_override(self, key):
        value = self._config.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            value = [float(part.strip()) for part in value.split(",") if part.strip()]
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _viewer_color(self, is_visual, color):
        override = self._ref_color_override if is_visual else self._sim_color_override
        if override is not None:
            return override
        return color

    def _visual_debug_kind(self, obj_id):
        return "visual" if self._is_visual_obj(obj_id) else "sim"

    def _is_visual_obj(self, obj_id):
        obj_id = int(obj_id)
        return bool(self._objects and self._objects[0][obj_id].is_visual)

    def _is_static_obj(self, obj_id):
        obj_id = int(obj_id)
        if not self._objects:
            return False
        obj = self._objects[0][obj_id]
        return not obj.is_visual and obj.is_static

    def _static_root_batch(self, obj_id, *, rotation):
        getter = "get_root_rotation" if rotation else "get_root_position"
        values = [
            np.asarray(
                getattr(self._world.rigid(env_id, obj_id), getter)(),
                dtype=np.float32,
            )
            for env_id in range(self._num_envs)
        ]
        return torch.as_tensor(np.stack(values), device=self._device)

    def _static_body_pose_batch(self, obj_id, *, rotation):
        specs = rigid_shape_specs(self._objects[0][obj_id].data)
        local_pos = np.stack([spec.local_pos for spec in specs], axis=0)
        local_rot = np.stack([spec.local_rot for spec in specs], axis=0)
        positions = []
        rotations = []
        for env_id in range(self._num_envs):
            rigid = self._world.rigid(env_id, obj_id)
            body_pos, body_rot = expand_rigid_body_state(
                np.asarray(rigid.get_root_position(), dtype=np.float32),
                np.asarray(rigid.get_root_rotation(), dtype=np.float32),
                local_pos,
                local_rot,
            )
            positions.append(body_pos)
            rotations.append(body_rot)
        values = rotations if rotation else positions
        return torch.as_tensor(np.stack(values), device=self._device)

    def _select_flat_batch_value(
        self,
        value: ArrayLike,
        env_ids: list[int],
        width: int,
    ) -> ArrayLike:
        env_count = len(env_ids)
        if torch.is_tensor(value):
            arr = value
            if arr.ndim == 0:
                return torch.full(
                    (env_count, width),
                    float(arr.item()),
                    dtype=torch.float32,
                    device=arr.device,
                )
            if arr.ndim == 1:
                if arr.shape[0] == width:
                    if env_count == 1:
                        return arr.reshape(width).contiguous()
                    return arr.reshape(1, width).expand(env_count, width).contiguous()
                if width == 1:
                    if arr.shape[0] == self._num_envs:
                        return arr[env_ids].reshape(env_count, 1).contiguous()
                    if arr.shape[0] == env_count:
                        return arr.reshape(env_count, 1).contiguous()
            if arr.ndim >= 2:
                if arr.shape[0] == self._num_envs:
                    return arr[env_ids].reshape(env_count, width).contiguous()
                if arr.shape[0] == env_count:
                    return arr.reshape(env_count, width).contiguous()
            if arr.numel() == width:
                if env_count == 1:
                    return arr.reshape(width).contiguous()
                return arr.reshape(1, width).expand(env_count, width).contiguous()
            if arr.numel() == env_count * width:
                return arr.reshape(env_count, width).contiguous()
            raise ValueError(
                f"Cannot select tensor with shape {list(arr.shape)} as {env_count} values of width {width}"
            )

        arr = self._as_numpy(value)
        if arr.ndim == 0:
            return np.full((env_count, width), float(arr), dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] == width:
                if env_count == 1:
                    return np.asarray(arr, dtype=np.float32).reshape(width)
                return np.repeat(arr.reshape(1, width), env_count, axis=0).astype(
                    np.float32,
                    copy=False,
                )
            if width == 1:
                if arr.shape[0] == self._num_envs:
                    return np.asarray(arr[env_ids], dtype=np.float32).reshape(
                        env_count, 1
                    )
                if arr.shape[0] == env_count:
                    return np.asarray(arr, dtype=np.float32).reshape(env_count, 1)
        if arr.ndim >= 2:
            if arr.shape[0] == self._num_envs:
                return np.asarray(arr[env_ids], dtype=np.float32).reshape(
                    env_count, width
                )
            if arr.shape[0] == env_count:
                return np.asarray(arr, dtype=np.float32).reshape(env_count, width)
        try:
            flat = np.asarray(arr, dtype=np.float32)
            if flat.size == width:
                if env_count == 1:
                    return flat.reshape(width)
                return np.repeat(flat.reshape(1, width), env_count, axis=0)
            return flat.reshape(env_count, width)
        except ValueError as exc:
            raise ValueError(
                f"Cannot select value with shape {arr.shape} as {env_count} values of width {width}"
            ) from exc

    def _world_pos_batch_for_env_ids(
        self,
        value: ArrayLike,
        env_ids: list[int],
    ) -> ArrayLike:
        positions = self._select_flat_batch_value(value, env_ids, 3)
        if torch.is_tensor(positions):
            env_tensor = torch.as_tensor(
                env_ids, dtype=torch.long, device=positions.device
            )
            offsets = self._env_offsets_torch.to(
                device=positions.device, dtype=positions.dtype
            )[env_tensor]
            if positions.ndim == 1:
                if len(env_ids) == 1:
                    return (positions + offsets[0]).contiguous()
                positions = positions.reshape(1, 3).expand(len(env_ids), 3)
            return (positions + offsets).contiguous()
        offsets_np = self._env_offsets[np.asarray(env_ids, dtype=np.int64)]
        offsets = np.asarray(offsets_np, dtype=np.float32)
        if np.asarray(positions).ndim == 1:
            if len(env_ids) == 1:
                return np.asarray(positions, dtype=np.float32) + offsets[0]
            positions = np.repeat(
                np.asarray(positions, dtype=np.float32).reshape(1, 3),
                len(env_ids),
                axis=0,
            )
        return np.asarray(positions, dtype=np.float32) + offsets

    def _flat_batch_to_numpy(
        self,
        value: ArrayLike,
        env_count: int,
        width: int,
    ) -> np.ndarray:
        arr = self._as_numpy(value)
        if arr.ndim == 1:
            return np.repeat(arr.reshape(1, width), env_count, axis=0).astype(
                np.float32,
                copy=False,
            )
        return np.asarray(arr, dtype=np.float32).reshape(env_count, width)

    def _select_flat_value(
        self,
        value: ArrayLike,
        env_id: int,
        width: int,
        local_id: int | None = None,
        env_count: int | None = None,
    ) -> np.ndarray:
        arr = self._as_numpy(value)
        if arr.ndim == 0:
            return np.full(width, float(arr), dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] == width:
                return arr
            if width == 1:
                if arr.shape[0] == self._num_envs:
                    return np.asarray([arr[int(env_id)]], dtype=np.float32)
                if (
                    local_id is not None
                    and env_count is not None
                    and arr.shape[0] == env_count
                ):
                    return np.asarray([arr[int(local_id)]], dtype=np.float32)
        if arr.ndim >= 2:
            if arr.shape[0] == self._num_envs:
                return np.asarray(arr[int(env_id)], dtype=np.float32).reshape(width)
            if (
                local_id is not None
                and env_count is not None
                and arr.shape[0] == env_count
            ):
                return np.asarray(arr[int(local_id)], dtype=np.float32).reshape(width)
        try:
            return np.asarray(arr, dtype=np.float32).reshape(width)
        except ValueError as exc:
            raise ValueError(
                f"Cannot select value with shape {arr.shape} as width {width}"
            ) from exc

    def _select_body_vector_value(
        self,
        value: ArrayLike,
        env_id: int,
        num_bodies: int,
        width: int,
        local_id: int | None = None,
        env_count: int | None = None,
    ) -> np.ndarray:
        arr = self._as_numpy(value)
        if arr.ndim == 0:
            return np.full((num_bodies, width), float(arr), dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] == width:
                return np.repeat(arr.reshape(1, width), num_bodies, axis=0)
            if arr.shape[0] == num_bodies * width:
                return arr.reshape(num_bodies, width)
        if arr.ndim == 2:
            if arr.shape == (num_bodies, width):
                return arr
            if arr.shape == (self._num_envs, width):
                return np.repeat(arr[int(env_id)].reshape(1, width), num_bodies, axis=0)
            if (
                local_id is not None
                and env_count is not None
                and arr.shape == (env_count, width)
            ):
                return np.repeat(
                    arr[int(local_id)].reshape(1, width), num_bodies, axis=0
                )
            if arr.shape[0] == self._num_envs:
                return self._select_body_vector_value(
                    arr[int(env_id)], env_id, num_bodies, width
                )
            if (
                local_id is not None
                and env_count is not None
                and arr.shape[0] == env_count
            ):
                return self._select_body_vector_value(
                    arr[int(local_id)], env_id, num_bodies, width
                )
        if arr.ndim >= 3:
            if arr.shape[0] == self._num_envs:
                return np.asarray(arr[int(env_id)], dtype=np.float32).reshape(
                    num_bodies, width
                )
            if (
                local_id is not None
                and env_count is not None
                and arr.shape[0] == env_count
            ):
                return np.asarray(arr[int(local_id)], dtype=np.float32).reshape(
                    num_bodies, width
                )
        return np.asarray(arr, dtype=np.float32).reshape(num_bodies, width)

    def _select_body_vector_batch(
        self,
        value: ArrayLike,
        env_ids: list[int],
        num_bodies: int,
        width: int,
    ) -> torch.Tensor:
        env_count = len(env_ids)
        if torch.is_tensor(value):
            arr = value.to(dtype=torch.float32)
            if arr.ndim == 0:
                return torch.full(
                    (env_count, num_bodies, width),
                    float(arr.item()),
                    dtype=torch.float32,
                    device=arr.device,
                )
            if arr.ndim == 1:
                if arr.shape[0] == width:
                    return (
                        arr.reshape(1, 1, width)
                        .expand(env_count, num_bodies, width)
                        .contiguous()
                    )
                if arr.numel() == num_bodies * width:
                    return (
                        arr.reshape(1, num_bodies, width)
                        .expand(env_count, num_bodies, width)
                        .contiguous()
                    )
            if arr.ndim == 2:
                if tuple(arr.shape) == (num_bodies, width):
                    return (
                        arr.reshape(1, num_bodies, width)
                        .expand(env_count, num_bodies, width)
                        .contiguous()
                    )
                if tuple(arr.shape) == (self._num_envs, width):
                    return (
                        arr[env_ids]
                        .reshape(env_count, 1, width)
                        .expand(env_count, num_bodies, width)
                        .contiguous()
                    )
                if tuple(arr.shape) == (env_count, width):
                    return (
                        arr.reshape(env_count, 1, width)
                        .expand(env_count, num_bodies, width)
                        .contiguous()
                    )
            if arr.ndim >= 3:
                if arr.shape[0] == self._num_envs:
                    return (
                        arr[env_ids].reshape(env_count, num_bodies, width).contiguous()
                    )
                if arr.shape[0] == env_count:
                    return arr.reshape(env_count, num_bodies, width).contiguous()
            return arr.reshape(env_count, num_bodies, width).contiguous()

        selected = [
            self._select_body_vector_value(
                value, eid, num_bodies, width, local_id, env_count
            )
            for local_id, eid in enumerate(env_ids)
        ]
        return torch.as_tensor(
            np.stack(selected, axis=0),
            dtype=torch.float32,
            device=self._device,
        )

    def _env_offset(self, env_id):
        return self._env_offsets[int(env_id)]

    def _world_pos(self, env_id: int, value: ArrayLike) -> np.ndarray:
        arr = self._as_numpy(value)
        return arr + self._env_offset(env_id)

    def _world_pos_batch(self, values: ArrayLike) -> np.ndarray:
        if torch.is_tensor(values):
            arr = values.clone()
            if arr.ndim >= 2 and arr.shape[0] == self._num_envs:
                view_shape = (self._num_envs,) + (1,) * (arr.ndim - 2) + (3,)
                offsets = self._env_offsets_torch.to(
                    device=arr.device, dtype=arr.dtype
                ).reshape(view_shape)
                arr = arr + offsets
            return arr
        arr = self._as_numpy(values).copy()
        if arr.ndim >= 2 and arr.shape[0] == self._num_envs:
            view_shape = (self._num_envs,) + (1,) * (arr.ndim - 2) + (3,)
            arr += self._env_offsets.reshape(view_shape)
        return arr

    def _local_pos(self, env_id: int, value: ArrayLike) -> np.ndarray:
        arr = self._as_numpy(value)
        return arr - self._env_offset(env_id)

    def _local_pos_batch(self, values: ArrayLike) -> np.ndarray:
        if torch.is_tensor(values):
            arr = values.clone()
            if arr.ndim >= 2 and arr.shape[0] == self._num_envs:
                view_shape = (self._num_envs,) + (1,) * (arr.ndim - 2) + (3,)
                offsets = self._env_offsets_torch.to(
                    device=arr.device, dtype=arr.dtype
                ).reshape(view_shape)
                arr = arr - offsets
            return arr
        arr = self._as_numpy(values).copy()
        if arr.ndim >= 2 and arr.shape[0] == self._num_envs:
            view_shape = (self._num_envs,) + (1,) * (arr.ndim - 2) + (3,)
            arr -= self._env_offsets.reshape(view_shape)
        return arr

    def _env_ids(self, env_id: EnvIdLike) -> list[int]:
        return env_id_list(env_id, self._num_envs)

    def _as_numpy(self, value: ArrayLike) -> np.ndarray:
        return as_cpu_numpy(value)

    def _to_tensor(self, value: ArrayLike, *, device=None) -> torch.Tensor:
        return as_tensor(value, device=self._device if device is None else device)

    def _out(self, value: ArrayLike) -> ArrayLike:
        return as_tensor(value, device=self._device)

    def _zeros_like(self, value):
        return torch.zeros_like(self._out(value))


######## install the engine to MimicKit ############
def build_engine(config, num_envs, device=None, visualize=False, record_video=False):
    """Factory with the same call shape as MimicKit's engine_builder."""

    return KangEngineEngine(
        config,
        num_envs=num_envs,
        device=device,
        visualize=visualize,
        record_video=record_video,
    )


def install_mimickit_engine_builder():
    """Patch MimicKit's engine_builder at runtime without editing MimicKit files.

    Call this before invoking ``engines.engine_builder.build_engine``. If a
    caller already imported the function directly with
    ``from engines.engine_builder import build_engine``, call this before that
    import happens.
    """

    import engines.engine_builder as engine_builder

    original = engine_builder.build_engine
    if getattr(original, "_kangengine_patched", False):
        return original

    def patched_build_engine(config, num_envs, device, visualize, record_video=False):
        if config.get("engine_name") == "kangengine":
            return build_engine(
                config,
                num_envs=num_envs,
                device=device,
                visualize=visualize,
                record_video=record_video,
            )
        return original(config, num_envs, device, visualize, record_video)

    patched_build_engine._kangengine_patched = True
    patched_build_engine._kangengine_original = original
    engine_builder.build_engine = patched_build_engine
    return patched_build_engine
