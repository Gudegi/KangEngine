"""MimicKit-style KangEngine adapter.

This module intentionally stays thin: it maps MimicKit's engine method names to
``KangSimWorld`` while KangSimWorld owns PhysX stepping, command buffers, reset
buffers, and state refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._core import _ke
from .app import App
from .rigid import rigid_body_names
from .sim import ControlMode, KangSimWorld
from .visual import KangWorldVisualBridge

try:
    import torch as _torch
except ImportError:
    _torch = None

try:
    import engines.engine as _mk_engine
except ImportError:
    _mk_engine = None

try:
    _ArticulationConfig = _ke.ArticulationConfig
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
EnvIdLike: TypeAlias = (
    int | np.integer[Any] | NDArray[np.integer[Any]] | list[int] | tuple[int, ...] | None
)
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
    fix_root: bool
    start_pos: np.ndarray
    start_rot: np.ndarray


@dataclass(slots=True)
class _BodyVelocityOverrideBuffers:
    num_bodies: np.ndarray
    visual_mask: np.ndarray
    vel_values: np.ndarray
    vel_masks: np.ndarray
    ang_vel_values: np.ndarray
    ang_vel_masks: np.ndarray


@dataclass(slots=True)
class _PendingStateBuffers:
    root_pos: np.ndarray
    root_rot: np.ndarray
    root_vel: np.ndarray
    root_ang_vel: np.ndarray
    root_pos_mask: np.ndarray
    root_rot_mask: np.ndarray
    root_vel_mask: np.ndarray
    root_ang_vel_mask: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    dof_pos_mask: np.ndarray
    dof_vel_mask: np.ndarray
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


class _FrameRecorder:
    """MimicKit-compatible RGB recorder.

    Frames are always mirrored to binary PPM files for easy debugging.  The
    in-memory frame list follows MimicKit's Video API closely enough for
    diagnostics/loggers to call ``save(path)`` and produce an mp4/gif.
    """

    def __init__(self, out_dir, fps=30):
        if out_dir is None:
            out_dir = tempfile.mkdtemp(prefix="kangengine_frames_")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fps = int(fps)
        self.frame_count = 0
        self._frames = []
        self._active = False

    def clear(self):
        self.frame_count = 0
        self._frames = []
        return

    def start(self):
        self.clear()
        self._active = True

    def stop(self):
        self._active = False

    def is_active(self):
        return self._active

    def add_frame(self, frame):
        return self.write(frame, force=True)

    def write(self, rgb, force=False):
        if not self._active:
            if not force:
                return None
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"expected RGB frame with shape [H, W, 3], got {arr.shape}")
        self._frames.append(arr.copy())
        path = self.out_dir / f"frame_{self.frame_count:06d}.ppm"
        height, width, _ = arr.shape
        with path.open("wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            f.write(np.ascontiguousarray(arr).tobytes())
        self.frame_count += 1
        return path

    def get_fps(self):
        return self._fps

    def get_num_frames(self):
        return len(self._frames)

    def get_resolution(self):
        if not self._frames:
            return (0, 0)
        frame = self._frames[0]
        return (frame.shape[0], frame.shape[1])

    def get_frames(self):
        return self._frames

    def save(self, file_path):
        if not self._frames:
            return
        path = Path(file_path)
        suffix = path.suffix.lower()
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except Exception as exc:
            raise RuntimeError(
                "Saving KangEngine MimicKit recordings requires moviepy. "
                "Install kangengine[mimickit-video] or kangengine[mimickit-full]."
            ) from exc

        clip = ImageSequenceClip(self._frames, fps=self._fps)
        if suffix == ".gif":
            clip.write_gif(str(path), logger=None)
        else:
            clip.write_videofile(str(path), logger=None)
        return


class _KangEngineViewer(App):
    def __init__(self, world, headless=False):
        super().__init__()
        self.world = world
        self.headless = bool(headless)
        self.visual_bridge = None
        self.robot_shader = None
        self.ground_shader = None
        self._setup_done = False
        self._debug_line_handles = {}

    def setup_viewer(self, width=1920, height=1080, headless=None):
        if self._setup_done:
            return
        if headless is None:
            headless = self.headless
        self.initialize(width, height, False, _ke.UpAxis.Z, headless=bool(headless))
        device = self.getGraphicsDevice()
        vs = _asset_path("shaders", "common.vs")
        fs = _asset_path("shaders", "common.fs")
        checker_fs = _asset_path("shaders", "checkerboard.fs")
        self.robot_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)
        for shader in (self.robot_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", _ke.vec4(1.0, 1.0, 1.0, 1.0))
        pg = _ke.ColorLibrary.get(_ke.ColorType.PASTEL_GREEN)
        self.ground_shader.setVec4("checkerColor2", _ke.vec4(pg.r, pg.g, pg.b, pg.a))

        ground = self.getScene().define_prim("/ground", _ke.scene.PrimType.Mesh)
        ground.set_mesh_data(_ke.scene.Prim.create_plane_data(100.0, _ke.UpAxis.Z))
        self.addShape(self.ground_shader, ground)
        self.visual_bridge = KangWorldVisualBridge(self, self.world)
        self._setup_done = True

    def add_articulation(self, env_id, obj_id, asset_file, order, color=None):
        self.setup_viewer()
        self.visual_bridge.add_articulation(
            env_id,
            obj_id,
            asset_file,
            prim_base_path=f"/env_{env_id}/obj_{obj_id}",
            order=order,
            shader=self.robot_shader,
            color=color,
        )

    def add_visual_articulation(self, env_id, obj_id, asset_file, order, color=None):
        self.setup_viewer()
        self.visual_bridge.add_visual_articulation(
            env_id,
            obj_id,
            asset_file,
            prim_base_path=f"/env_{env_id}/visual_obj_{obj_id}",
            order=order,
            shader=self.robot_shader,
            color=color,
        )

    def add_rigid(
        self,
        env_id,
        obj_id,
        asset_file,
        prim_base_path,
        order,
        shader=None,
        color=None,
    ):
        self.setup_viewer()
        self.visual_bridge.add_rigid(
            env_id,
            obj_id,
            asset_file,
            prim_base_path=prim_base_path,
            order=order,
            shader=self.robot_shader if shader is None else shader,
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
        handle = self._debug_line_handles.get(int(slot))
        if handle is None:
            handle = _ke.scene.DebugDraw.log_lines(
                self,
                self.robot_shader,
                path,
                starts,
                ends,
                colors,
                radius,
                8,
            )
            if handle == 0xFFFFFFFF:
                return
            self._debug_line_handles[int(slot)] = handle
        else:
            _ke.scene.DebugDraw.update_lines(self, handle, starts, ends, colors)

    def clear_unused_debug_lines(self, active_slots):
        for slot, handle in self._debug_line_handles.items():
            if slot >= active_slots:
                _ke.scene.DebugDraw.update_lines(self, handle, [], [], [])

    def render(self):
        pass


class KangEngineEngine(_BaseEngine):
    """MVP KangEngine backend with MimicKit-like method names."""

    def __init__(self, config, num_envs, device=None, visualize=False, record_video=False):
        super().__init__(visualize)
        self._config = dict(config)
        self._num_envs = int(num_envs)
        self._device = device
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

        self._world = KangSimWorld(
            num_envs=self._num_envs,
            sim_dt=self._sim_timestep,
            add_ground=bool(self._config.get("add_ground", True)),
        )
        self._created_envs: list[int] = []
        self._objects: list[list[KangEngineObject]] = [[] for _ in range(self._num_envs)]
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
            _FrameRecorder(record_dir, fps=record_fps)
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
        self._debug_visual_pose = (
            os.environ.get("KANGENGINE_DEBUG_VISUAL_POSE") == "1"
            or _parse_bool(self._config.get("debug_visual_pose", False))
        )
        self._debug_visual_pose_stats = {}
        self._sim_color_override = self._color_override("sim_color")
        self._ref_color_override = self._color_override("ref_color")

    def get_name(self):
        return "kangengine"

    def create_env(self, env_id=None):
        if env_id is None:
            env_id = len(self._created_envs)
        env_id = int(env_id)
        if env_id < 0 or env_id >= self._num_envs:
            raise ValueError(f"env_id {env_id} out of range for num_envs={self._num_envs}")
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
        num_dofs = 0 if is_rigid else sum(len(joints) for joints in data.joints.values())

        if not is_visual and is_articulated:
            if _ArticulationConfig is None:
                raise RuntimeError("ArticulationConfig not available: build with USE_PHYSX=ON")
            cfg = _ke.ArticulationConfig.fixed_base() if fix_root else _ke.ArticulationConfig.free_base()
            if "enable_self_collisions" in self._config:
                enable_self_collisions = _parse_bool(
                    self._config["enable_self_collisions"]
                )
            cfg.disable_self_collision = not bool(enable_self_collisions)
            cfg.contact_offset = float(self._config.get("contact_offset", 0.02))
            cfg.rest_offset = float(self._config.get("rest_offset", 0.0))
            self._world.add_articulation(data, env_id, obj_id, name, cfg)
            self._apply_drive_config(env_id, obj_id)
        elif not is_visual and is_rigid:
            self._world.add_rigid(
                data,
                env_id,
                obj_id,
                name,
                pos=self._world_pos(env_id, start_pos),
                rot_xyzw=start_rot,
                density=float(self._config.get("rigid_density", 1.0)),
                contact_offset=float(self._config.get("contact_offset", 0.02)),
                rest_offset=float(self._config.get("rest_offset", 0.0)),
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
            bool(fix_root),
            start_pos,
            start_rot,
        )
        self._objects[env_id].append(obj)
        if not is_visual:
            self._world.set_root_state(
                env_id, obj_id, self._world_pos(env_id, start_pos), start_rot
            )
        if self._viewer is not None:
            viewer_color = self._viewer_color(is_visual, color)
            if is_visual:
                self._viewer.add_visual_articulation(
                    env_id, obj_id, asset_file, mjcf_order, color=viewer_color
                )
            elif is_articulated:
                self._viewer.add_articulation(
                    env_id, obj_id, asset_file, mjcf_order, color=viewer_color
                )
            else:
                self._viewer.add_rigid(
                    env_id,
                    obj_id,
                    asset_file,
                    prim_base_path=f"/env_{env_id}/rigid_obj_{obj_id}",
                    order=mjcf_order,
                    shader=self._viewer.robot_shader,
                    color=viewer_color,
                )
        if is_visual:
            self.set_root_pos(env_id, obj_id, start_pos)
            self.set_root_rot(env_id, obj_id, start_rot)
        return obj_id

    def initialize_sim(self):
        self._validate_envs()
        for env_objs in self._objects:
            for obj in env_objs:
                if obj.is_visual:
                    continue
                self._world.set_root_state(
                    obj.env_id,
                    obj.obj_id,
                    self._world_pos(obj.env_id, obj.start_pos),
                    obj.start_rot,
                )
        self._world.step(substeps=0, apply_commands=False)
        self._init_body_velocity_override_buffers()
        self._init_state_pending_buffers()
        self._initialized = True

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
        self._world.step(substeps=self._sim_steps)
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
        camera = self._viewer.getCamera()
        p = self._as_numpy(pos).reshape(3)
        t = self._as_numpy(look_at).reshape(3)
        camera.set_camera_pos(_ke.vec3(float(p[0]), float(p[1]), float(p[2])))
        camera.set_target_pos(_ke.vec3(float(t[0]), float(t[1]), float(t[2])))

    def get_camera_pos(self):
        if self._viewer is None:
            return np.zeros(3, dtype=np.float32)
        p = self._viewer.getCamera().get_camera_pos()
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    def get_camera_dir(self):
        if self._viewer is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d = self._viewer.getCamera().get_camera_look_dir()
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
            self._recorder.start()

    def stop_video_recording(self):
        if self._recorder is not None:
            self._recorder.stop()
            self._save_video_recording()

    def get_video_recording(self):
        self._save_video_recording()
        return self._recorder

    def get_record_dir(self):
        return "" if self._recorder is None else str(self._recorder.out_dir)

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
        return self._gravity.copy()

    def get_objs_per_env(self):
        return len(self._objects[0]) if self._objects else 0

    def get_root_pos(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(self._visual_root_batch(self._visual_root_pos, obj_id, 3))
        return self._out(self._local_pos_batch(self._world.state.get_root_pos(obj_id)))

    def get_root_rot(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(self._visual_root_batch(self._visual_root_rot, obj_id, 4))
        return self._out(self._world.state.get_root_rot(obj_id))

    def get_root_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(np.zeros((self._num_envs, 3), dtype=np.float32))
        return self._out(self._world.state.get_root_vel(obj_id))

    def get_root_ang_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(np.zeros((self._num_envs, 3), dtype=np.float32))
        return self._out(self._world.state.get_root_ang_vel(obj_id))

    def get_dof_pos(self, obj_id):
        return self._out(self._world.state.get_dof_pos(obj_id))

    def get_dof_vel(self, obj_id):
        return self._out(self._world.state.get_dof_vel(obj_id))

    def get_dof_forces(self, obj_id):
        return self._out(self._world.state.get_dof_forces(obj_id))

    def get_body_pos(self, obj_id):
        if self._is_visual_obj(obj_id):
            value = self._visual_body_pose_batch(self._visual_body_pos, obj_id, 3)
            return self._out(self._local_pos_batch(value))
        return self._out(self._local_pos_batch(self._world.state.get_body_pos(obj_id)))

    def get_body_rot(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(self._visual_body_pose_batch(self._visual_body_rot, obj_id, 4))
        return self._out(self._world.state.get_body_rot(obj_id))

    def get_body_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(
                self._body_velocity_override_batch("vel", obj_id)
            )
        values = self._world.state.get_body_vel(obj_id)
        return self._out(
            self._body_velocity_override_batch("vel", obj_id, values)
        )

    def get_body_ang_vel(self, obj_id):
        if self._is_visual_obj(obj_id):
            return self._out(
                self._body_velocity_override_batch("ang_vel", obj_id)
            )
        values = self._world.state.get_body_ang_vel(obj_id)
        return self._out(
            self._body_velocity_override_batch("ang_vel", obj_id, values)
        )

    def get_contact_forces(self, obj_id):
        return self._out(self._world.state.get_contact_forces(obj_id))

    def get_ground_contact_forces(self, obj_id):
        return self._out(self._world.state.get_ground_contact_forces(obj_id))

    def set_root_pos(self, env_id: EnvIdLike, obj_id: int, root_pos: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            self._set_visual_root_override(
                self._visual_root_pos, env_id, obj_id, root_pos, 3
            )
            self._apply_visual_root_overrides(obj_id)
            return
        root_rot = self._root_component(None, obj_id, "rot")
        root_vel = self._root_component(None, obj_id, "vel")
        root_ang_vel = self._root_component(None, obj_id, "ang_vel")
        self._set_root_state(env_id, obj_id, root_pos, root_rot, root_vel, root_ang_vel)

    def set_root_rot(self, env_id: EnvIdLike, obj_id: int, root_rot: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            self._set_visual_root_override(
                self._visual_root_rot, env_id, obj_id, root_rot, 4
            )
            self._apply_visual_root_overrides(obj_id)
            return
        root_pos = self._root_component(None, obj_id, "pos")
        root_vel = self._root_component(None, obj_id, "vel")
        root_ang_vel = self._root_component(None, obj_id, "ang_vel")
        self._set_root_state(env_id, obj_id, root_pos, root_rot, root_vel, root_ang_vel)

    def set_root_vel(self, env_id: EnvIdLike, obj_id: int, root_vel: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            return
        root_pos = self._root_component(None, obj_id, "pos")
        root_rot = self._root_component(None, obj_id, "rot")
        ang_vel = self._root_component(None, obj_id, "ang_vel")
        self._set_root_state(env_id, obj_id, root_pos, root_rot, root_vel, ang_vel)

    def set_root_ang_vel(
        self, env_id: EnvIdLike, obj_id: int, root_ang_vel: ArrayLike
    ) -> None:
        if self._is_visual_obj(obj_id):
            return
        root_pos = self._root_component(None, obj_id, "pos")
        root_rot = self._root_component(None, obj_id, "rot")
        root_vel = self._root_component(None, obj_id, "vel")
        self._set_root_state(env_id, obj_id, root_pos, root_rot, root_vel, root_ang_vel)

    def set_dof_pos(self, env_id: EnvIdLike, obj_id: int, dof_pos: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            return
        dof_vel = self._dof_component(None, obj_id, "vel")
        self._set_dof_state(env_id, obj_id, dof_pos, dof_vel)

    def set_dof_vel(self, env_id: EnvIdLike, obj_id: int, dof_vel: ArrayLike) -> None:
        if self._is_visual_obj(obj_id):
            return
        dof_pos = self._dof_component(None, obj_id, "pos")
        self._set_dof_state(env_id, obj_id, dof_pos, dof_vel)

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
        # for visualization. Physics simulation state is still owned by root/DOF setters.
        if self._viewer is not None:
            self._set_visual_body_override(self._visual_body_pos, env_id, obj_id, body_pos, 3)
            self._debug_visual_body_pose(obj_id)
            self._apply_visual_body_overrides()
        return

    def set_body_rot(self, env_id: EnvIdLike, obj_id: int, body_rot: ArrayLike) -> None:
        # MimicKit body rotations are xyzw quaternions in world space. Cache them
        # for the viewer path; the physics backend does not accept direct body poses.
        if self._viewer is not None:
            self._set_visual_body_override(self._visual_body_rot, env_id, obj_id, body_rot, 4)
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
        starts = [_ke.vec3(float(v[0]), float(v[1]), float(v[2])) for v in starts_np]
        ends = [_ke.vec3(float(v[0]), float(v[1]), float(v[2])) for v in ends_np]
        colors = [
            _ke.vec4(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
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
        if self._is_visual_obj(obj_id):
            return self._objects[0][obj_id].num_dofs
        return self._world.state.get_obj_num_dofs(obj_id)

    def get_obj_num_bodies(self, obj_id):
        if self._is_visual_obj(obj_id):
            return len(self._objects[0][obj_id].body_names)
        return self._world.state.get_obj_num_bodies(obj_id)

    def get_obj_body_names(self, obj_id):
        return list(self._objects[0][obj_id].body_names)

    def find_obj_body_id(self, obj_id, body_name):
        return self.get_obj_body_names(obj_id).index(body_name)

    def get_obj_torque_limits(self, env_id, obj_id):
        if self._is_visual_obj(obj_id):
            return np.full(self.get_obj_num_dofs(obj_id), np.inf, dtype=np.float32)
        return self._world.state.get_obj_effort_limits(obj_id)

    def get_obj_dof_limits(self, env_id, obj_id):
        if self._is_visual_obj(obj_id):
            n = self.get_obj_num_dofs(obj_id)
            return -np.full(n, np.inf, dtype=np.float32), np.full(
                n, np.inf, dtype=np.float32
            )
        limits = self._world.state.get_obj_dof_limits(obj_id)
        return limits[:, 0], limits[:, 1]

    def get_obj_pd_gains(self, env_id, obj_id):
        if self._is_visual_obj(obj_id):
            n = self.get_obj_num_dofs(obj_id)
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        return self._world.state.get_obj_pd_gains(obj_id)

    def calc_obj_mass(self, env_id, obj_id):
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
            base = np.asarray(base, dtype=np.float32).reshape(-1)
        if value is None:
            if np.any(np.abs(base) > 0):
                return base.copy()
            return np.full(base.shape, fallback_scalar, dtype=np.float32)
        return self._drive_param_array(value, dof_names, base)

    def _drive_param_array(self, value, dof_names, base):
        out = np.asarray(base, dtype=np.float32).copy()
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
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            out.fill(float(arr[0]))
            return out
        if arr.size != len(dof_names):
            raise ValueError(
                f"drive config expected scalar, {len(dof_names)} values, "
                f"or name->value dict; got {arr.size} values"
            )
        return arr.astype(np.float32, copy=True)

    def _root_component(
        self, env_id: EnvIdLike, obj_id: int, component: RootComponent
    ) -> ArrayLike:
        pending_values, pending_mask = self._root_pending_storage(component)
        if self._is_single_env_id(env_id):
            eid = int(env_id)
            oid = int(obj_id)
            if pending_mask is not None and pending_mask[eid, oid]:
                value = pending_values[eid, oid]
                if component == "pos":
                    return self._local_pos(eid, value)
                return value

        state_getters = {
            "pos": self.get_root_pos,
            "rot": self.get_root_rot,
            "vel": self.get_root_vel,
            "ang_vel": self.get_root_ang_vel,
        }
        values = state_getters[component](obj_id)
        if not self._is_single_env_id(env_id):
            oid = int(obj_id)
            if pending_mask is None:
                return values
            mask = pending_mask[:, oid]
            if not bool(np.any(mask)):
                return values
            values_np = self._as_numpy(values).copy()
            if component == "pos":
                values_np[mask] = pending_values[:, oid][mask] - self._env_offsets[mask]
            else:
                values_np[mask] = pending_values[:, oid][mask]
            return self._out(values_np)
        return values[int(env_id)]

    def _dof_component(
        self, env_id: EnvIdLike, obj_id: int, component: DofComponent
    ) -> ArrayLike:
        pending_values, pending_mask = self._dof_pending_storage(component)
        width = self.get_obj_num_dofs(obj_id)
        if self._is_single_env_id(env_id):
            eid = int(env_id)
            oid = int(obj_id)
            if pending_mask is not None and pending_mask[eid, oid]:
                return pending_values[eid, oid, :width]

        values = self.get_dof_pos(obj_id) if component == "pos" else self.get_dof_vel(obj_id)
        if not self._is_single_env_id(env_id):
            oid = int(obj_id)
            if pending_mask is None:
                return values
            mask = pending_mask[:, oid]
            if not bool(np.any(mask)):
                return values
            values_np = self._as_numpy(values).copy()
            values_np[mask] = pending_values[:, oid, :width][mask]
            return self._out(values_np)
        return values[int(env_id)]

    def _set_root_state(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        root_pos: ArrayLike,
        root_rot: ArrayLike,
        root_vel: ArrayLike | None = None,
        root_ang_vel: ArrayLike | None = None,
    ) -> None:
        env_ids = self._env_ids(env_id)
        for local_id, eid in enumerate(env_ids):
            pos = self._world_pos(
                eid,
                self._select_flat_value(root_pos, eid, 3, local_id, len(env_ids)),
            )
            rot = self._select_flat_value(root_rot, eid, 4, local_id, len(env_ids))
            vel = (
                None
                if root_vel is None
                else self._select_flat_value(root_vel, eid, 3, local_id, len(env_ids))
            )
            ang_vel = (
                None
                if root_ang_vel is None
                else self._select_flat_value(
                    root_ang_vel, eid, 3, local_id, len(env_ids)
                )
            )
            self._world.set_root_state(
                eid,
                obj_id,
                pos,
                rot,
                vel,
                ang_vel,
            )
            self._set_root_pending_value(eid, obj_id, "pos", pos)
            self._set_root_pending_value(eid, obj_id, "rot", rot)
            if vel is not None:
                self._set_root_pending_value(eid, obj_id, "vel", vel)
            if ang_vel is not None:
                self._set_root_pending_value(eid, obj_id, "ang_vel", ang_vel)

    def _set_dof_state(
        self,
        env_id: EnvIdLike,
        obj_id: int,
        dof_pos: ArrayLike,
        dof_vel: ArrayLike | None = None,
    ) -> None:
        width = self.get_obj_num_dofs(obj_id)
        env_ids = self._env_ids(env_id)
        for local_id, eid in enumerate(env_ids):
            pos = self._select_flat_value(dof_pos, eid, width, local_id, len(env_ids))
            vel = (
                None
                if dof_vel is None
                else self._select_flat_value(
                    dof_vel, eid, width, local_id, len(env_ids)
                )
            )
            self._world.set_dof_state(
                eid,
                obj_id,
                pos,
                vel,
            )
            self._set_dof_pending_value(eid, obj_id, "pos", pos, width)
            if vel is not None:
                self._set_dof_pending_value(eid, obj_id, "vel", vel, width)

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
        for local_id, eid in enumerate(env_ids):
            selected = self._select_body_vector_value(
                value, eid, num_bodies, 3, local_id, len(env_ids)
            )
            values[int(eid), obj_id, :num_bodies] = np.asarray(
                selected, dtype=np.float32
            ).reshape(
                num_bodies, 3
            )
            mask[int(eid), obj_id] = True

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
            vel_values=np.zeros(shape, dtype=np.float32),
            vel_masks=np.zeros(mask_shape, dtype=bool),
            ang_vel_values=np.zeros(shape, dtype=np.float32),
            ang_vel_masks=np.zeros(mask_shape, dtype=bool),
        )

    def _body_velocity_override_storage(
        self, kind: BodyVelocityKind
    ) -> tuple[np.ndarray, np.ndarray]:
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
                return np.zeros((self._num_envs, num_bodies, 3), dtype=np.float32)
            return base
        if base is None:
            out = np.zeros((self._num_envs, num_bodies, 3), dtype=np.float32)
        else:
            out = np.asarray(base, dtype=np.float32).copy()
        out[mask] = values[mask, obj_id, :num_bodies]
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

        root_shape3 = (self._num_envs, objs_per_env, 3)
        root_shape4 = (self._num_envs, objs_per_env, 4)
        root_mask_shape = (self._num_envs, objs_per_env)
        dof_shape = (self._num_envs, objs_per_env, max_dofs)
        self._pending_state = _PendingStateBuffers(
            root_pos=np.zeros(root_shape3, dtype=np.float32),
            root_rot=np.zeros(root_shape4, dtype=np.float32),
            root_vel=np.zeros(root_shape3, dtype=np.float32),
            root_ang_vel=np.zeros(root_shape3, dtype=np.float32),
            root_pos_mask=np.zeros(root_mask_shape, dtype=bool),
            root_rot_mask=np.zeros(root_mask_shape, dtype=bool),
            root_vel_mask=np.zeros(root_mask_shape, dtype=bool),
            root_ang_vel_mask=np.zeros(root_mask_shape, dtype=bool),
            dof_pos=np.zeros(dof_shape, dtype=np.float32),
            dof_vel=np.zeros(dof_shape, dtype=np.float32),
            dof_pos_mask=np.zeros(root_mask_shape, dtype=bool),
            dof_vel_mask=np.zeros(root_mask_shape, dtype=bool),
            dof_widths=dof_widths,
        )

    def _root_pending_storage(
        self, component: RootComponent
    ) -> tuple[np.ndarray, np.ndarray]:
        buffers = self._require_pending_state()
        if component == "pos":
            return buffers.root_pos, buffers.root_pos_mask
        if component == "rot":
            return buffers.root_rot, buffers.root_rot_mask
        if component == "vel":
            return buffers.root_vel, buffers.root_vel_mask
        if component == "ang_vel":
            return buffers.root_ang_vel, buffers.root_ang_vel_mask
        raise ValueError(f"unsupported root component: {component}")

    def _dof_pending_storage(
        self, component: DofComponent
    ) -> tuple[np.ndarray, np.ndarray]:
        buffers = self._require_pending_state()
        if component == "pos":
            return buffers.dof_pos, buffers.dof_pos_mask
        if component == "vel":
            return buffers.dof_vel, buffers.dof_vel_mask
        raise ValueError(f"unsupported dof component: {component}")

    def _require_pending_state(self) -> _PendingStateBuffers:
        if self._pending_state is None:
            raise RuntimeError(
                "pending state buffers are not initialized; call initialize_sim() "
                "before querying or mutating simulated state"
            )
        return self._pending_state

    def _set_root_pending_value(
        self, env_id: int, obj_id: int, component: RootComponent, value: ArrayLike
    ) -> None:
        values, mask = self._root_pending_storage(component)
        values[int(env_id), int(obj_id)] = np.asarray(value, dtype=np.float32)
        mask[int(env_id), int(obj_id)] = True

    def _set_dof_pending_value(
        self,
        env_id: int,
        obj_id: int,
        component: DofComponent,
        value: ArrayLike,
        width: int,
    ) -> None:
        values, mask = self._dof_pending_storage(component)
        eid = int(env_id)
        oid = int(obj_id)
        values[eid, oid, :width] = np.asarray(value, dtype=np.float32).reshape(width)
        mask[eid, oid] = True

    def _clear_state_pending_overrides(self) -> None:
        buffers = self._pending_state
        if buffers is None:
            return
        buffers.root_pos_mask[:, :] = False
        buffers.root_rot_mask[:, :] = False
        buffers.root_vel_mask[:, :] = False
        buffers.root_ang_vel_mask[:, :] = False
        buffers.dof_pos_mask[:, :] = False
        buffers.dof_vel_mask[:, :] = False

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
            self._viewer.visual_bridge.set_body_transforms(
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
            if (env_id, oid) in self._visual_body_pos or (env_id, oid) in self._visual_body_rot:
                continue
            self._viewer.visual_bridge.set_root_transform(
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
                return np.repeat(
                    arr[int(env_id)].reshape(1, width), num_bodies, axis=0
                )
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

    def _env_offset(self, env_id):
        return self._env_offsets[int(env_id)]

    def _world_pos(self, env_id: int, value: ArrayLike) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        return arr + self._env_offset(env_id)

    def _world_pos_batch(self, values: ArrayLike) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).copy()
        if arr.ndim >= 2 and arr.shape[0] == self._num_envs:
            view_shape = (self._num_envs,) + (1,) * (arr.ndim - 2) + (3,)
            arr += self._env_offsets.reshape(view_shape)
        return arr

    def _local_pos(self, env_id: int, value: ArrayLike) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        return arr - self._env_offset(env_id)

    def _local_pos_batch(self, values: ArrayLike) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).copy()
        if arr.ndim >= 2 and arr.shape[0] == self._num_envs:
            view_shape = (self._num_envs,) + (1,) * (arr.ndim - 2) + (3,)
            arr -= self._env_offsets.reshape(view_shape)
        return arr

    def _env_ids(self, env_id: EnvIdLike) -> list[int]:
        if env_id is None:
            return list(range(self._num_envs))
        if self._is_single_env_id(env_id):
            return [int(env_id)]
        ids = self._as_numpy(env_id).reshape(-1)
        return [int(x) for x in ids]

    def _is_single_env_id(self, env_id: EnvIdLike) -> bool:
        return isinstance(env_id, (int, np.integer))

    def _as_numpy(self, value: ArrayLike) -> np.ndarray:
        if _torch is not None and _torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _out(self, value: ArrayLike) -> ArrayLike:
        arr = np.asarray(value, dtype=np.float32)
        if _torch is not None and self._device is not None:
            return _torch.as_tensor(arr, dtype=_torch.float32, device=self._device)
        return arr

    def _zeros_like(self, value):
        if _torch is not None and _torch.is_tensor(value):
            return _torch.zeros_like(value)
        return np.zeros_like(value, dtype=np.float32)

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
