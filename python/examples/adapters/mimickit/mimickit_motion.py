"""Play a MimicKit robot clip with the KE, Newton, or MuJoCo backend.

Run with::

    PYTHONPATH=python python/.venv/bin/python \
        python/examples/adapters/mimickit/mimickit_motion.py \
        --backend ke --robot go2 --mimickit-root ../MimicKit
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import time

import numpy as np

import kangengine as ke
from kangengine.adapters.mimickit import load_articulation_motion


_ROBOT_PRESETS = {
    "g1": ("g1/g1.xml", "g1", "g1_walk.pkl"),
    "g1_mesh": ("g1/g1_mesh.xml", "g1", "g1_walk.pkl"),
    "go2": ("go2/go2.xml", "go2", "go2_pace.pkl"),
    "pi": (
        "hightorque_pi_plus/pi_22dof.xml",
        "hightorque_pi_plus",
        "pi_plus_walk.pkl",
    ),
    "humanoid": ("humanoid/humanoid.xml", "humanoid", "humanoid_spinkick.pkl"),
    "smpl": ("smpl/smpl.xml", "smpl", "smpl_walk.pkl"),
    "sword_shield": (
        "sword_shield/humanoid_sword_shield.xml",
        "reallusion",
        "RL_Avatar_Atk_2xCombo01_Motion.pkl",
    ),
}


def _default_mimickit_root() -> Path:
    configured = os.environ.get("MIMICKIT_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[5] / "MimicKit"


def _load_model_with_floor(mjcf_path: Path, mujoco):
    """Compile a MimicKit robot without its floor and add a Z=0 plane."""

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    # Some MimicKit assets rely on MuJoCo's inertia balancing compiler option.
    spec.compiler.balanceinertia = True
    for geom in tuple(spec.worldbody.geoms):
        if geom.name in ("floor", "ground"):
            spec.delete(geom)
    spec.worldbody.add_geom(
        name="example_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 0.125),
        friction=(1.0, 0.1, 0.1),
        rgba=(0.8, 0.9, 0.8, 1.0),
    )
    return spec.compile()


class KangEngineMimicKitMotionApp(ke.App):
    """Apply MotionLibrary samples to a kinematic PhysX articulation."""

    def __init__(self, mjcf_path: Path, motion, layout, args) -> None:
        super().__init__()
        self.mjcf_path = mjcf_path
        self.motion = motion
        self.layout = layout
        self.args = args

    def setup(self) -> None:
        from kangengine.adapters.physx import PhysXMotionAdapter

        self.set_camera_view([3.5, -5.5, 2.4], [0.0, 0.0, 0.9])
        self.materials = self.create_standard_materials()
        self.scene.add_ground(scale=20.0, material=self.materials.ground)
        self.configure_timing(
            ke.SimulationTimingConfig(
                render_hz=60.0,
                physics_hz=self.motion.fps(),
                fixed_update_hz=self.motion.fps(),
            )
        )
        articulation_data = ke.asset.MJCFLoader.load(str(self.mjcf_path), order="DFS")
        self.world = ke.sim.KangSimWorld(
            num_envs=1,
            sim_dt=1.0 / self.motion.fps(),
            add_ground=False,
        )
        self.robot = self.world.add_articulation(
            articulation_data,
            env_id=0,
            obj_id=0,
            name=f"mimickit_{self.args.robot}",
            config=ke.physics.ArticulationConfig.free_base(),
        )
        adapter = PhysXMotionAdapter(self.layout)
        adapter.validate_articulation(self.robot.articulation)
        self.library = ke.animation.MotionLibrary(
            [self.motion], adapter=adapter, device="cpu"
        )
        self.sim_visual = ke.visual.sim.SimWorldVisualizer(self, self.world)
        self.sim_visual.add_articulation_scene_graph(
            0,
            0,
            str(self.mjcf_path),
            path="/Robot",
            order="DFS",
            material=self.materials.pbr,
            color=np.array([0.2, 0.55, 1.0, 1.0], dtype=np.float32),
        )
        self.playback_time = 0.0
        self.rendered_frames = 0
        self._apply_frame()

    def _apply_frame(self) -> None:
        sample = self.library.sample(0, self.playback_time)
        if sample.backend_state is None:
            raise RuntimeError("PhysX MotionLibrary adapter produced no state")
        backend_state = sample.backend_state
        self.robot.set_root_state(
            None,
            backend_state.root_positions,
            backend_state.root_rotations_xyzw,
            backend_state.root_linear_velocities,
            backend_state.root_angular_velocities,
            immediate=True,
        )
        self.robot.set_dof_state(
            None,
            backend_state.joint_positions,
            backend_state.joint_velocities,
            immediate=True,
        )
        self.world.step(substeps=0, apply_commands=False)
        self.sim_visual.sync()

    def fixed_update(self, fixed_dt: float) -> None:
        self.playback_time += fixed_dt
        self._apply_frame()

    def render(self) -> None:
        self.rendered_frames += 1
        if self.args.frames > 0 and self.rendered_frames >= self.args.frames:
            self.request_close()

    def cleanup(self) -> None:
        if hasattr(self, "sim_visual"):
            self.sim_visual = None
        if hasattr(self, "world") and self.world is not None:
            self.world.release()
            self.world = None


def _run_ke(mjcf_path: Path, motion, layout, args) -> None:
    app = KangEngineMimicKitMotionApp(mjcf_path, motion, layout, args)
    app.initialize(args.width, args.height, False, ke.UpAxis.Z, headless=args.headless)
    app.start()


def _run_newton(mjcf_path: Path, motion, layout, args) -> None:
    import newton
    import warp as wp

    from kangengine.adapters.newton import NewtonMotionAdapter
    from newton.viewer import ViewerGL

    newton.use_coord_layout_targets = True
    adapter = NewtonMotionAdapter(layout)
    library = ke.animation.MotionLibrary([motion], adapter=adapter, device=args.device)
    parse_meshes = importlib.util.find_spec("trimesh") is not None
    if not parse_meshes:
        print("Newton: trimesh is unavailable; showing collision geometry only")
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(mjcf_path),
        floating=True,
        ignore_names=["floor", "ground"],
        parse_meshes=parse_meshes,
        enable_self_collisions=False,
    )
    model = builder.finalize(device=args.device)
    adapter.validate_model(model)
    state = model.state()
    viewer = ViewerGL(width=args.width, height=args.height, headless=args.headless)
    viewer.set_model(model)
    viewer.show_joints = bool(args.show_joints)
    viewer.set_camera(wp.vec3(3.5, -5.5, 2.4), pitch=-8.0, yaw=125.0)
    playback_time = 0.0
    previous_tick = time.perf_counter()
    rendered_frames = 0
    try:
        while viewer.is_running():
            current_tick = time.perf_counter()
            if viewer.should_step():
                if viewer.is_paused():
                    playback_time += 1.0 / motion.fps()
                else:
                    playback_time += current_tick - previous_tick
            previous_tick = current_tick
            sample = library.sample(0, playback_time)
            if sample.backend_state is None:
                raise RuntimeError("Newton MotionLibrary adapter produced no state")
            model.joint_q.assign(
                wp.from_torch(sample.backend_state.joint_q.contiguous())
            )
            model.joint_qd.assign(
                wp.from_torch(sample.backend_state.joint_qd.contiguous())
            )
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            viewer.begin_frame(playback_time)
            viewer.log_state(state)
            viewer.end_frame()
            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
    finally:
        viewer.close()


def _run_mujoco(mjcf_path: Path, motion, layout, args) -> None:
    import mujoco
    import mujoco.viewer

    from kangengine.adapters.mujoco import MuJoCoMotionAdapter

    adapter = MuJoCoMotionAdapter(layout)
    library = ke.animation.MotionLibrary([motion], adapter=adapter, device="cpu")
    model = _load_model_with_floor(mjcf_path, mujoco)
    data = mujoco.MjData(model)
    adapter.validate_model(model)
    playback_started = time.perf_counter()
    rendered_frames = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            playback_time = started - playback_started
            sample = library.sample(0, playback_time)
            if sample.backend_state is None:
                raise RuntimeError("MuJoCo MotionLibrary adapter produced no state")
            data.qpos[:] = sample.backend_state.qpos.numpy()
            data.qvel[:] = sample.backend_state.qvel.numpy()
            mujoco.mj_forward(model, data)
            viewer.sync()
            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
            remaining = 1.0 / 60.0 - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ke", "newton", "mujoco"), default="ke")
    parser.add_argument("--robot", choices=tuple(_ROBOT_PRESETS), default="g1")
    parser.add_argument("--mimickit-root", type=Path, default=_default_mimickit_root())
    parser.add_argument(
        "--motion",
        help="Motion filename in the selected robot's MimicKit motion directory.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--show-joints", action="store_true")
    args = parser.parse_args()

    mimickit_root = args.mimickit_root.expanduser().resolve()
    mjcf_relative, motion_directory, default_motion = _ROBOT_PRESETS[args.robot]
    mjcf_path = mimickit_root / "data/assets" / mjcf_relative
    motion_value = default_motion if args.motion is None else args.motion
    requested_motion = Path(motion_value).expanduser()
    if requested_motion.is_absolute():
        motion_path = requested_motion
    elif requested_motion.parent != Path("."):
        motion_path = mimickit_root / "data/motions" / requested_motion
    else:
        motion_path = (
            mimickit_root / "data/motions" / motion_directory / requested_motion
        )
    if not mjcf_path.is_file():
        raise FileNotFoundError(mjcf_path)
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)

    articulation_data = ke.asset.MJCFLoader.load(str(mjcf_path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=articulation_data, free_root=True
    )
    motion = load_articulation_motion(motion_path, layout)
    print(
        f"MimicKit motion: {motion_path}\n"
        f"robot={args.robot} backend={args.backend} "
        f"frames={motion.num_frames()} fps={motion.fps():g} "
        f"bodies={articulation_data.skeleton_tree.num_joints()} "
        f"canonical nq/nv={layout.nq}/{layout.nv}"
    )
    runners = {"ke": _run_ke, "newton": _run_newton, "mujoco": _run_mujoco}
    runners[args.backend](mjcf_path, motion, layout, args)


if __name__ == "__main__":
    main()
