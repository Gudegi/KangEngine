"""Compare native SkeletonMotion with KELab's original MotionLib through PhysX.

Run with::

    PYTHONPATH=python KELab-private/.venv/bin/python \
        python/examples/adapters/physx/physx_articulation_motion.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

import kangengine as ke
from kangengine import visual

from kangengine.adapters.physx import PhysXMotionAdapter


def _default_mjcf() -> Path:
    return Path(ke.__file__).resolve().parent / "assets/characters/kw/kw.xml"


def _default_motion() -> Path:
    return (
        Path(ke.__file__).resolve().parent / "assets/characters/kw/motions/Walking.npy"
    )


def _original_motion_lib_type():
    root = Path(__file__).resolve().parents[4]
    for source in (
        root / "KELab-private/source/ke_lab",
        root / "KELab-private/source/ke_lab_tasks",
    ):
        if not source.is_dir():
            raise FileNotFoundError(source)
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    from ke_lab_tasks.direct.amp.motion import MotionLib

    return MotionLib


def _motion_lib_input(path: Path, dfs_body_names: list[str]) -> Path:
    """Reorder a legacy DFS clip into the original MotionLib policy order."""

    policy_names = (
        "Hips",
        "LeftHip",
        "RightHip",
        "Chest",
        "LeftKnee",
        "RightKnee",
        "LeftShoulder",
        "RightShoulder",
        "Neck",
        "LeftAnkle",
        "RightAnkle",
        "LeftElbow",
        "RightElbow",
    )
    saved = np.load(path, allow_pickle=True).item()
    rotations = np.asarray(saved["localRotations"])
    indices = [dfs_body_names.index(name) for name in policy_names]
    converted = dict(saved)
    converted["localRotations"] = rotations[:, indices].copy()
    converted["skeleton_tree"] = {"node_names": list(policy_names)}
    handle = tempfile.NamedTemporaryFile(
        prefix="kangengine_kw_motionlib_", suffix=".npy", delete=False
    )
    handle.close()
    output = Path(handle.name)
    np.save(output, converted, allow_pickle=True)
    return output


def _load_kw_motion(path: Path, tree) -> ke.animation.SkeletonMotion:
    saved = np.load(path, allow_pickle=True).item()
    rotations = np.asarray(saved["localRotations"], dtype=np.float32)
    roots = np.asarray(saved["rootTranslations"], dtype=np.float32)
    fps = float(np.asarray(saved["frameRate"]).item())
    # Legacy KW motion is y-up/z-forward; KangEngine is z-up/x-forward.
    roots = roots[:, [2, 0, 1]]
    rotations = rotations[..., [0, 3, 1, 2]]
    if rotations.shape[1] != tree.num_joints():
        raise ValueError(
            "legacy KW motion must match the MJCF DFS body count because it "
            "does not store body names"
        )
    return ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=tree,
        root_translations=roots,
        local_rotations_wxyz=rotations,
        fps=fps,
        motion_name=path.name,
    )


class PhysXArticulationMotionApp(ke.App):
    def __init__(
        self,
        mjcf_path: Path,
        motion_path: Path,
        separation: float,
        max_frames: int,
        loop: bool,
        motion_source: str,
    ) -> None:
        super().__init__()
        self.mjcf_path = mjcf_path
        self.motion_path = motion_path
        self.separation = float(separation)
        self.max_frames = int(max_frames)
        self.loop = bool(loop)
        self.motion_source = motion_source

    def setup(self) -> None:
        self.set_camera_view([4.5, -7.0, 3.0], [0.0, 0.0, 1.0])
        self.materials = self.create_standard_materials()
        self.scene.add_ground(scale=20.0, material=self.materials.ground)

        data = ke.asset.MJCFLoader.load(str(self.mjcf_path), order="DFS")
        self.source_motion = _load_kw_motion(self.motion_path, data.skeleton_tree)
        self.configure_timing(
            ke.SimulationTimingConfig(
                render_hz=60.0,
                physics_hz=self.source_motion.fps(),
                fixed_update_hz=self.source_motion.fps(),
            )
        )
        self.set_simulation_hotkeys_enabled(True)

        self.world = ke.sim.KangSimWorld(
            num_envs=1,
            sim_dt=1.0 / self.source_motion.fps(),
            add_ground=False,
        )
        self.robot = self.world.add_articulation(
            data,
            env_id=0,
            obj_id=0,
            name="kw_physx_motion",
            config=ke.physics.ArticulationConfig.free_base(),
        )
        body_names = data.skeleton_tree.node_names()
        if self.motion_source == "canonical":
            layout = ke.animation.ArticulationCoordinateLayout.from_data(
                data=data, free_root=True
            )
            adapter = PhysXMotionAdapter(layout)
            adapter.validate_articulation(self.robot.articulation)
            buffers = adapter.pack_skeleton_motion(self.source_motion)
            self.root_positions = buffers.root_positions
            self.root_rotations = buffers.root_rotations_xyzw
            self.root_velocities = buffers.root_linear_velocities
            self.root_angular_velocities = buffers.root_angular_velocities
            self.dof_positions = buffers.joint_positions
            self.dof_velocities = buffers.joint_velocities
            self.num_frames = self.source_motion.num_frames()
            self.motion_fps = self.source_motion.fps()
            self.max_residual = 0.0
        else:
            motion_lib_type = _original_motion_lib_type()
            dof_offsets = (0, 3, 6, 9, 10, 11, 14, 17, 20, 23, 26, 27, 28)
            policy_body_names = (
                "LeftHip",
                "RightHip",
                "Chest",
                "LeftKnee",
                "RightKnee",
                "LeftShoulder",
                "RightShoulder",
                "Neck",
                "LeftAnkle",
                "RightAnkle",
                "LeftElbow",
                "RightElbow",
            )
            motion_lib_path = _motion_lib_input(self.motion_path, body_names)
            try:
                self.motion_lib = motion_lib_type(
                    motion_lib_path,
                    data,
                    self.robot.articulation,
                    "cpu",
                    key_body_ids=(),
                    dof_body_ids=tuple(range(1, 13)),
                    dof_offsets=dof_offsets,
                    dof_obs_size=52,
                    local_root_obs=True,
                    is_T_pose=True,
                )
            finally:
                motion_lib_path.unlink(missing_ok=True)
            physical_spans = {}
            physical_offset = 0
            for body_id in sorted(data.joints):
                size = len(data.joints[body_id])
                physical_spans[body_names[body_id]] = range(
                    physical_offset, physical_offset + size
                )
                physical_offset += size
            policy_to_physics = []
            for name in policy_body_names:
                policy_to_physics.extend(physical_spans[name])
            self.root_positions = self.motion_lib._root_pos.clone()
            self.root_rotations = self.motion_lib._root_rot
            self.root_velocities = self.motion_lib._root_vel
            self.root_angular_velocities = self.motion_lib._root_ang_vel
            self.dof_positions = self.motion_lib._dof_pos.new_empty(
                self.motion_lib._dof_pos.shape
            )
            self.dof_velocities = self.motion_lib._dof_vel.new_empty(
                self.motion_lib._dof_vel.shape
            )
            self.dof_positions[:, policy_to_physics] = self.motion_lib._dof_pos
            self.dof_velocities[:, policy_to_physics] = self.motion_lib._dof_vel
            self.num_frames = self.motion_lib.num_frames
            self.motion_fps = self.motion_lib._fps
            self.max_residual = 0.0

        self.root_positions[:, 0] += 0.5 * self.separation
        self.sim_visual = ke.visual.sim.SimWorldVisualizer(self, self.world)
        self.sim_visual.add_articulation_scene_graph(
            0,
            0,
            str(self.mjcf_path),
            path="/PhysX",
            order="DFS",
            material=self.materials.pbr,
            color=np.array([1.0, 0.42, 0.14, 1.0], dtype=np.float32),
        )

        source_asset = visual.ArticulatedSurfaceAsset.from_mjcf(
            self.mjcf_path, order="DFS"
        )
        self.source_surface = source_asset.create(
            self,
            "/NativeSource",
            material=self.materials.common,
            color=(0.2, 0.55, 1.0, 1.0),
        )
        self.frame = 0
        self.rendered_frames = 0
        self._apply_frame()
        print(
            f"PhysX articulation motion: {self.mjcf_path}\n"
            f"source={self.motion_path}\n"
            f"frames={self.num_frames} fps={self.motion_fps:g}\n"
            f"blue/left=native SkeletonMotion, orange/right={self.motion_source}\n"
            f"maximum mapping residual={self.max_residual:.6f} rad\n"
            f"loop={'on' if self.loop else 'off'}"
        )

    def _apply_frame(self) -> None:
        index = self.frame
        self.robot.set_root_state(
            None,
            self.root_positions[index],
            self.root_rotations[index],
            self.root_velocities[index],
            self.root_angular_velocities[index],
            immediate=True,
        )
        self.robot.set_dof_state(
            None,
            self.dof_positions[index],
            self.dof_velocities[index],
            immediate=True,
        )
        self.world.step(substeps=0, apply_commands=False)

        source = self.source_motion.frame(index)
        position = source.root_translation()
        source.set_root_translation(
            (position.x - 0.5 * self.separation, position.y, position.z)
        )
        self.source_surface.apply_state(source)
        self.sim_visual.sync()

    def fixed_update(self, fixed_dt: float) -> None:
        del fixed_dt
        if self.frame + 1 >= self.num_frames:
            if not self.loop:
                return
            self.frame = 0
        else:
            self.frame += 1
        self._apply_frame()

    def render(self) -> None:
        self.rendered_frames += 1
        if self.max_frames > 0 and self.rendered_frames >= self.max_frames:
            self.request_close()

    def cleanup(self) -> None:
        if hasattr(self, "sim_visual"):
            self.sim_visual = None
        if hasattr(self, "world") and self.world is not None:
            self.world.release()
            self.world = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=_default_mjcf())
    parser.add_argument("--motion", type=Path, default=_default_motion())
    parser.add_argument("--separation", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument(
        "--motion-source",
        choices=("canonical", "motionlib"),
        default="canonical",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop even though the walking clip has non-periodic root displacement.",
    )
    args = parser.parse_args()

    mjcf_path = args.mjcf.expanduser().resolve()
    motion_path = args.motion.expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(mjcf_path)
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    app = PhysXArticulationMotionApp(
        mjcf_path,
        motion_path,
        args.separation,
        args.frames,
        args.loop,
        args.motion_source,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Z, headless=args.headless)
    app.start()


if __name__ == "__main__":
    main()
