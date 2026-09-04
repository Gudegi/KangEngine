"""Play a canonical ``ArticulationMotion`` NPZ on a PhysX articulation.

Example::

    PYTHONPATH=python python/.venv/bin/python \
        python/tools/animation/view_articulation_motion.py motion.npz \
        --mjcf robot.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.adapters.physx import PhysXMotionAdapter


def _default_g1_mjcf() -> Path:
    return Path(ke.__file__).resolve().parent / "assets/characters/g1/g1.xml"


class ArticulationMotionViewer(ke.App):
    """Interactive playback of one canonical motion using a target MJCF."""

    def __init__(
        self,
        motion_path: Path,
        mjcf_path: Path,
        *,
        source_bvh: Path | None,
        source_offset: tuple[float, float, float],
        source_time_offset: float,
        loop: bool,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.motion_path = motion_path
        self.mjcf_path = mjcf_path
        self.source_bvh = source_bvh
        self.source_offset = source_offset
        self.source_time_offset = float(source_time_offset)
        self.loop = bool(loop)
        self.max_frames = int(max_frames)

    def setup(self) -> None:
        self.set_ui_layout_mode(ke.UILayoutMode.EDITOR)
        self.set_camera_view([4.5, -7.0, 3.0], [0.0, 0.0, 1.0])
        self.materials = self.create_standard_materials()
        self.scene.add_ground(scale=20.0, material=self.materials.ground)

        data = ke.asset.MJCFLoader.load(str(self.mjcf_path), order="DFS")
        layout = ke.animation.ArticulationCoordinateLayout.from_data(
            data=data,
            free_root=True,
        )
        self.motion = ke.animation.load_articulation_motion_npz(
            self.motion_path,
            layout,
        )
        fps = self.motion.fps()
        self.configure_timing(
            ke.SimulationTimingConfig(
                render_hz=60.0,
                physics_hz=fps,
                fixed_update_hz=fps,
            )
        )
        self.set_simulation_hotkeys_enabled(False)

        self.world = ke.sim.KangSimWorld(
            num_envs=1,
            sim_dt=1.0 / fps,
            add_ground=False,
        )
        self.robot = self.world.add_articulation(
            data,
            env_id=0,
            obj_id=0,
            name="articulation_motion_viewer",
            config=ke.physics.ArticulationConfig.free_base(),
        )
        adapter = PhysXMotionAdapter(layout)
        adapter.validate_articulation(self.robot.articulation)
        self.buffers = adapter.pack(self.motion)
        if self.buffers.root_positions is None:
            raise ValueError(
                "viewer currently requires a free-root articulation motion"
            )

        self.sim_visual = ke.visual.sim.SimWorldVisualizer(self, self.world)
        self.sim_visual.add_articulation_scene_graph(
            0,
            0,
            str(self.mjcf_path),
            path="/ArticulationMotion",
            order="DFS",
            material=self.materials.pbr,
        )

        self.frame = 0
        self._applied_frame = -1
        self.rendered_frames = 0
        self.sequencer = ke.MotionSequencerPanel()
        self.sequencer.set_motion(
            self.motion_path.name,
            self.motion.num_frames(),
            fps,
        )
        self.sequencer.set_overlay(True)
        self.sequencer.set_overlay_width_ratio(1.0)
        self.sequencer.set_playing(True)
        self.sequencer.set_loop(self.loop)
        self.source_motion = None
        self.source_visual = None
        if self.source_bvh is not None:
            source = ke.asset.BVHLoader.load_motion(str(self.source_bvh))
            self.source_motion = ke.animation.convert_motion_coordinates(
                source,
                source=ke.animation.CoordinateSystem.Y_UP_Z_FORWARD,
                target=ke.animation.CoordinateSystem.Z_UP_X_FORWARD,
            )
            source_state = self._source_state_at(0.0)
            self.source_visual = ke.visual.SkeletalVisual.define(
                app=self,
                material=self.materials.common,
                path="/SourceBVH",
                state=source_state,
                config=ke.visual.SkeletalVisualConfig(
                    bone_color=ke.Vec4(1.0, 0.55, 0.18, 1.0),
                    joint_color=ke.Vec4(1.0, 0.85, 0.25, 1.0),
                    show_joints=True,
                ),
            )
            self.sequencer.set_motions(
                [self.motion_path.name, self.source_bvh.name],
                [self.motion.num_frames(), self.source_motion.num_frames()],
                [fps, self.source_motion.fps()],
            )
        self._apply_frame()
        print(
            f"ArticulationMotion viewer: {self.motion_path}\n"
            f"target={self.mjcf_path}\n"
            f"frames={self.motion.num_frames()} fps={fps:g} "
            f"loop={'on' if self.loop else 'off'}\n"
            "playback and frame seeking: bottom Motion Sequencer panel"
        )
        if self.source_motion is not None:
            print(
                f"source={self.source_bvh} frames={self.source_motion.num_frames()} "
                f"fps={self.source_motion.fps():g} offset={self.source_offset}"
            )

    def _source_state_at(self, time: float):
        state = self.source_motion.sample(
            max(0.0, time + self.source_time_offset),
            loop=self.sequencer.loop(),
        )
        root = state.root_translation()
        state.set_root_translation(
            [
                root.x + self.source_offset[0],
                root.y + self.source_offset[1],
                root.z + self.source_offset[2],
            ]
        )
        return state

    def _apply_frame(self) -> None:
        index = self.frame
        buffers = self.buffers
        self.robot.set_root_state(
            None,
            buffers.root_positions[index],
            buffers.root_rotations_xyzw[index],
            buffers.root_linear_velocities[index],
            buffers.root_angular_velocities[index],
            immediate=True,
        )
        self.robot.set_dof_state(
            None,
            buffers.joint_positions[index],
            buffers.joint_velocities[index],
            immediate=True,
        )
        self.world.step(substeps=0, apply_commands=False)
        self.sim_visual.sync()
        self._applied_frame = index

    def fixed_update(self, fixed_dt: float) -> None:
        del fixed_dt

    def pre_render(self) -> None:
        fps = float(self.motion.fps())
        frame_count = self.motion.num_frames()
        playback_duration = frame_count / fps
        current_time = float(self.sequencer.current_time())
        if self.sequencer.is_playing():
            current_time += self.get_delta_time() * self.sequencer.time_scale()
            if self.sequencer.loop():
                current_time %= playback_duration
            elif current_time >= playback_duration:
                current_time = max((frame_count - 1) / fps, 0.0)
                self.sequencer.set_playing(False)
            self.sequencer.set_current_time(current_time)
        self.frame = max(0, min(frame_count - 1, int(current_time * fps)))
        if self.frame != self._applied_frame:
            self._apply_frame()
        if self.source_visual is not None:
            self.source_visual.apply_state(self._source_state_at(current_time))

    def render(self) -> None:
        self.sequencer.build_panel()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="Canonical ArticulationMotion NPZ")
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=_default_g1_mjcf(),
        help="Target MJCF model (default: bundled Unitree G1 29-DOF)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Close after this many rendered frames; 0 runs until closed",
    )
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--source-bvh",
        type=Path,
        help="Optional original BVH to play beside the retargeted articulation.",
    )
    parser.add_argument(
        "--source-offset",
        nargs=3,
        type=float,
        default=(0.0, -1.5, 0.0),
        metavar=("X", "Y", "Z"),
        help="World offset for the source skeleton; use 0 0 0 to overlay it.",
    )
    parser.add_argument(
        "--source-time-offset",
        type=float,
        default=0.0,
        help="Source BVH playback offset in seconds.",
    )
    args = parser.parse_args()

    motion_path = args.motion.expanduser().resolve()
    mjcf_path = args.mjcf.expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    if not mjcf_path.is_file():
        raise FileNotFoundError(mjcf_path)
    source_bvh = (
        None if args.source_bvh is None else args.source_bvh.expanduser().resolve()
    )
    if source_bvh is not None and not source_bvh.is_file():
        raise FileNotFoundError(source_bvh)

    app = ArticulationMotionViewer(
        motion_path,
        mjcf_path,
        source_bvh=source_bvh,
        source_offset=tuple(args.source_offset),
        source_time_offset=args.source_time_offset,
        loop=not args.no_loop,
        max_frames=args.frames,
    )
    app.initialize(
        args.width,
        args.height,
        False,
        ke.UpAxis.Z,
        headless=args.headless,
    )
    app.start()


if __name__ == "__main__":
    main()
