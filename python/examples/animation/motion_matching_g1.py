"""SEED G1 CSV -> ArticulationMotion -> SkeletonMotion -> MM -> joint-limited FK."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import animation, imgui, visual
from kangengine.utils.math import (
    quat_wxyz_multiply_numpy,
    quat_wxyz_to_rotation_vector_numpy,
)
from motion_matching_common import MotionMatchingViewer, benchmark_search, parse_args


REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO.parent / "dataset/seed/g1/csv"
DEFAULT_CHARACTER = REPO / "assets/characters/g1/g1_29dof.xml"
POSITION_FILTER_WINDOW = 2.0
ROTATION_FILTER_WINDOW = 2.0
FPS = 120.0
TAKES_PER_FAMILY = 3
CLIP_PREFIXES = (
    "neutral_idle_loop_001__A082",
    "walk_180_R_",
    "walk_hands_on_back_180_start_R_",
    "walk_hands_on_back_180_loop_R_",
    "walk_hands_on_back_180_stop_R_",
    "walk_arc_cw_start_R_",
    "walk_arc_cw_loop_R_",
    "walk_arc_cw_stop_R_",
    "jog_ff_start_180_R_",
    "jog_ff_loop_180_R_",
    "jog_ff_stop_180_R_",
    "jog_arc_cw_start_R_",
    "jog_arc_cw_loop_R_",
    "jog_arc_cw_stop_R_",
    "run_start_180_R_",
    "run_loop_180_R_",
    "run_stop_180_R_",
    "turn_jog_270_R_",
    "turn_jog_360_R_",
    "turn_run_270_R_",
    "turn_run_360_R_",
)
CONFIG = animation.MotionMatchingConfig(
    position_joints=("left_ankle_roll_link", "right_ankle_roll_link"),
    velocity_joints=("left_ankle_roll_link", "right_ankle_roll_link", "pelvis"),
    up_axis="Z",
    forward_axis="X",
)


def _select_clips(dataset: Path) -> list[Path]:
    """Select locomotion takes and their supplied mirrors."""
    paths = sorted(dataset.rglob("*.csv"))
    selected = []
    for prefix in CLIP_PREFIXES:
        takes = [
            p for p in paths if p.name.startswith(prefix) and not p.stem.endswith("_M")
        ][:TAKES_PER_FAMILY]
        if not takes:
            raise ValueError(f"No {prefix}*.csv in {dataset}; check CLIP_PREFIXES")
        for path in takes:
            selected.append(path)
            mirror = path.with_stem(path.stem + "_M")
            if mirror.is_file():
                selected.append(mirror)
    return selected


def _load_seed_csv(
    csv_path: Path, layout: animation.ArticulationCoordinateLayout
) -> animation.ArticulationMotion:
    """Convert SEED G1 CSV (Z-up, 120 FPS) to ArticulationMotion.

    - Root position: cm -> m; joints: deg -> rad.
    - Root rotation: extrinsic XYZ degrees -> WXYZ quaternion.
    - qd: centered differences, one-sided at endpoints.
    """
    rows = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64, ndmin=1)
    hinges = [
        b
        for b in layout.blocks
        if b.type == animation.ArticulationCoordinateType.REVOLUTE
    ]
    q = np.zeros((len(rows), layout.nq), dtype=np.float32)
    q[:, :3] = np.column_stack([rows["root_translate" + a] for a in "XYZ"]) * 0.01
    rotations = np.tile((1, 0, 0, 0), (len(rows), 1)).astype(np.float32)
    for axis, name in enumerate("XYZ", 1):
        half_angle = np.deg2rad(rows["root_rotate" + name]) * 0.5
        rotation = np.zeros_like(rotations)
        rotation[:, 0], rotation[:, axis] = np.cos(half_angle), np.sin(half_angle)
        rotations = quat_wxyz_multiply_numpy(rotation, rotations)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    q[:, 3:7] = rotations
    qd = np.zeros((len(rows), layout.nv), dtype=np.float32)
    qd[:, :3] = np.gradient(q[:, :3], 1 / FPS, axis=0)
    before = np.maximum(np.arange(len(rows)) - 1, 0)
    after = np.minimum(np.arange(len(rows)) + 1, len(rows) - 1)
    # Root angular velocity is world-space.
    delta = quat_wxyz_multiply_numpy(
        rotations[after], rotations[before] * (1, -1, -1, -1)
    )
    qd[:, 3:6] = (
        quat_wxyz_to_rotation_vector_numpy(delta) * (FPS / (after - before))[:, None]
    )
    for block in hinges:
        angles = np.deg2rad(rows[block.joint_name + "_dof"])
        q[:, block.q_offset] = angles
        qd[:, block.qd_offset] = np.gradient(np.unwrap(angles), 1 / FPS)
    return animation.ArticulationMotion.from_arrays(
        layout=layout, q=q, qd=qd, fps=FPS, motion_name=csv_path.stem
    )


def _load_mapper(mjcf_path: Path) -> animation.ArticulationMotionMapper:
    data = ke.asset.MJCFLoader.load(mjcf_path=str(mjcf_path), order="DFS")
    layout = animation.ArticulationCoordinateLayout.from_data(data=data, free_root=True)
    if layout.nq != 36 or layout.nv != 35:
        raise ValueError(
            "This example requires a free-root G1 model with 29 hinge joints"
        )
    return animation.ArticulationMotionMapper(layout)


def _build_motions(args: argparse.Namespace) -> list[animation.SkeletonMotion]:
    mapper = _load_mapper(DEFAULT_CHARACTER)
    csv_paths = _select_clips(args.dataset)
    motions = []
    for path in csv_paths:
        motion = _load_seed_csv(path, mapper.layout)
        motions.append(mapper.to_skeleton_motion(motion))
        print(f"  {path.name}: {motion.num_frames()} frames, {motion.fps():g} FPS")
    return motions


def load_database(args: argparse.Namespace) -> animation.MotionMatchingDatabase:
    database = animation.MotionMatchingDatabase.from_motions(
        _build_motions(args),
        config=CONFIG,
        feature_builder=animation.WalkingFeatures(
            position_filter_window=POSITION_FILTER_WINDOW,
            rotation_filter_window=ROTATION_FILTER_WINDOW,
        ),
    )
    print(
        f"Database: {len(database.motions)} clips, {database.num_frames} candidates, "
        f"{database.features.shape[1]} features, {len(database.motions[0].node_names())} joints"
    )
    return database


def _demo_input(time: float) -> tuple[np.ndarray, float, str]:
    speed, angle, label = (
        (0, 0, "Idle"),
        (0.8, 0, "Walk"),
        (1.6, 0, "Jog"),
        (1.2, 90, "Turn left"),
        (1.2, 180, "Reverse"),
        (1.2, -90, "Turn right"),
        (0.8, 0, "Walk"),
        (0, 0, "Stop"),
    )[int(time / 3) % 8]
    angle = math.radians(angle)
    facing = np.array((math.cos(angle), math.sin(angle), 0))
    return speed * facing, angle, label


def _project_pose(mapper, state):
    """Clamp the smoothed output to joint limits; keep source playback unchanged."""
    raw = np.asarray(mapper.to_articulation_coordinates(state, clamp_to_limits=False).q)
    q = np.asarray(mapper.to_articulation_coordinates(state, clamp_to_limits=True).q)
    return q, float(np.max(np.abs(q - raw)))


class Viewer(MotionMatchingViewer):
    def __init__(
        self, database: animation.MotionMatchingDatabase, args: argparse.Namespace
    ):
        super().__init__(
            database,
            transition_half_life=0.1,
            walk_speed=1.0,
            run_speed=3.0,
            demo_input=_demo_input,
        )
        self.mapper = _load_mapper(DEFAULT_CHARACTER)
        self.character_path = DEFAULT_CHARACTER
        self.correction = 0.0

    def setup(self):
        materials = self.create_standard_materials()
        self.add_ground(material=materials.ground)
        self.character_visual: visual.ArticulatedSurface = (
            visual.ArticulatedSurface.create_from_mjcf(
                app=self,
                path="/g1",
                mjcf_path=str(self.character_path),
                material=materials.common,
                order="DFS",
            )
        )
        self.apply_pose()
        self.update_camera()

    def apply_pose(self):
        self.q, self.correction = _project_pose(self.mapper, self.controller.state)
        self.character_visual.apply_state(state=self.mapper.to_skeleton_state(self.q))

    def render(self):
        super().render()
        imgui.begin("G1")
        imgui.text(f"Joint limit correction: {math.degrees(self.correction):.2f} deg")
        imgui.text(
            "Kinematic reference playback; joint limits applied after smoothing."
        )
        imgui.end()


def main():
    args = parse_args(default_dataset=DEFAULT_DATASET, description=__doc__)
    database = load_database(args)
    if args.benchmark_search:
        benchmark_search(database)
        return
    app = Viewer(database, args)
    app.initialize(width=1920, height=1080, up_axis=ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
