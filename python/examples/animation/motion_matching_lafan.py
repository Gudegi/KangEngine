"""LAFAN-resolved Motion Matching with the Geno skin."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import animation, asset, imgui, visual
from kangengine.utils.math import quat_wxyz_multiply_numpy
from motion_matching_common import (
    benchmark_search,
    MotionMatchingViewer,
    parse_args,
    target_direction,
)


REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO.parent / "dataset/lafan_resolved"
DEFAULT_CHARACTER = REPO / "assets/external/Geno/Geno_bind.fbx"
CACHE_PATH = REPO / "output/motion_matching_lafan/database.npz"
POSITION_FILTER_WINDOW = 2.0
ROTATION_FILTER_WINDOW = 2.0
# [start, stop) from orangeduck/GenoViewPython-MotionMatching.
CLIPS = (
    ("pushAndStumble1_subject5", 397, 706),
    ("run1_subject5", 172, 14136),
    ("walk1_subject5", 160, 15518),
)
CONFIG = animation.MotionMatchingConfig(
    position_joints=("LeftToeBase", "RightToeBase"),
    velocity_joints=("LeftToeBase", "RightToeBase", "Hips"),
    up_axis="Y",
    forward_axis="Z",
)


def _load_range(
    path: Path, start: int, stop: int
) -> tuple[animation.SkeletonTree, np.ndarray, np.ndarray]:
    """Read a LAFAN range with XYZ-position / ZYX-rotation channels per joint."""
    tree = asset.BVHLoader.load_skeleton(bvh_path=str(path), scale=0.01)
    with path.open() as file:
        for line in file:
            if line.split()[:2] == ["Frame", "Time:"]:
                break
        data = np.loadtxt(
            file, skiprows=start, max_rows=stop - start, dtype=np.float32, ndmin=2
        )
    data = data.reshape(stop - start, tree.num_joints(), 6)
    positions = data[..., :3] * 0.01
    rotations = np.zeros((*data.shape[:2], 4), dtype=np.float32)
    rotations[..., 0] = 1
    for channel, axis in enumerate((3, 2, 1)):
        half_angle = np.deg2rad(data[..., 3 + channel]) * 0.5
        q = np.zeros_like(rotations)
        q[..., 0], q[..., axis] = np.cos(half_angle), np.sin(half_angle)
        rotations = quat_wxyz_multiply_numpy(rotations, q)
    return tree, positions, rotations


def _mirror_order(tree: animation.SkeletonTree) -> np.ndarray:
    names = list(tree.node_names())
    return np.array(
        [
            names.index(
                name.replace("Left", "Right")
                if "Left" in name
                else name.replace("Right", "Left")
                if "Right" in name
                else name
            )
            for name in names
        ]
    )


def _build_motions(args: argparse.Namespace) -> list[animation.SkeletonMotion]:
    motions = []
    canonical = None
    for name, start, stop in CLIPS:
        tree, positions, rotations = _load_range(
            args.dataset / f"{name}.bvh", start, stop
        )
        if canonical is None:
            # Root translation comes from frames, not the shared rig offset.
            offsets = np.array(
                [
                    [getattr(tree.local_translation(j), a) for a in "xyz"]
                    for j in range(tree.num_joints())
                ]
            )
            offsets[0] = (0, 0, 0)
            canonical = animation.SkeletonTree(
                names=tree.node_names(),
                parents=tree.parent_indices(),
                local_translations=offsets,
                local_rotations_wxyz=np.tile((1, 0, 0, 0), (tree.num_joints(), 1)),
            )
        if (
            tree.node_names() != canonical.node_names()
            or tree.parent_indices() != canonical.parent_indices()
        ):
            raise ValueError(f"{name}: skeleton topology differs from the first clip")
        deviation = float(
            np.max(np.linalg.norm(positions[:, 1:] - offsets[None, 1:], axis=-1))
        )
        if deviation > 0.02:
            raise ValueError(
                f"{name}: non-root translations differ by {deviation:.4f}m"
            )
        print(
            f"{name}: frames [{start}:{stop}], {stop - start} frames, 60 FPS; "
            f"fixed-offset approximation <= {deviation * 1000:.2f}mm"
        )
        order = _mirror_order(canonical)
        for mirrored in (True, False):
            roots = positions[:, 0] * (-1, 1, 1) if mirrored else positions[:, 0]
            # Mirror across X and swap left/right joints.
            local = rotations[:, order] * (1, 1, -1, -1) if mirrored else rotations
            motions.append(
                animation.SkeletonMotion.from_arrays(
                    skeleton_tree=canonical,
                    root_translations=roots,
                    local_rotations_wxyz=local,
                    fps=60,
                    motion_name=name + ("_mirrored" if mirrored else ""),
                )
            )
    return motions


def load_database(args: argparse.Namespace) -> animation.MotionMatchingDatabase:
    builder = animation.WalkingFeatures(
        position_filter_window=POSITION_FILTER_WINDOW,
        rotation_filter_window=ROTATION_FILTER_WINDOW,
    )
    if CACHE_PATH.is_file():
        database = animation.MotionMatchingDatabase.from_npz(CACHE_PATH)
        print(f"Loaded NPZ: {CACHE_PATH}")
        if database.schema["features"].get("reference_filter") != builder.describe(
            CONFIG
        ).get("reference_filter"):
            print(
                f"Cached reference filter differs from settings; delete {CACHE_PATH} to rebuild."
            )
    else:
        database = animation.MotionMatchingDatabase.from_motions(
            _build_motions(args), config=CONFIG, feature_builder=builder
        )
        database.save_npz(CACHE_PATH, write_summary=True)
        print(f"Built NPZ: {CACHE_PATH}")
    print(
        f"Reference filter: {database.schema['features'].get('reference_filter', 'off')}"
    )
    print(
        f"Database: {len(database.motions)} clips, {database.num_frames} candidates, "
        f"{database.features.shape[1]} features, {len(database.motions[0].node_names())} joints"
    )
    return database


def _demo_input(time: float) -> tuple[np.ndarray, float, str]:
    t = time % 24
    if t < 2:
        speed, angle, label = 0.0, 0.0, "Idle"
    elif t < 6:
        speed, angle, label = 1.4, 0.0, "Walk"
    elif t < 10:
        speed, angle, label = 3.5, 0.0, "Run"
    elif t < 14:
        speed, angle, label = 1.4, (t - 10) * math.pi / 4, "Turn while walking"
    elif t < 17:
        speed, angle, label = 0.0, math.pi, "Stop"
    elif t < 21:
        speed, angle, label = (
            1.4,
            math.pi - (t - 17) * math.pi / 4,
            "Turn while walking",
        )
    else:
        speed, angle, label = 0.0, 0.0, "Stop"
    return speed * target_direction(CONFIG, angle), angle, label


class Viewer(MotionMatchingViewer):
    def __init__(
        self, database: animation.MotionMatchingDatabase, args: argparse.Namespace
    ):
        super().__init__(database, transition_half_life=0.075, demo_input=_demo_input)
        self.character_path = DEFAULT_CHARACTER

    def setup(self):
        materials = self.create_standard_materials()
        self.scene.add_ground(scale=100, material=materials.ground)
        self.character_visual: visual.SkinVisual = visual.SkinVisual.from_fbx(
            app=self,
            path="/geno",
            fbx_path=str(self.character_path),
            skeleton_tree=self.database.motions[0].skeleton_tree,
            material=materials.pbr,
            scale=0.01,
            use_materials=False,
        )
        self.character_visual.set_color(ke.Vec4(*ke.preset_rgba(ke.ColorType.ORANGE)))
        self.apply_pose()
        self.update_camera()

    def render(self):
        super().render()
        imgui.begin("LAFAN / Geno")
        imgui.text("WASD / gamepad controls are in the Motion Matching panel")
        imgui.text("3 source ranges + mirrored copies / 60 FPS")
        imgui.end()


def main():
    args = parse_args(default_dataset=DEFAULT_DATASET, description=__doc__)
    database = load_database(args)
    if args.benchmark_search:
        benchmark_search(database)
        return
    app = Viewer(database, args)
    app.initialize(width=1920, height=1080, up_axis=ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
