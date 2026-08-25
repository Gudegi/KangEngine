"""Visually compare a KW SkeletonMotion with its ArticulationMotion round trip."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import imgui, keys, visual


_KW_MOTION_BODY_NAMES = (
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _saved_array(value) -> np.ndarray:
    if isinstance(value, dict) and "arr" in value:
        value = value["arr"]
    return np.asarray(value, dtype=np.float32)


def _reference_rotations(tree) -> np.ndarray:
    return np.asarray(
        [
            [
                tree.local_rotation(i).w,
                tree.local_rotation(i).x,
                tree.local_rotation(i).y,
                tree.local_rotation(i).z,
            ]
            for i in range(tree.num_joints())
        ],
        dtype=np.float32,
    )


def _load_kw_motion(path: Path, tree) -> ke.animation.SkeletonMotion:
    saved = np.load(path, allow_pickle=True).item()
    if "rotation" in saved:
        rotations = _saved_array(saved["rotation"])
        roots = _saved_array(saved["root_translation"])
        fps = float(saved["fps"])
        skeleton = saved.get("skeleton_tree")
        names = (
            list(skeleton.get("node_names", ())) if isinstance(skeleton, dict) else []
        )
    else:
        rotations = _saved_array(saved["localRotations"])
        roots = _saved_array(saved["rootTranslations"])
        fps = float(np.asarray(saved["frameRate"]).item())
        # Legacy KW Walking is authored in y-up/z-forward.
        roots = roots[:, [2, 0, 1]]
        rotations = rotations[..., [0, 3, 1, 2]]
        names = list(_KW_MOTION_BODY_NAMES)

    tree_names = tree.node_names()
    if not names:
        if rotations.shape[1] != len(tree_names):
            raise ValueError(
                "Motion has no body names and does not match the KW skeleton"
            )
        names = tree_names
    if len(names) != rotations.shape[1]:
        raise ValueError("Motion body-name count does not match its rotation array")
    unknown = sorted(set(names).difference(tree_names))
    if unknown:
        raise ValueError(f"Motion contains bodies absent from KW5: {unknown}")

    full = np.repeat(_reference_rotations(tree)[None, :, :], len(rotations), axis=0)
    target_indices = [tree_names.index(name) for name in names]
    full[:, target_indices] = rotations
    return ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=tree,
        root_translations=roots,
        local_rotations_wxyz=full,
        fps=fps,
        motion_name=path.name,
    )


class ArticulationMotionComparison(ke.App):
    def __init__(self, mjcf: Path, motion: Path, separation: float):
        super().__init__()
        self.mjcf = mjcf
        self.motion_path = motion
        self.separation = separation

    def setup(self):
        self.set_camera_view([0.0, -7.0, 2.8], [0.0, 0.0, 1.0])
        self.materials = self.create_standard_materials()
        self.scene.add_ground(scale=20.0, material=self.materials.ground)

        desc = ke.asset.MJCFLoader.load(str(self.mjcf), order="DFS")
        self.source_motion = _load_kw_motion(self.motion_path, desc.skeleton_tree)
        self.layout = ke.animation.ArticulationCoordinateLayout.from_data(
            data=desc, free_root=True
        )
        self.mapper = ke.animation.ArticulationMotionMapper(self.layout)
        mapped = self.mapper.to_articulation_motion(self.source_motion)
        self.articulation_motion = mapped.motion
        self.round_trip_motion = self.mapper.to_skeleton_motion(
            self.articulation_motion
        )
        self.residuals = np.asarray(mapped.residual_angles, dtype=np.float32).reshape(
            self.source_motion.num_frames(), self.source_motion.num_joints()
        )
        self.body_names = desc.skeleton_tree.node_names()

        asset = visual.ArticulatedSurfaceAsset.from_mjcf(self.mjcf, order="DFS")
        self.source_surface = asset.create(
            self,
            "/Comparison/Source",
            material=self.materials.common,
            color=(0.25, 0.65, 1.0, 1.0),
        )
        self.round_trip_surface = asset.create(
            self,
            "/Comparison/RoundTrip",
            material=self.materials.common,
            color=(1.0, 0.5, 0.18, 1.0),
        )
        self.source_skeleton = visual.SkeletalVisual.define(
            app=self,
            material=self.materials.common,
            path="/Comparison/SourceSkeleton",
            state=self.source_motion.frame(0),
            config=visual.SkeletalVisualConfig(
                bone_color=ke.Vec4(0.1, 0.5, 1.0, 1.0), show_joints=True
            ),
        )
        self.round_trip_skeleton = visual.SkeletalVisual.define(
            app=self,
            material=self.materials.common,
            path="/Comparison/RoundTripSkeleton",
            state=self.round_trip_motion.frame(0),
            config=visual.SkeletalVisualConfig(
                bone_color=ke.Vec4(1.0, 0.35, 0.05, 1.0), show_joints=True
            ),
        )

        self.frame = 0
        self.time = 0.0
        self.paused = False
        self.show_skeletons = True
        self._apply_frame()
        print(
            f"Loaded {self.motion_path.name}: frames={self.source_motion.num_frames()} "
            f"nq={self.layout.nq} nv={self.layout.nv}"
        )
        print("Blue/left: source, orange/right: q round trip")

    @staticmethod
    def _offset_state(state, x_offset: float):
        root = state.root_translation()
        state.set_root_translation((root.x + x_offset, root.y, root.z))
        return state

    def _apply_frame(self):
        source = self._offset_state(
            self.source_motion.frame(self.frame), -0.5 * self.separation
        )
        reconstructed = self._offset_state(
            self.round_trip_motion.frame(self.frame),
            0.5 * self.separation,
        )
        self.source_surface.apply_state(source)
        self.round_trip_surface.apply_state(reconstructed)
        self.source_skeleton.apply_state(source)
        self.round_trip_skeleton.apply_state(reconstructed)

    def pre_render(self):
        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused
        if self.was_key_pressed(keys.R):
            self.time = 0.0
            self.frame = 0
            self._apply_frame()
        if self.paused:
            return
        self.time += self.get_delta_time()
        frame = int(self.time * self.source_motion.fps())
        frame %= self.source_motion.num_frames()
        if frame != self.frame:
            self.frame = frame
            self._apply_frame()

    def render(self):
        frame_residual = np.degrees(self.residuals[self.frame])
        worst = int(np.argmax(frame_residual))
        imgui.begin("Articulation Motion Comparison")
        imgui.text("Blue / left: source SkeletonMotion")
        imgui.text("Orange / right: reconstructed from canonical q")
        imgui.separator()
        imgui.text(f"Frame {self.frame} / {self.source_motion.num_frames() - 1}")
        imgui.text(f"Current max residual: {frame_residual[worst]:.4f} deg")
        imgui.text(f"Worst body: {self.body_names[worst]}")
        imgui.text(f"Clip max residual: {np.degrees(self.residuals).max():.4f} deg")
        changed, self.show_skeletons = imgui.checkbox(
            "Show skeleton overlays", self.show_skeletons
        )
        if changed:
            self.source_skeleton.set_visible(self.show_skeletons)
            self.round_trip_skeleton.set_visible(self.show_skeletons)
        imgui.text("Space: pause/resume    R: restart")
        imgui.end()


def _parse_args():
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=root / "assets/characters/kw/kw5.xml",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=root / "assets/characters/kw/motions/Walking.npy",
    )
    parser.add_argument("--separation", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.mjcf.is_file():
        raise FileNotFoundError(args.mjcf)
    if not args.motion.is_file():
        raise FileNotFoundError(args.motion)
    app = ArticulationMotionComparison(
        args.mjcf.resolve(), args.motion.resolve(), args.separation
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
