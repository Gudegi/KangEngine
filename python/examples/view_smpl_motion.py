"""Play a BVH or AMASS NPZ motion on an SMPL surface with a sequencer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import asset, imgui
from kangengine.asset.smpl import (
    SMPL_JOINT_NAMES,
    SMPLH_JOINT_NAMES,
    SMPLX_JOINT_NAMES,
    SMPLModel,
    SMPLHModel,
    SMPLXModel,
    repository_smpl_model_path,
    repository_smplh_model_path,
    repository_smplx_model_path,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_motion_file() -> Path:
    return (
        repo_root()
        / "assets"
        / "external"
        / "SMPL_AMASS_T_HDM_bk_01-01_01_120_poses.bvh"
    )


class SMPLMotionViewer(ke.App):
    def __init__(
        self,
        model_file: Path,
        motion_file: Path,
        model_type: str,
        scale: float,
        betas: list[float],
    ):
        super().__init__()
        self.model_file = model_file
        self.motion_file = motion_file
        self.model_type = model_type
        self.scale = scale
        self.betas = betas

    def setup(self):
        self.show_mesh = True
        self.pose_correctives = False

        materials = self.create_standard_materials()
        self.scene.add_ground(scale=20.0, material=materials.ground)
        self.set_camera_view([0.0, 1.35, 3.2], [0.0, 0.9, 0.0])

        self.amass_info = (
            asset.AMASSLoader.inspect(self.motion_file)
            if self.motion_file.suffix.lower() == ".npz"
            else None
        )
        shape = self.betas or (
            self.amass_info.betas if self.amass_info is not None else None
        )
        model_class = {
            "smpl": SMPLModel,
            "smplh": SMPLHModel,
            "smplx": SMPLXModel,
        }[self.model_type]
        self.model = model_class.load(self.model_file)
        self.body = self.model.create_body(shape)
        self.surface = self.body.create_visual(
            self,
            "/smpl_character",
            material=materials.pbr,
            color=ke.Vec4(0.72, 0.82, 0.95, 1.0),
        )

        if self.amass_info is not None:
            self.motion = asset.AMASSLoader.load_motion(
                self.motion_file,
                self.body.skeleton_tree,
                model_type=self.model_type,
                up_axis=ke.UpAxis.Y,
                scale=self.scale,
            )
            self.motion_joint_indices = list(range(self.motion.num_joints()))
        else:
            self.motion = asset.BVHLoader.load_motion(str(self.motion_file), self.scale)
            target_names = {
                "smpl": SMPL_JOINT_NAMES,
                "smplh": SMPLH_JOINT_NAMES,
                "smplx": SMPLX_JOINT_NAMES,
            }[self.model_type]
            motion_names = self.motion.node_names()
            name_to_index = {name.lower(): i for i, name in enumerate(motion_names)}
            self.motion_joint_indices = [
                name_to_index.get(name.lower()) for name in target_names
            ]
        self.editor = ke.motion_module.MotionEditor(
            self.motion,
            motion_name=Path(self.motion.motion_name() or self.motion_file).name,
        )
        self.playback_controller.add_target(self.editor)
        self._apply_motion_time()

        print(
            f"SMPL motion loaded: model={self.model_file.name} "
            f"motion={self.motion_file.name} frames={self.motion.num_frames()} "
            f"fps={self.motion.fps():.3f}"
        )
        self.check_error()

    def _apply_motion_time(self):
        state = self.motion.sample(self.editor.player.time, self.editor.player.loop)
        rotations = np.zeros((len(self.motion_joint_indices), 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        for target, source in enumerate(self.motion_joint_indices):
            if source is None:
                continue
            rotation = state.rotation(source)
            rotations[target] = [rotation.w, rotation.x, rotation.y, rotation.z]
        root = state.root_translation()
        root_translation = np.asarray([root.x, root.y, root.z], dtype=np.float32)
        self.body.update_pose_correctives(
            self.surface,
            rotations,
            enabled=self.pose_correctives,
        )
        self.surface.apply_pose(
            root_translation=root_translation,
            local_rotations_wxyz=rotations,
        )

    def pre_render(self):
        if self.editor.update(self.get_delta_time()):
            self._apply_motion_time()

    def render(self):
        imgui.begin("SMPL Character")
        imgui.text(self.model_file.name)
        imgui.text(f"body model={self.model_type.upper()}")
        imgui.text(f"frames={self.motion.num_frames()} fps={self.motion.fps():.2f}")
        changed, self.show_mesh = imgui.checkbox("show mesh", self.show_mesh)
        if changed:
            self.surface.set_visible(self.show_mesh)
        changed, self.pose_correctives = imgui.checkbox(
            "pose correctives", self.pose_correctives
        )
        if changed:
            self._apply_motion_time()
        imgui.end()

        if self.editor.render():
            self._apply_motion_time()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--model-type",
        choices=("auto", "smpl", "smplh", "smplx"),
        default="auto",
        help="Use SMPL-X automatically for AMASS NPZ input.",
    )
    parser.add_argument(
        "--gender",
        choices=("female", "male", "neutral"),
        default=None,
        help="Repository model gender; AMASS gender is used when omitted.",
    )
    parser.add_argument("--motion", type=Path, default=default_motion_file())
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs="*", default=[])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()
    motion_file = args.motion.expanduser().resolve()
    if not motion_file.exists():
        raise FileNotFoundError(motion_file)
    gender = args.gender
    if args.model is None and gender is None and motion_file.suffix.lower() == ".npz":
        gender = asset.AMASSLoader.inspect(motion_file).gender
    gender = gender or "neutral"
    model_type = args.model_type
    if model_type == "auto":
        model_type = "smplx" if motion_file.suffix.lower() == ".npz" else "smpl"
    model_file = (
        args.model.expanduser().resolve()
        if args.model is not None
        else (
            {
                "smpl": repository_smpl_model_path,
                "smplh": repository_smplh_model_path,
                "smplx": repository_smplx_model_path,
            }[model_type](gender)
        )
    )
    if not model_file.exists():
        raise FileNotFoundError(model_file)

    app = SMPLMotionViewer(
        model_file,
        motion_file,
        model_type,
        args.scale,
        args.betas,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
