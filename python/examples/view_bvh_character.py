"""View a BVH motion with the C++ SkeletalVisual."""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
from kangengine import asset, imgui, visual


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_bvh_file() -> Path:
    return (
        repo_root()
        / "assets"
        / "external"
        / "SMPL_AMASS_T_HDM_bk_01-01_01_120_poses.bvh"
    )


class BVHCharacterViewer(ke.App):
    def __init__(self, bvh_file: Path, scale: float):
        super().__init__()
        self.bvh_file = str(bvh_file)
        self.scale = scale

    def setup(self):
        self.show_skeleton = True
        self.show_joints = True

        self.shaders = self.create_standard_shaders()
        self.skeleton_shader = self.shaders.common
        self.add_ground(shader=self.shaders.ground)
        self.set_camera_view([0.0, 1.2, 3.0], [0.0, 0.9, 0.0])

        self.motion = asset.BVHLoader.load_motion(self.bvh_file, self.scale)
        motion_name = Path(self.motion.motion_name() or self.bvh_file).name
        self.editor = ke.MotionEditor(self.motion, motion_name=motion_name)

        config = visual.SkeletalVisualConfig()
        config.bone_radius = 0.03
        config.joint_radius = 0.025
        config.visible = self.show_skeleton
        config.show_joints = self.show_joints
        self.skeleton_visual = visual.SkeletalVisual.define(
            self,
            self.skeleton_shader,
            "/bvh_skeleton",
            self.motion,
            0.0,
            True,
            config,
        )

        print(
            f"BVH loaded: {Path(self.bvh_file).name} "
            f"joints={self.motion.num_joints()} "
            f"frames={self.motion.num_frames()} "
            f"fps={self.motion.fps():.3f}"
        )
        self.check_error()

    def preRender(self):
        if self.editor.update(self.get_delta_time()):
            self._apply_motion_time()

    def render(self):
        imgui.begin("BVH Character")
        imgui.text(Path(self.bvh_file).name)
        imgui.text(
            f"joints={self.motion.num_joints()} "
            f"frames={self.motion.num_frames()} "
            f"fps={self.motion.fps():.2f}"
        )
        changed, self.show_skeleton = imgui.checkbox(
            "show skeleton", self.show_skeleton
        )
        if changed:
            self.skeleton_visual.set_visible(self.show_skeleton)
        changed, self.show_joints = imgui.checkbox("show joints", self.show_joints)
        if changed:
            self.skeleton_visual.set_show_joints(self.show_joints)
        imgui.end()

        if self.editor.render():
            self._apply_motion_time()

    def _apply_motion_time(self):
        self.skeleton_visual.apply_motion(
            self.motion,
            self.editor.player.time,
            self.editor.player.loop,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bvh", nargs="?", type=Path, default=default_bvh_file())
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()
    app = BVHCharacterViewer(args.bvh, args.scale)
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
