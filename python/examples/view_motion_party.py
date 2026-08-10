"""Play BVH, FBX, and SMPL/AMASS motions together with sequencers."""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
from kangengine import asset, imgui, visual
from kangengine.asset.smpl import SMPLXModel, repository_smplx_model_path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_bvh_file() -> Path:
    return (
        repository_root() / "assets/external/SMPL_AMASS_T_HDM_bk_01-01_01_120_poses.bvh"
    )


def default_fbx_file() -> Path:
    return repository_root() / "assets/external/Capoeira (1).fbx"


def default_amass_file() -> Path:
    return Path.home() / "Dev/dataset/AMASS/TotalCapture/s3/walking2_poses.npz"


def _shift_root(state, offset: tuple[float, float, float]):
    root = state.root_translation()
    state.set_root_translation(
        [root.x + offset[0], root.y + offset[1], root.z + offset[2]]
    )
    return state


class MultiMotionViewer(ke.App):
    def __init__(
        self,
        bvh_file: Path,
        fbx_file: Path,
        smpl_motion_file: Path,
        fbx_clip_index: int,
        bvh_scale: float,
        fbx_scale: float,
        smpl_scale: float,
    ):
        super().__init__()
        self.bvh_file = bvh_file
        self.fbx_file = fbx_file
        self.smpl_motion_file = smpl_motion_file
        self.fbx_clip_index = fbx_clip_index
        self.bvh_scale = bvh_scale
        self.fbx_scale = fbx_scale
        self.smpl_scale = smpl_scale

    def setup(self):
        self.show_bvh = True
        self.show_fbx = True
        self.show_smpl = True

        materials = self.create_standard_materials()
        self.scene.add_ground(scale=30.0, material=materials.ground)
        self.set_camera_view([0.0, 1.8, 7.5], [0.0, 0.9, 0.0])

        self._setup_bvh(materials)
        self._setup_fbx()
        self._setup_smpl(materials)

        self.motions = {
            "BVH": self.bvh_motion,
            "FBX": self.fbx_motion,
            "SMPL/AMASS": self.smpl_motion,
        }
        timeline_motion = max(
            self.motions.values(),
            key=lambda motion: motion.num_frames() / motion.fps(),
        )
        self.editor = ke.motion_module.MotionEditor(
            timeline_motion,
            motion_name="BVH + FBX + SMPL/AMASS",
        )
        self.editor.panel.set_motions(
            list(self.motions),
            [motion.num_frames() for motion in self.motions.values()],
            [motion.fps() for motion in self.motions.values()],
        )
        self.playback_controller.add_target(self.editor)
        self._apply_all_motion_times()

        for label, motion in self.motions.items():
            print(
                f"{label} loaded: frames={motion.num_frames()} "
                f"joints={motion.num_joints()} fps={motion.fps():.3f}"
            )
        self.check_error()

    def _setup_bvh(self, materials):
        self.bvh_motion: ke.animation.SkeletonMotion = asset.BVHLoader.load_motion(
            str(self.bvh_file), self.bvh_scale
        )
        config = visual.SkeletalVisualConfig(
            bone_color=ke.Vec4(0.35, 0.75, 1.0, 1.0),
            joint_color=ke.Vec4(1.0, 0.55, 0.3, 1.0),
            bone_radius=0.025,
            joint_radius=0.035,
            show_joints=True,
        )
        state = _shift_root(self.bvh_motion.sample(0.0, True), (-2.2, 0.0, 0.0))
        self.bvh_visual = visual.SkeletalVisual.define(
            self, materials.common, "/bvh_motion", state, config
        )

    def _setup_fbx(self):
        result = asset.FBXLoader.parse(
            str(self.fbx_file),
            clip_index=self.fbx_clip_index,
            fps=-1.0,
            scale=self.fbx_scale,
        )
        self.fbx_motion: ke.animation.SkeletonMotion = result.motion
        self.fbx_surface = visual.SkinnedSurface.create_from_fbx_result(
            self, "/fbx_motion", result
        )

    def _setup_smpl(self, materials):
        info = asset.AMASSLoader.inspect(self.smpl_motion_file)
        model = SMPLXModel.load(repository_smplx_model_path("neutral"))
        self.smpl_body = model.create_body(info.betas)
        self.smpl_surface = self.smpl_body.create_visual(
            self,
            "/smpl_motion",
            materials.pbr,
            color=ke.Vec4(0.72, 0.82, 0.95, 1.0),
        )
        self.smpl_motion: ke.animation.SkeletonMotion = asset.AMASSLoader.load_motion(
            self.smpl_motion_file,
            self.smpl_body.skeleton_tree,
            model_type="smplx",
            up_axis=ke.UpAxis.Y,
            scale=self.smpl_scale,
        )

    def _apply_bvh_time(self):
        state = _shift_root(
            self.bvh_motion.sample(self.editor.player.time, self.editor.player.loop),
            (-2.5, 0.0, 0.0),
        )
        self.bvh_visual.apply_state(state)

    def _apply_fbx_time(self):
        state = _shift_root(
            self.fbx_motion.sample(self.editor.player.time, self.editor.player.loop),
            (0.0, 0.0, 0.0),
        )
        self.fbx_surface.apply_state(state)

    def _apply_smpl_time(self):
        state = self.smpl_motion.sample(
            self.editor.player.time, self.editor.player.loop
        )
        _shift_root(state, (2.5, 0.0, 0.0))
        self.smpl_surface.apply_state(state)

    def _apply_all_motion_times(self):
        self._apply_bvh_time()
        self._apply_fbx_time()
        self._apply_smpl_time()

    def pre_render(self):
        if self.editor.update(self.get_delta_time()):
            self._apply_all_motion_times()

    def render(self):
        imgui.begin("BVH + FBX + SMPL Motion")
        imgui.text("BVH skeleton    FBX skinned mesh    SMPL-X / AMASS")
        changed, self.show_bvh = imgui.checkbox("show BVH", self.show_bvh)
        if changed:
            self.bvh_visual.set_visible(self.show_bvh)
        changed, self.show_fbx = imgui.checkbox("show FBX", self.show_fbx)
        if changed:
            self.fbx_surface.set_visible(self.show_fbx)
        changed, self.show_smpl = imgui.checkbox("show SMPL", self.show_smpl)
        if changed:
            self.smpl_surface.set_visible(self.show_smpl)
        if changed:
            self._apply_smpl_time()
        imgui.end()

        if self.editor.render():
            self._apply_all_motion_times()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-file", type=Path, default=default_bvh_file())
    parser.add_argument("--fbx-file", type=Path, default=default_fbx_file())
    parser.add_argument("--smpl-motion", type=Path, default=default_amass_file())
    parser.add_argument("--fbx-clip-index", type=int, default=0)
    parser.add_argument("--bvh-scale", type=float, default=1.0)
    parser.add_argument("--fbx-scale", type=float, default=0.01)
    parser.add_argument("--smpl-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = {
        "BVH": args.bvh_file.expanduser().resolve(),
        "FBX": args.fbx_file.expanduser().resolve(),
        "SMPL motion": args.smpl_motion.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    app = MultiMotionViewer(
        paths["BVH"],
        paths["FBX"],
        paths["SMPL motion"],
        args.fbx_clip_index,
        args.bvh_scale,
        args.fbx_scale,
        args.smpl_scale,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
