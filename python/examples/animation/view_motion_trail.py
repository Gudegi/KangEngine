"""Inspect one FBX motion on a character and six independently posed clones."""

import argparse
from pathlib import Path

import kangengine as ke
from kangengine import imgui


CLONE_COUNT = 6
POSE_SPACING = 2.0
POSE_COLORS = (
    ke.ColorType.PASTEL_PURPLE,
    ke.ColorType.PASTEL_BLUE,
    ke.ColorType.PASTEL_CYAN,
    ke.ColorType.PASTEL_GREEN,
    ke.ColorType.PASTEL_YELLOW,
    ke.ColorType.PASTEL_ORANGE,
    ke.ColorType.PASTEL_RED,
)
DEFAULT_FBX = Path(__file__).resolve().parents[3] / "assets/external/Capoeira (1).fbx"


class MotionTrailApp(ke.App):
    def __init__(self, fbx_path: Path):
        super().__init__()
        self.fbx_path = fbx_path

    def setup(self):
        self.interval = 0.2
        self.separate_poses = True
        self.controls_docked = False
        self.timeline_docked = False
        self.set_ui_layout_mode(mode=ke.UILayoutMode.EDITOR)
        self.set_camera_view(position=[0, 2.4, 11], target=[0, 0.9, 0])
        self.scene.add_ground(path="/ground", scale=25)

        # Import once; every clone shares the same source meshes and material.
        result: ke.asset.FBXImportResult = ke.asset.FBXLoader.parse(
            fbx_path=str(self.fbx_path), clip_index=0, fps=-1.0, scale=0.01,
        )
        self.motion: ke.animation.SkeletonMotion = result.motion
        material: ke.material.PBRMaterial = self.create_pbr_material(roughness=0.7)
        self.character: ke.visual.SkinnedSurface = ke.visual.SkinnedSurface.create_from_fbx_result(
            app=self, path="/trail/current", result=result,
            material=material, use_imported_materials=False,
        )
        self.clones = [
            self.character.create_instance(path=f"/trail/previous_{index}")
            for index in range(CLONE_COUNT)
        ]
        for clone, color in zip(self.clones, POSE_COLORS[:-1]):
            clone.set_color(ke.preset_rgba(color))
        self.character.set_color(ke.preset_rgba(POSE_COLORS[-1]))

        self.editor = ke.motion_module.MotionEditor(
            motion=self.motion, motion_name=self.fbx_path.name,
        )
        self.editor.panel.set_overlay(False)
        self.playback_controller.add_target(self.editor)
        self.playback_controller.set_playing(False)
        self.editor.player.time = min(3.0, self.motion.duration())
        self.apply_trail()

    def apply_trail(self):
        current_time = max(0.0, self.editor.player.time)
        self.sample_times = []

        # Clones show earlier poses, ordered from oldest to newest.
        for index, clone in enumerate(self.clones):
            time_offset = (CLONE_COUNT - index) * self.interval
            # Clamp at the start instead of wrapping ghosts to the clip's end.
            sample_time = max(0.0, current_time - time_offset)
            self.sample_times.append(sample_time)
            self._apply_pose(clone, time=sample_time, pose_index=index)

        # The original character shows the current pose.
        self.sample_times.append(current_time)
        self._apply_pose(self.character, time=current_time, pose_index=CLONE_COUNT)

    def _apply_pose(
        self, surface: ke.visual.SkinnedSurface, *, time: float, pose_index: int
    ):
        state: ke.animation.SkeletonState = self.motion.sample(time=time, loop=False)
        if self.separate_poses:
            root = state.root_translation()
            state.set_root_translation(
                [(pose_index - CLONE_COUNT / 2) * POSE_SPACING, root.y, 0.0],
            )
        surface.apply_state(state=state)

    def pre_render(self):
        if self.editor.update(dt=self.get_delta_time()):
            self.apply_trail()
        if self.hide_ui:
            return
        if not self.controls_docked:
            self.controls_docked = imgui.dock_window("Motion Trail", "Inspector")
        if not self.timeline_docked:
            self.timeline_docked = imgui.dock_window(
                "Motion Sequencer", "Viewport", position="below", ratio=0.3,
            )
        if imgui.begin("Motion Trail"):
            imgui.text("1 motion / 7 poses")
            imgui.text_wrapped("Purple -> red: oldest -> current.")
            imgui.text_wrapped("Space: play/pause. Scrub the timeline to change time.")
            imgui.text("Interval (seconds)")
            imgui.set_next_item_width(-1)
            changed, self.interval = imgui.slider_float(
                "##interval", self.interval, 0.02, 0.5,
            )
            layout_changed, self.separate_poses = imgui.checkbox(
                "Separate poses", self.separate_poses,
            )
            if layout_changed:
                for clone in self.clones:
                    clone.set_alpha(1.0 if self.separate_poses else 0.3)
            if changed or layout_changed:
                self.apply_trail()
            imgui.text("Oldest -> current (seconds)")
            for index, time in enumerate(self.sample_times):
                imgui.text(f"{index}: {time:.2f}")
        imgui.end()
        if self.editor.render():
            self.apply_trail()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx-path", type=Path, default=DEFAULT_FBX)
    args = parser.parse_args()
    path = args.fbx_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    app = MotionTrailApp(path)
    try:
        app.initialize(width=1440, height=900, up_axis=ke.UpAxis.Y)
        app.start()
    finally:
        app.close()


if __name__ == "__main__":
    main()
