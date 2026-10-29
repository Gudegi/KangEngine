"""Interactively load and compare motion-dataset clips in a Y-up scene.

Set ``KE_MOTION_DATASETS_DIR`` to the directory containing the datasets, then
use the AMASS and Ascend buttons to append clips to the current viewer.

export KE_MOTION_DATASETS_DIR=/path/to/datasets

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import kangengine as ke
from kangengine import imgui


_COLORS = (
    (0.30, 0.72, 1.00),
    (1.00, 0.55, 0.25),
    (0.45, 0.90, 0.55),
    (0.85, 0.48, 0.95),
    (1.00, 0.82, 0.30),
    (0.35, 0.90, 0.88),
)


@dataclass
class LoadedMotion:
    dataset: str
    path: Path
    motion: object
    visual: object
    offset: tuple[float, float, float]

    @property
    def label(self) -> str:
        return f"{self.dataset}: {self.path.name}"


class DatasetMotionViewer(ke.App):
    """Append-only dataset motion viewer with shared playback controls."""

    def __init__(
        self,
        *,
        amass_model_type: str,
        amass_model_path: str | None,
        ascend_model_path: str | None,
        spacing: float,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.amass_model_type = amass_model_type
        self.amass_model_path = amass_model_path
        self.ascend_model_path = ascend_model_path
        self.spacing = float(spacing)
        self.max_frames = int(max_frames)
        self.rendered_frames = 0
        self.loaded: list[LoadedMotion] = []
        self.status = "Choose a dataset and append a motion."

    def setup(self) -> None:
        self.set_ui_layout_mode(ke.UILayoutMode.EDITOR)
        self.set_camera_view([7.0, 4.0, 10.0], [2.0, 1.0, 0.0])
        self.materials = self.create_standard_materials()
        self.scene.add_ground(
            path="/DatasetViewer/Ground",
            scale=30.0,
            material=self.materials.ground,
        )
        self.sequencer = ke.MotionSequencerPanel()
        self.sequencer.set_overlay(True)
        self.sequencer.set_overlay_width_ratio(1.0)
        self.sequencer.set_loop(True)
        self.sequencer.set_playing(False)
        self._reset_sequencer()

    def _dataset_root(self, child: str) -> str:
        root = os.environ.get("KE_MOTION_DATASETS_DIR")
        if root is None:
            return str(Path.cwd())
        candidate = Path(root).expanduser() / child
        return str(candidate if candidate.is_dir() else Path(root).expanduser())

    def _open_dataset_dialog(self, dataset: str) -> None:
        if dataset == "amass":
            imgui.open_file_dialog(
                "dataset_amass",
                "Select AMASS Motion",
                "AMASS motion{.npz}",
                self._dataset_root("AMASS"),
            )
        else:
            imgui.open_file_dialog(
                "dataset_ascend",
                "Select Ascend Motion",
                "Ascend motion{.npy,.npz,.pkl}",
                self._dataset_root("Ascend"),
            )

    def _load(self, dataset: str, path: str | Path) -> None:
        source_path = Path(path).expanduser().resolve()
        if dataset == "amass":
            config = ke.animation.datasets.AMASSConfig(
                model_type=self.amass_model_type,
                model_path=self.amass_model_path,
            )
        else:
            config = ke.animation.datasets.AscendConfig(
                smpl_model_path=self.ascend_model_path,
            )
        motion = ke.animation.datasets.load_motion(
            source_path,
            dataset=dataset,
            target_world=ke.animation.CoordinateSystem.Y_UP_Z_FORWARD,
            config=config,
        )
        index = len(self.loaded)
        offset = (index * self.spacing, 0.0, 0.0)
        color = _COLORS[index % len(_COLORS)]
        state = self._sample_with_offset(motion, 0.0, offset)
        visual = ke.visual.SkeletalVisual.define(
            app=self,
            material=self.materials.common,
            path=f"/DatasetViewer/Motions/Motion_{index}",
            state=state,
            config=ke.visual.SkeletalVisualConfig(
                bone_color=ke.Vec4(*color, 1.0),
                joint_color=ke.Vec4(1.0, 0.82, 0.28, 1.0),
                bone_radius=0.018,
                joint_radius=0.026,
                show_joints=True,
            ),
        )
        self.loaded.append(LoadedMotion(dataset, source_path, motion, visual, offset))
        self._update_sequencer()
        self.sequencer.set_playing(True)
        self.status = (
            f"Loaded {source_path.name}: {motion.num_frames()} frames, "
            f"{motion.num_joints()} joints, {motion.fps():g} fps"
        )

    def _sample_with_offset(self, motion, time: float, offset):
        state = motion.sample(time, loop=self.sequencer.loop())
        root = state.root_translation()
        state.set_root_translation(
            (root.x + offset[0], root.y + offset[1], root.z + offset[2])
        )
        return state

    def _reset_sequencer(self) -> None:
        self.sequencer.set_motion("No motions", 1, 30.0)
        self.sequencer.set_current_time(0.0)

    def _update_sequencer(self) -> None:
        self.sequencer.set_motions(
            [item.label for item in self.loaded],
            [item.motion.num_frames() for item in self.loaded],
            [item.motion.fps() for item in self.loaded],
        )

    def _clear_all(self) -> None:
        for item in self.loaded:
            item.visual.remove()
        count = len(self.loaded)
        self.loaded.clear()
        if hasattr(self, "sequencer"):
            self.sequencer.set_playing(False)
            self._reset_sequencer()
        self.status = f"Removed {count} motion(s)."

    def _render_dialogs(self) -> None:
        for key, dataset in (
            ("dataset_amass", "amass"),
            ("dataset_ascend", "ascend"),
        ):
            finished, selected_path = imgui.display_file_dialog(key)
            if finished and selected_path is not None:
                try:
                    self._load(dataset, selected_path)
                except Exception as error:
                    self.status = f"Failed to load {dataset}: {error}"

    def pre_render(self) -> None:
        if not self.loaded:
            return
        current_time = float(self.sequencer.current_time())
        if self.sequencer.is_playing():
            current_time += self.get_delta_time() * self.sequencer.time_scale()
            duration = float(self.sequencer.duration())
            if duration > 0.0:
                if self.sequencer.loop():
                    current_time %= duration
                elif current_time >= duration:
                    current_time = duration
                    self.sequencer.set_playing(False)
            self.sequencer.set_current_time(current_time)
        for item in self.loaded:
            item.visual.apply_state(
                self._sample_with_offset(item.motion, current_time, item.offset)
            )

    def render(self) -> None:
        imgui.begin("Dataset Motions")
        imgui.text("Application coordinates: Y-up / +Z forward")
        imgui.text(
            f"KE_MOTION_DATASETS_DIR: {os.environ.get('KE_MOTION_DATASETS_DIR', '<not set>')}"
        )
        if imgui.button("Load AMASS..."):
            self._open_dataset_dialog("amass")
        imgui.same_line()
        if imgui.button("Load Ascend..."):
            self._open_dataset_dialog("ascend")
        imgui.same_line()
        if imgui.button("Remove All Motions"):
            self._clear_all()
        imgui.separator()
        imgui.text(f"Loaded motions: {len(self.loaded)}")
        for index, item in enumerate(self.loaded):
            imgui.text(
                f"{index + 1}. {item.label} | {item.motion.num_frames()} frames "
                f"@ {item.motion.fps():g} fps"
            )
        imgui.separator()
        imgui.text(self.status)
        imgui.end()
        self._render_dialogs()
        if self.loaded:
            self.sequencer.build_panel()
        self.rendered_frames += 1
        if self.max_frames > 0 and self.rendered_frames >= self.max_frames:
            self.request_close()

    def cleanup(self) -> None:
        self._clear_all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--spacing", type=float, default=2.5)
    parser.add_argument(
        "--amass-model-type", choices=("smpl", "smplh", "smplx"), default="smplx"
    )
    parser.add_argument("--amass-model-path")
    parser.add_argument("--ascend-model-path")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()

    app = DatasetMotionViewer(
        amass_model_type=args.amass_model_type,
        amass_model_path=args.amass_model_path,
        ascend_model_path=args.ascend_model_path,
        spacing=args.spacing,
        max_frames=args.frames,
    )
    app.initialize(
        width=args.width,
        height=args.height,
        hide_ui=False,
        up_axis=ke.UpAxis.Y,
        headless=args.headless,
    )
    app.start()


if __name__ == "__main__":
    main()
