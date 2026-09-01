"""Shared source-motion plumbing for the angle and IK retarget editors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import kangengine as ke
from kangengine import imgui
from kangengine.animation.retarget import load_retarget_motion


def _color(color_type: ke.ColorType) -> ke.Vec4:
    color = ke.ColorLibrary.get(color_type)
    return ke.Vec4(color.r, color.g, color.b, color.a)


_BUTTON_COLORS = {
    "neutral": _color(ke.ColorType.SLATE_GRAY),
    "primary": _color(ke.ColorType.STEEL_BLUE),
    "success": _color(ke.ColorType.SEA_GREEN),
    "warning": _color(ke.ColorType.ORANGE),
    "danger": _color(ke.ColorType.CRIMSON),
}


class RetargetViewMode(Enum):
    """Shared high-level viewport modes used by both retarget editors."""

    EDIT_MAPPING = "edit_mapping"
    PREVIEW_SOURCE = "preview_source"
    PREVIEW_RESULT = "preview_result"


def retarget_view_mode_buttons(
    mode: RetargetViewMode,
    *,
    source_available: bool,
    result_available: bool,
) -> RetargetViewMode:
    """Draw the common Mapping/Source/Result mode switch."""

    options = (
        (RetargetViewMode.EDIT_MAPPING, "Edit Mapping", True),
        (RetargetViewMode.PREVIEW_SOURCE, "Preview Source", source_available),
        (RetargetViewMode.PREVIEW_RESULT, "Preview Result", result_available),
    )
    selected = mode
    for index, (candidate, label, enabled) in enumerate(options):
        if index:
            imgui.same_line()
        active = candidate is mode
        button_label = f"* {label}" if active else label
        if enabled and retarget_button(
            button_label, "primary" if active else "neutral"
        ):
            selected = candidate
        elif not enabled:
            imgui.text_disabled(label)
    return selected


def retarget_button(label: str, tone: str = "neutral") -> bool:
    """Draw a consistently styled retarget-editor button."""
    try:
        color = _BUTTON_COLORS[tone]
    except KeyError as error:
        raise ValueError(f"unknown retarget button tone: {tone}") from error
    return bool(ke.imgui.colored_button(label, color))


def dialog_location(value: str) -> tuple[str, str]:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return str(candidate), ""
    parent = candidate.parent
    if not parent.is_dir():
        parent = Path.cwd()
    return str(parent.resolve()), candidate.name


def retarget_assets_directory() -> Path:
    """Return the repository's shared retarget-config directory."""

    return Path(__file__).resolve().parents[3] / "assets" / "retarget"


def retarget_path_input(
    label: str, value: str, *, width: float = 360.0
) -> tuple[bool, str]:
    """Draw a compact path field, leaving room for actions on the same row."""

    imgui.set_next_item_width(float(width))
    return imgui.input_text(label, value)


class RetargetFileDialogs:
    """Small adapter around the shared ImGui file-dialog lifecycle."""

    @staticmethod
    def open_file(key: str, title: str, filters: str, current_value: str = "") -> None:
        directory, _ = dialog_location(current_value)
        ke.imgui.open_file_dialog(key, title, filters, directory)

    @staticmethod
    def open_save(key: str, title: str, filters: str, current_value: str = "") -> None:
        directory, file_name = dialog_location(current_value)
        ke.imgui.open_save_dialog(key, title, filters, directory, file_name)

    @staticmethod
    def selected(key: str) -> str | None:
        finished, selected_path = ke.imgui.display_file_dialog(key)
        return selected_path if finished else None


class RetargetMotionPreview:
    """Own one scalable skeletal preview and its sampled motion state."""

    def __init__(
        self,
        app,
        material,
        path: str,
        motion,
        *,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        config=None,
        joint_pickable: bool = False,
    ) -> None:
        self.app = app
        self.path = path
        self.source_motion = motion
        self.motion = motion
        self.offset = tuple(float(value) for value in offset)
        self.scale = 1.0
        visual_config = ke.visual.SkeletalVisualConfig() if config is None else config
        self.visual = ke.visual.SkeletalVisual.define(
            app=app,
            material=material,
            path=path,
            state=self._state(0),
            config=visual_config,
        )
        self.visual.set_bone_pickable(False)
        self.visual.set_joint_pickable(joint_pickable)

    def joint_from_pick(self, result) -> int | None:
        """Resolve a picked joint-sphere instance to its motion joint index."""

        if result is None or not getattr(result, "hit", False):
            return None
        if int(getattr(result, "handle", -1)) != int(self.visual.joint_handle()):
            return None
        joint = int(getattr(result, "instance_index", -1))
        if not 0 <= joint < self.motion.skeleton_tree.num_joints():
            return None
        return joint

    def _state(self, frame: int):
        index = max(0, min(int(frame), self.motion.num_frames() - 1))
        state = self.motion.frame(index)
        root = state.root_translation()
        state.set_root_translation(
            (
                root.x + self.offset[0],
                root.y + self.offset[1],
                root.z + self.offset[2],
            )
        )
        return state

    def set_motion(self, motion) -> None:
        self.source_motion = motion
        self.scale = 1.0
        self.motion = motion
        self.apply_frame(0)

    def set_scale(self, scale: float) -> None:
        factor = float(scale)
        if abs(factor - self.scale) <= 1.0e-12:
            return
        self.motion = ke.animation.scale_skeleton_motion(self.source_motion, factor)
        self.scale = factor

    def apply_frame(self, frame: int) -> None:
        self.visual.apply_state(self._state(frame))

    def apply_time(self, time: float, *, loop: bool = True) -> None:
        frame = int(float(time) * self.motion.fps())
        if loop:
            frame %= self.motion.num_frames()
        self.apply_frame(frame)

    def set_visible(self, visible: bool) -> None:
        self.visual.set_visible(visible=visible)

    def set_joint_pickable(self, pickable: bool) -> None:
        self.visual.set_joint_pickable(bool(pickable))

    def remove(self) -> bool:
        return bool(self.visual.remove())


@dataclass(frozen=True)
class RetargetSourceLoader:
    """Load one clip according to a motion-family profile."""

    profile: ke.animation.MotionSourceProfile
    target_coordinates: ke.animation.CoordinateSystem
    pair_scale: float = 1.0
    required_joints: tuple[str, ...] = ()

    def load(self, path: str | Path):
        motion = load_retarget_motion(
            path,
            self.profile,
            target_coordinate_system=self.target_coordinates,
            pair_scale=self.pair_scale,
        )
        self.validate(motion)
        return motion

    def validate(self, motion) -> None:
        available = set(motion.node_names())
        missing = sorted(set(self.required_joints) - available)
        if missing:
            raise ValueError(
                "motion is incompatible with the source profile; missing joints: "
                + ", ".join(missing)
            )


@dataclass
class RetargetSourceSettings:
    """Source import settings shared by angle and IK retarget editors."""

    coordinate_system: ke.animation.CoordinateSystem = (
        ke.animation.CoordinateSystem.Y_UP_Z_FORWARD
    )
    translation_unit_scale: float = 1.0
    has_armature_joint: bool = False

    @staticmethod
    def default_scale(path: str | Path) -> float:
        return 0.01 if Path(path).suffix.lower() == ".fbx" else 1.0

    def use_file_defaults(self, path: str | Path) -> None:
        self.translation_unit_scale = self.default_scale(path)

    def apply_profile(self, profile: ke.animation.MotionSourceProfile) -> None:
        self.coordinate_system = ke.animation.CoordinateSystem(
            profile.coordinate_system
        )
        self.translation_unit_scale = float(profile.translation_unit_scale)
        self.has_armature_joint = bool(profile.has_armature_joint)

    def profile(self, path: str | Path, *, name: str | None = None):
        source_path = Path(path).expanduser().resolve()
        return ke.animation.MotionSourceProfile(
            name=name or source_path.stem or "source",
            coordinate_system=self.coordinate_system.value,
            translation_unit_scale=self.translation_unit_scale,
            has_armature_joint=self.has_armature_joint,
            reference_skeleton=source_path,
        )

    def render(self, label: str = "Source input") -> None:
        imgui.text(label)
        imgui.same_line()
        choices = (
            (ke.animation.CoordinateSystem.Y_UP_Z_FORWARD, "Y-up Z-fwd"),
            (
                ke.animation.CoordinateSystem.Y_UP_NEG_Z_FORWARD,
                "Y-up -Z-fwd",
            ),
            (ke.animation.CoordinateSystem.Z_UP_X_FORWARD, "Z-up X-fwd"),
        )
        for index, (value, text) in enumerate(choices):
            if index:
                imgui.same_line()
            active = self.coordinate_system == value
            if retarget_button(
                f"{'* ' if active else ''}{text}##{label}_{index}",
                "primary" if active else "neutral",
            ):
                self.coordinate_system = value
        _, self.translation_unit_scale = imgui.slider_float(
            f"Import scale##{label}",
            self.translation_unit_scale,
            0.0001,
            100.0,
        )
        _, self.has_armature_joint = imgui.checkbox(
            f"Source has armature joint##{label}", self.has_armature_joint
        )
        imgui.same_line()
        imgui.text_disabled("Removes the top-level container before loading.")


@dataclass(frozen=True)
class RetargetSourcePanelAction:
    load_requested: bool = False
    selected_path: str | None = None
    scale_factor: float | None = None


@dataclass
class RetargetSourcePanel:
    """Shared source-motion import and post-load scale controls."""

    path_text: str = ""
    settings: RetargetSourceSettings | None = None
    dialog_key: str = "source_motion"

    def __post_init__(self) -> None:
        if self.settings is None:
            self.settings = RetargetSourceSettings(
                translation_unit_scale=(
                    RetargetSourceSettings.default_scale(self.path_text)
                    if self.path_text
                    else 1.0
                )
            )

    def render(
        self,
        dialogs: RetargetFileDialogs,
        *,
        label: str,
        loaded: bool,
        current_scale: float | None = None,
        filters: str = "Motion files{.bvh,.fbx,.npz,.npy,.pkl}",
        use_file_scale_on_select: bool = True,
    ) -> RetargetSourcePanelAction:
        assert self.settings is not None
        self.settings.render(f"{label} input")
        _, self.path_text = retarget_path_input(label, self.path_text)
        imgui.same_line()
        if retarget_button(f"Browse##{self.dialog_key}"):
            dialogs.open_file(
                self.dialog_key,
                f"Select {label}",
                filters,
                self.path_text,
            )
        imgui.same_line()
        load_requested = retarget_button(f"Load##{self.dialog_key}", "primary")

        selected_path = dialogs.selected(self.dialog_key)
        if selected_path is not None:
            self.path_text = selected_path
            if use_file_scale_on_select:
                self.settings.use_file_defaults(selected_path)

        scale_factor = None
        if loaded:
            scale_text = "?" if current_scale is None else f"{current_scale:g}"
            imgui.text(f"Source quick scale: {scale_text}")
            imgui.same_line()
            factors = (100.0, 10.0, 0.1, 0.01)
            for index, factor in enumerate(factors):
                if retarget_button(
                    f"x{factor:g}##{self.dialog_key}_quick_scale_{factor:g}"
                ):
                    scale_factor = factor
                if index != len(factors) - 1:
                    imgui.same_line()

        return RetargetSourcePanelAction(
            load_requested=load_requested,
            selected_path=selected_path,
            scale_factor=scale_factor,
        )


def validate_same_skeleton(reference, candidate) -> None:
    if candidate.node_names() != reference.node_names():
        raise ValueError("replacement motion joint names differ from the loaded motion")
    if candidate.parent_indices() != reference.parent_indices():
        raise ValueError("replacement motion hierarchy differs from the loaded motion")


__all__ = [
    "RetargetFileDialogs",
    "RetargetMotionPreview",
    "RetargetSourceLoader",
    "RetargetSourcePanel",
    "RetargetSourcePanelAction",
    "RetargetSourceSettings",
    "RetargetViewMode",
    "dialog_location",
    "retarget_button",
    "retarget_path_input",
    "retarget_view_mode_buttons",
    "retarget_assets_directory",
    "validate_same_skeleton",
]
