from __future__ import annotations

import argparse
from enum import Enum, auto
from pathlib import Path
import re

import numpy as np
import torch

import kangengine as ke
from kangengine import animation, asset, imgui, scene, visual
from kangengine.exports import save_motion_bvh
from kangengine.utils.batched_rotations import quat_wxyz_multiply


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


def _button(label: str, tone: str = "neutral") -> bool:
    return bool(imgui.colored_button(label, _BUTTON_COLORS[tone]))


class EditMode(Enum):
    INSPECT = auto()
    FACING = auto()
    SCALE = auto()
    BIND_POSE = auto()
    MAPPING = auto()


def _load_motion(path: Path, scale: float) -> animation.SkeletonMotion:
    if path.suffix.lower() == ".bvh":
        return asset.BVHLoader.load_motion(bvh_path=str(path), scale=scale)
    if path.suffix.lower() == ".fbx":
        return asset.FBXLoader.load_motion(fbx_path=str(path), scale=scale)
    raise ValueError("source motion must be BVH or FBX")


def _load_skeleton(path: Path, scale: float) -> animation.SkeletonTree:
    if path.suffix.lower() == ".bvh":
        return asset.BVHLoader.load_skeleton(bvh_path=str(path), scale=scale)
    if path.suffix.lower() == ".fbx":
        return asset.FBXLoader.load_skeleton(fbx_path=str(path), scale=scale)
    raise ValueError("target skeleton must be BVH or FBX")


def _default_scale(path: Path) -> float:
    return 0.01 if path.suffix.lower() == ".fbx" else 1.0


def _retarget_config_output_path(path: Path) -> Path:
    if path.name.endswith("_retarget.json"):
        return path
    if path.suffix.lower() == ".json":
        return path.with_name(f"{path.stem}_retarget.json")
    return path.with_name(f"{path.name}_retarget.json")


def _motion_output_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".bvh" else path.with_suffix(".bvh")


def _quat_tuple(rotation) -> tuple[float, float, float, float]:
    return (rotation.w, rotation.x, rotation.y, rotation.z)


def _vec3_tuple(value) -> tuple[float, float, float]:
    return (value.x, value.y, value.z)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return value or "joint"


def _joint_name_key(name: str) -> str:
    """Return a conservative key for matching common joint-name styles."""
    leaf = re.split(r"[:|/\\]", name)[-1].lower()
    leaf = re.sub(r"^(mixamorig|bip\d*|bone)[_\-. ]*", "", leaf)
    leaf = re.sub(r"left", "l", leaf)
    leaf = re.sub(r"right", "r", leaf)
    leaf = re.sub(r"[^a-z0-9]", "", leaf)
    leaf = re.sub(r"(joint|bone)$", "", leaf)
    return {"hips": "pelvis", "hip": "pelvis"}.get(leaf, leaf)


def _slice_motion(
    motion: animation.SkeletonMotion, frame_count: int
) -> animation.SkeletonMotion:
    count = min(frame_count, motion.num_frames())
    return animation.SkeletonMotion.from_arrays(
        motion.skeleton_tree,
        motion.root_translations()[:count].copy(),
        motion.local_rotations_wxyz()[:count].copy(),
        motion.fps(),
        f"{motion.motion_name()} ({count} frames)",
    )


def _scaled_motion(
    motion: animation.SkeletonMotion, scale: float
) -> animation.SkeletonMotion:
    if abs(scale - 1.0) <= 1.0e-6:
        return motion
    tree = motion.skeleton_tree
    translations = np.asarray(
        [_vec3_tuple(tree.local_translation(i)) for i in range(tree.num_joints())],
        dtype=np.float32,
    )
    rotations = np.asarray(
        [_quat_tuple(tree.local_rotation(i)) for i in range(tree.num_joints())],
        dtype=np.float32,
    )
    scaled_tree = animation.SkeletonTree(
        tree.node_names(),
        tree.parent_indices(),
        translations * scale,
        rotations,
    )
    return animation.SkeletonMotion.from_arrays(
        scaled_tree,
        motion.root_translations() * scale,
        motion.local_rotations_wxyz(),
        motion.fps(),
        motion.motion_name(),
    )


class EditableSkeleton:
    """SceneGraph joint handles whose local transforms represent a bind pose."""

    def __init__(
        self,
        app: RetargetEditor,
        label: str,
        tree: animation.SkeletonTree,
        asset_path: Path,
        bind_root: tuple[float, float, float],
        x_offset: float,
        color: tuple[float, float, float, float],
    ) -> None:
        self.app = app
        self.label = label
        self.tree = tree
        self.bind_root = bind_root
        self.color = np.asarray(color, dtype=np.float32)
        self.root_path = f"/Retarget/{label}"
        self.root_prim = scene.Prim.define_manipulation_group(
            app.scene.native, self.root_path
        )
        self.root_prim.set_local_translation(ke.Vec3(x_offset, 0.0, 0.0))
        self.joint_prims = []
        self.joint_paths: dict[str, int] = {}
        self.handles = []
        self.skin: visual.SkinVisual | None = None
        sphere = scene.Prim.create_sphere_data(0.028, 12, 8)

        for joint, name in enumerate(tree.node_names()):
            parent = tree.parent_index(joint)
            parent_path = (
                self.root_path if parent < 0 else self.joint_prims[parent].get_path()
            )
            path = f"{parent_path}/j{joint:03d}_{_safe_name(name)}"
            prim = scene.Prim.define_manipulation_group(app.scene.native, path)
            translation = bind_root if joint == 0 else tree.local_translation(joint)
            prim.set_local_translation(translation)
            prim.set_local_rotation(tree.local_rotation(joint))
            handle = app.scene.add_mesh(f"{path}/handle", sphere, app.materials.common)
            handle.set_base_color(color)
            self.joint_prims.append(prim)
            self.joint_paths[path] = joint
            self.handles.append(handle)

        if asset_path.suffix.lower() == ".fbx":
            try:
                self.skin = visual.SkinVisual.from_fbx(
                    app=app,
                    material=app.materials.common,
                    fbx_path=str(asset_path),
                    path=f"{self.root_path}/skin",
                    scale=_default_scale(asset_path),
                    use_materials=False,
                )
                self.skin.set_color((*color[:3], 0.18))
                self.skin.set_casts_shadow(False)
                self.skin.set_pickable(False)
                self.sync_skin()
            except Exception as error:
                app.status = f"Skeleton loaded; skinned mesh unavailable: {error}"

    def joint_from_prim(self, prim) -> int | None:
        if prim is None:
            return None
        target = prim.resolve_manipulation_target()
        if target is None:
            return None
        return self.joint_paths.get(target.get_path())

    def local_rotations(self) -> np.ndarray:
        rotations = np.empty((self.tree.num_joints(), 4), dtype=np.float32)
        for joint, prim in enumerate(self.joint_prims):
            rotations[joint] = _quat_tuple(prim.get_local_rotation())
        return rotations

    def sync_skin(self) -> None:
        if self.skin is None:
            return
        self.skin.apply_pose(self.bind_root, self.local_rotations())

    def set_local_rotations(
        self, rotations: dict[str, tuple[float, float, float, float]]
    ) -> None:
        indices = {name: i for i, name in enumerate(self.tree.node_names())}
        for name, value in rotations.items():
            joint = indices.get(name)
            if joint is not None:
                self.joint_prims[joint].set_local_rotation(ke.Quat(*value))

    def set_bind_root(self, value: tuple[float, float, float]) -> None:
        self.bind_root = value
        self.joint_prims[0].set_local_translation(value)

    def reset_joint(self, joint: int) -> None:
        self.joint_prims[joint].set_local_rotation(self.tree.local_rotation(joint))

    def reset_all(self) -> None:
        for joint in range(self.tree.num_joints()):
            self.reset_joint(joint)

    def set_visible(self, visible: bool) -> None:
        for handle in self.handles:
            handle.set_visible(visible)
        if self.skin is not None:
            if visible:
                self.skin.set_color((*self.color[:3], 0.18))
            else:
                self.skin.set_visible(False)
        if not visible:
            self.app.clear_debug_lines(f"{self.root_path}/bones")
            for joint in range(self.tree.num_joints()):
                self.app.clear_debug_lines(f"{self.root_path}/axis/{joint}")

    def sync_handle_scale(self) -> None:
        # Handles are editor affordances, not skeleton geometry. Cancel the
        # editable root's uniform scale so their world-space radius stays
        # constant while A or B is resized for calibration.
        inverse_scale = 1.0 / max(self.uniform_root_scale(), 1.0e-4)
        scale = ke.Vec3(inverse_scale, inverse_scale, inverse_scale)
        for handle in self.handles:
            handle.set_local_scale(scale)

    def sync_debug(self, show_axes: bool) -> None:
        starts = []
        ends = []
        for joint, prim in enumerate(self.joint_prims):
            parent = self.tree.parent_index(joint)
            if parent >= 0:
                starts.append(
                    _vec3_tuple(self.joint_prims[parent].get_world_translation())
                )
                ends.append(_vec3_tuple(prim.get_world_translation()))
            axis_path = f"{self.root_path}/axis/{joint}"
            if show_axes:
                self.app.log_debug_axes(
                    axis_path,
                    np.asarray(prim.compute_world_matrix(), dtype=np.float32),
                    0.09,
                    1.2,
                )
            else:
                self.app.clear_debug_lines(axis_path)
        if starts:
            colors = np.repeat(self.color[None], len(starts), axis=0)
            self.app.log_debug_lines(
                f"{self.root_path}/bones",
                np.asarray(starts, dtype=np.float32),
                np.asarray(ends, dtype=np.float32),
                colors,
                2.0,
            )

    def uniform_root_scale(self) -> float:
        matrix = np.asarray(self.root_prim.compute_local_matrix(), dtype=np.float32)
        scales = np.linalg.norm(matrix[:3, :3], axis=0)
        return float(np.mean(scales))

    def force_uniform_root_scale(self) -> float:
        value = max(self.uniform_root_scale(), 1.0e-4)
        self.root_prim.set_local_scale(ke.Vec3(value, value, value))
        return value


class RetargetEditor(ke.App):
    def __init__(
        self,
        source_path: Path | None = None,
        target_path: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.source_path_text = "" if source_path is None else str(source_path)
        self.target_path_text = "" if target_path is None else str(target_path)
        self.config_path_text = (
            "retarget_config_retarget.json" if config_path is None else str(config_path)
        )
        self.motion_path_text = "retargeted_motion.bvh"

    def setup(self) -> None:
        self.set_ui_layout_mode(ke.UILayoutMode.EDITOR)
        self.materials = self.create_standard_materials()
        self.scene.add_ground(
            path="/ground", scale=20.0, material=self.materials.ground
        )
        self.set_camera_view(position=(0.0, 1.8, 6.0), target=(0.0, 1.0, 0.0))
        self.set_interaction_mode(ke.InteractionMode.INSPECT)
        self.edit_mode = EditMode.INSPECT
        self.source_motion: animation.SkeletonMotion | None = None
        self.source_edit: EditableSkeleton | None = None
        self.target_edit: EditableSkeleton | None = None
        self.config: animation.RetargetConfig | None = None
        self.selected_source_joint: int | None = None
        self.selected_target_joint: int | None = None
        self.selected_side: str | None = None
        self.result_motion: animation.SkeletonMotion | None = None
        self.retarget_source_motion: animation.SkeletonMotion | None = None
        self.source_preview: visual.SkeletalVisual | None = None
        self.target_preview: visual.SkeletalVisual | None = None
        self.source_preview_skin: visual.SkinVisual | None = None
        self.target_preview_skin: visual.SkinVisual | None = None
        self.preview_source_motion: animation.SkeletonMotion | None = None
        self.preview_target_motion: animation.SkeletonMotion | None = None
        self.preview_source_rotations: np.ndarray | None = None
        self.preview_target_rotations: np.ndarray | None = None
        self.sequencer = ke.MotionSequencerPanel()
        self.sequencer.set_overlay(True)
        self.sequencer.set_overlay_width_ratio(1.0)
        self.sequencer.set_playing(False)
        self.showing_result = False
        self.preview_source_only = False
        self.active_preview: str | None = None
        self.preview_needs_rebuild = True
        self.confirm_clear_all = False
        self.scale_side = "B"
        self.overlapped = False
        self.target_position_before_overlap: tuple[float, float, float] | None = None
        self.status = "Load source A and target B to begin."

        if self.source_path_text:
            self._load_source()
        if self.target_path_text:
            self._load_target()

    def _path(self, value: str) -> Path:
        return Path(value).expanduser().resolve()

    def _dialog_location(self, value: str) -> tuple[str, str]:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return str(candidate), ""
        parent = candidate.parent
        if not parent.is_dir():
            parent = Path.cwd()
        return str(parent.resolve()), candidate.name

    def _load_source(self) -> None:
        try:
            path = self._path(self.source_path_text)
            if not path.is_file():
                raise FileNotFoundError(path)
            self.source_motion = _load_motion(path, _default_scale(path))
            self.source_edit = EditableSkeleton(
                self,
                "A",
                self.source_motion.skeleton_tree,
                path,
                tuple(float(v) for v in self.source_motion.root_translations()[0]),
                -1.5,
                (0.25, 0.65, 1.0, 1.0),
            )
            self.source_path_text = str(path)
            self.status = f"Loaded source A: {path.name}"
            self._check_loaded()
        except Exception as error:
            self.status = f"Source load failed: {error}"

    def _load_target(self) -> None:
        try:
            path = self._path(self.target_path_text)
            if not path.is_file():
                raise FileNotFoundError(path)
            tree = _load_skeleton(path, _default_scale(path))
            self.target_edit = EditableSkeleton(
                self,
                "B",
                tree,
                path,
                _vec3_tuple(tree.local_translation(0)),
                1.5,
                (1.0, 0.55, 0.2, 1.0),
            )
            self.target_path_text = str(path)
            self.status = f"Loaded target B: {path.name}"
            self._check_loaded()
        except Exception as error:
            self.status = f"Target load failed: {error}"

    def _check_loaded(self) -> None:
        if self.source_edit is None or self.target_edit is None:
            return
        self.config = animation.RetargetConfig(
            joint_map={},
            source_skeleton=self.source_path_text,
            target_skeleton=self.target_path_text,
        )
        self._set_edit_mode(EditMode.INSPECT)
        self.status = "A and B are ready. Choose any edit mode."

    def _show_calibration(self) -> None:
        self.showing_result = False
        self.preview_source_only = False
        self.active_preview = None
        self.preview_needs_rebuild = True
        if self.source_edit is not None:
            self.source_edit.set_visible(True)
        if self.target_edit is not None:
            self.target_edit.set_visible(True)
        if self.source_preview is not None:
            self.source_preview.set_visible(False)
        if self.target_preview is not None:
            self.target_preview.set_visible(False)
        if self.source_preview_skin is not None:
            self.source_preview_skin.set_visible(False)
        if self.target_preview_skin is not None:
            self.target_preview_skin.set_visible(False)

    def _clear_preview(self) -> None:
        self.sequencer.set_playing(False)
        self._show_calibration()
        self.clear_selection()
        self._remove_preview_visuals()
        self.result_motion = None
        self.retarget_source_motion = None
        self.preview_source_motion = None
        self.preview_target_motion = None
        self.preview_source_rotations = None
        self.preview_target_rotations = None
        self.sequencer.set_current_time(0.0)
        self.status = "Preview and computed retarget motion cleared."

    def _remove_preview_visuals(self) -> None:
        # Bridges retain render-component and texture state in addition to
        # their scene prims. Release those owners before the parent fallback.
        for preview in (self.source_preview_skin, self.target_preview_skin):
            if preview is not None:
                preview.remove()
        for preview in (self.source_preview, self.target_preview):
            if preview is not None:
                preview.remove()
        self.remove_prim("/RetargetPreview")
        self.source_preview = None
        self.target_preview = None
        self.source_preview_skin = None
        self.target_preview_skin = None
        self.active_preview = None
        self.preview_needs_rebuild = False

    def _clear_all(self) -> None:
        self._clear_preview()
        self.clear_selection()
        self.clear_debug_lines("/Retarget/A/bones")
        self.clear_debug_lines("/Retarget/B/bones")
        self.clear_debug_points("/Retarget/selection/A")
        self.clear_debug_points("/Retarget/selection/B")
        self.remove_prim("/Retarget")
        self.source_motion = None
        self.source_edit = None
        self.target_edit = None
        self.config = None
        self.selected_source_joint = None
        self.selected_target_joint = None
        self.selected_side = None
        self.overlapped = False
        self.target_position_before_overlap = None
        self.source_path_text = ""
        self.target_path_text = ""
        self.config_path_text = ""
        self.motion_path_text = ""
        self.edit_mode = EditMode.INSPECT
        self.scale_side = "B"
        self.set_interaction_mode(ke.InteractionMode.INSPECT)
        self.confirm_clear_all = False
        self.status = "Editor cleared. Load source A and target B to begin."

    def _set_edit_mode(self, mode: EditMode) -> None:
        self.edit_mode = mode
        self._show_calibration()
        self.clear_selection()
        if mode in (EditMode.INSPECT, EditMode.MAPPING):
            self.set_interaction_mode(ke.InteractionMode.INSPECT)
        else:
            self.set_interaction_mode(ke.InteractionMode.EDIT)
        if mode == EditMode.FACING and self.target_edit is not None:
            self.set_gizmo_operation(ke.GizmoOperation.ROTATE)
            self.set_gizmo_space(ke.GizmoSpace.WORLD)
            self.select_prim(self.target_edit.root_prim)
        elif mode == EditMode.SCALE and self.target_edit is not None:
            self._select_scale_root(self.scale_side)
        elif mode == EditMode.BIND_POSE:
            self.set_gizmo_operation(ke.GizmoOperation.ROTATE)
            self.set_gizmo_space(ke.GizmoSpace.LOCAL)
            self.status = "Click any A/B joint handle and rotate it in local space."
        elif mode == EditMode.MAPPING:
            self.status = "Select one A joint and one B joint, then map the pair."

    def _select_scale_root(self, side: str) -> None:
        editable = self.source_edit if side == "A" else self.target_edit
        if editable is None:
            return
        self.scale_side = side
        self.set_gizmo_operation(ke.GizmoOperation.SCALE)
        self.set_gizmo_space(ke.GizmoSpace.LOCAL)
        self.select_prim(editable.root_prim)
        self.status = f"Scaling {side}; uniform scale is enforced."

    def _toggle_overlap(self) -> None:
        if self.source_edit is None or self.target_edit is None:
            return
        target_root = self.target_edit.root_prim
        if self.overlapped and self.target_position_before_overlap is not None:
            target_root.set_world_translation(self.target_position_before_overlap)
            self.overlapped = False
            self.status = "Restored B root position."
            return

        self.target_position_before_overlap = _vec3_tuple(
            target_root.get_world_translation()
        )
        source_joint = self.source_edit.joint_prims[0].get_world_translation()
        target_joint = self.target_edit.joint_prims[0].get_world_translation()
        target_position = target_root.get_world_translation()
        target_root.set_world_translation(
            (
                target_position.x + source_joint.x - target_joint.x,
                target_position.y + source_joint.y - target_joint.y,
                target_position.z + source_joint.z - target_joint.z,
            )
        )
        self.overlapped = True
        self.status = "Moved B root onto A root."

    def _sync_config(self) -> animation.RetargetConfig:
        if self.source_edit is None or self.target_edit is None or self.config is None:
            raise RuntimeError("source and target are not ready")
        source_rotations = self.source_edit.local_rotations()
        target_rotations = self.target_edit.local_rotations()
        facing = torch.tensor(
            _quat_tuple(self.target_edit.root_prim.get_local_rotation()),
            dtype=torch.float32,
        )
        target_rotations[0] = quat_wxyz_multiply(
            facing, torch.from_numpy(target_rotations[0])
        ).numpy()
        self.config.source_bind_local_wxyz = {
            name: tuple(float(v) for v in source_rotations[i])
            for i, name in enumerate(self.source_edit.tree.node_names())
        }
        self.config.target_bind_local_wxyz = {
            name: tuple(float(v) for v in target_rotations[i])
            for i, name in enumerate(self.target_edit.tree.node_names())
        }
        source_scale = self.source_edit.uniform_root_scale()
        target_scale = self.target_edit.uniform_root_scale()
        # The gizmo scales are the factors needed to display A and B at the
        # same size: source_scale * A == target_scale * B. Therefore B/A,
        # which scales source root motion into target units, is A-scale/B-scale.
        self.config.translation_scale = source_scale / target_scale
        # Bind roots belong to the motion/skeleton coordinate systems. The
        # editor's side-by-side and overlap transforms are viewport-only.
        self.config.source_bind_root = self.source_edit.bind_root
        self.config.target_bind_root = self.target_edit.bind_root
        return self.config

    def _load_config(self) -> None:
        try:
            config_path = self._path(self.config_path_text)
            config = animation.RetargetConfig.load(config_path)
            if not config.source_skeleton or not config.target_skeleton:
                raise ValueError(
                    "config must contain source_skeleton and target_skeleton paths"
                )

            def resolve_asset_path(value: str) -> Path:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = config_path.parent / path
                return path.resolve()

            source_path = resolve_asset_path(config.source_skeleton)
            target_path = resolve_asset_path(config.target_skeleton)
            if not source_path.is_file():
                raise FileNotFoundError(f"source skeleton not found: {source_path}")
            if not target_path.is_file():
                raise FileNotFoundError(f"target skeleton not found: {target_path}")

            self._clear_preview()
            self.clear_selection()
            self.remove_prim("/Retarget")
            self.source_motion = None
            self.source_edit = None
            self.target_edit = None
            self.config = None
            self.selected_source_joint = None
            self.selected_target_joint = None
            self.selected_side = None
            self.overlapped = False
            self.target_position_before_overlap = None

            self.source_path_text = str(source_path)
            self.target_path_text = str(target_path)
            self._load_source()
            self._load_target()
            if self.source_edit is None or self.target_edit is None:
                raise RuntimeError(self.status)

            self.config = config
            self.source_edit.set_local_rotations(config.source_bind_local_wxyz)
            self.target_edit.set_local_rotations(config.target_bind_local_wxyz)
            self.source_edit.set_bind_root(config.source_bind_root)
            self.target_edit.set_bind_root(config.target_bind_root)
            scale = config.translation_scale
            self.source_edit.root_prim.set_local_scale(ke.Vec3(scale, scale, scale))
            self.target_edit.root_prim.set_local_scale(ke.Vec3(1.0, 1.0, 1.0))
            self._set_edit_mode(EditMode.INSPECT)
            self.status = (
                f"Config loaded with source {source_path.name} and "
                f"target {target_path.name}."
            )
        except Exception as error:
            self.status = f"Config load failed: {error}"

    def _export_config(self) -> None:
        try:
            if not self.config_path_text.strip():
                self.config_path_text = "retarget_config_retarget.json"
            output_path = _retarget_config_output_path(
                self._path(self.config_path_text)
            )
            self.config_path_text = str(output_path)
            path = self._sync_config().save(output_path)
            self.status = f"Config saved: {path}"
        except Exception as error:
            self.status = f"Config save failed: {error}"

    def _map_pair(self) -> None:
        if (
            self.config is None
            or self.source_edit is None
            or self.target_edit is None
            or self.selected_source_joint is None
            or self.selected_target_joint is None
        ):
            self.status = "Select one source A joint and one target B joint first."
            return
        source_name = self.source_edit.tree.node_name(self.selected_source_joint)
        target_name = self.target_edit.tree.node_name(self.selected_target_joint)
        self.config.joint_map = {
            source: target
            for source, target in self.config.joint_map.items()
            if source != source_name and target != target_name
        }
        self.config.joint_map[source_name] = target_name
        self.status = f"Mapped {source_name} → {target_name}"

    def _auto_map_names(self) -> None:
        if self.config is None or self.source_edit is None or self.target_edit is None:
            return
        target_candidates: dict[str, list[str]] = {}
        for target_name in self.target_edit.tree.node_names():
            target_candidates.setdefault(_joint_name_key(target_name), []).append(
                target_name
            )

        used_targets = set(self.config.joint_map.values())
        added = 0
        ambiguous = 0
        unmatched = 0
        for source_name in self.source_edit.tree.node_names():
            if source_name in self.config.joint_map:
                continue
            candidates = [
                target_name
                for target_name in target_candidates.get(
                    _joint_name_key(source_name), []
                )
                if target_name not in used_targets
            ]
            if len(candidates) == 1:
                target_name = candidates[0]
                self.config.joint_map[source_name] = target_name
                used_targets.add(target_name)
                added += 1
            elif candidates:
                ambiguous += 1
            else:
                unmatched += 1
        self.status = (
            f"Auto-mapped {added} joints; {unmatched} unmatched, "
            f"{ambiguous} ambiguous. Existing mappings were preserved."
        )

    def _unmap_pair(self) -> None:
        if self.config is None:
            return
        source_name = (
            None
            if self.selected_source_joint is None or self.source_edit is None
            else self.source_edit.tree.node_name(self.selected_source_joint)
        )
        target_name = (
            None
            if self.selected_target_joint is None or self.target_edit is None
            else self.target_edit.tree.node_name(self.selected_target_joint)
        )
        before = len(self.config.joint_map)
        self.config.joint_map = {
            source: target
            for source, target in self.config.joint_map.items()
            if source != source_name and target != target_name
        }
        self.status = f"Removed {before - len(self.config.joint_map)} mapping(s)."

    def _sync_selection_markers(self) -> None:
        selections = (
            (
                "/Retarget/selection/A",
                self.source_edit,
                self.selected_source_joint,
                (0.2, 0.7, 1.0, 1.0),
            ),
            (
                "/Retarget/selection/B",
                self.target_edit,
                self.selected_target_joint,
                (1.0, 0.45, 0.1, 1.0),
            ),
        )
        for path, editable, joint, color in selections:
            if editable is None or joint is None:
                self.clear_debug_points(path)
                continue
            position = _vec3_tuple(editable.joint_prims[joint].get_world_translation())
            self.log_debug_points(
                path,
                np.asarray([position], dtype=np.float32),
                np.asarray([color], dtype=np.float32),
                13.0,
            )

    def _compute_retarget(self, frame_limit: int | None) -> None:
        if self.source_motion is None or self.target_edit is None:
            return
        try:
            config = self._sync_config()
            source = (
                self.source_motion
                if frame_limit is None
                else _slice_motion(self.source_motion, frame_limit)
            )
            self.result_motion = animation.Retargeter(
                source.skeleton_tree, self.target_edit.tree, config
            ).retarget_motion(source)
            self.retarget_source_motion = source
            self._preview_retarget_result(force_rebuild=True)
            self.status = (
                f"Computed and previewing {self.result_motion.num_frames()} frames."
            )
        except Exception as error:
            self.status = f"Retarget failed: {error}"

    def _preview_retarget_result(self, *, force_rebuild: bool = False) -> None:
        if self.result_motion is None or self.retarget_source_motion is None:
            self.status = "Compute a retarget result before previewing it."
            return
        if not force_rebuild and self.active_preview == "retarget":
            self.sequencer.set_current_time(0.0)
            self.sequencer.set_playing(False)
            self._apply_preview_time(0.0)
            self.status = "Retarget preview reset to its first frame."
            return
        if force_rebuild or self.preview_needs_rebuild:
            self._remove_preview_visuals()
        source = self.retarget_source_motion
        self.preview_source_only = False
        self._ensure_preview_visuals(source)
        self.sequencer.set_motions(
            ["Source", "Retargeted"],
            [source.num_frames(), self.result_motion.num_frames()],
            [source.fps(), self.result_motion.fps()],
        )
        self.sequencer.set_current_time(0.0)
        self.sequencer.set_playing(False)
        self.active_preview = "retarget"
        self.preview_needs_rebuild = False
        self.status = "Previewing source and retargeted result."

    def _preview_source(self) -> None:
        if self.source_motion is None or self.source_edit is None:
            return
        if self.active_preview == "source":
            self.sequencer.set_current_time(0.0)
            self.sequencer.set_playing(False)
            self._apply_preview_time(0.0)
            self.status = "Source preview reset to its first frame."
            return
        source = self.source_motion
        if self.preview_needs_rebuild:
            self._remove_preview_visuals()
        self.source_edit.set_visible(False)
        if self.target_edit is not None:
            self.target_edit.set_visible(False)
        self.showing_result = True
        self.preview_source_only = True
        source = _scaled_motion(source, self.source_edit.uniform_root_scale())
        self._ensure_source_preview_visual(source)
        if self.target_preview is not None:
            self.target_preview.set_visible(False)
        if self.target_preview_skin is not None:
            self.target_preview_skin.set_visible(False)
        self.preview_source_motion = source
        self.preview_source_rotations = source.local_rotations_wxyz()
        self.sequencer.set_motions(["Source"], [source.num_frames()], [source.fps()])
        self.sequencer.set_current_time(0.0)
        self.sequencer.set_playing(False)
        self.active_preview = "source"
        self.preview_needs_rebuild = False
        self._apply_preview_time(0.0)
        self.status = f"Previewing source motion: {source.num_frames()} frames."

    def _ensure_source_preview_visual(self, source: animation.SkeletonMotion) -> None:
        if self.source_preview is None:
            self.source_preview = visual.SkeletalVisual.define(
                app=self,
                material=self.materials.common,
                path="/RetargetPreview/A",
                state=source.frame(0),
                config=visual.SkeletalVisualConfig(
                    bone_color=ke.Vec4(0.25, 0.65, 1.0, 1.0), show_joints=True
                ),
            )
        source_path = self._path(self.source_path_text)
        if self.source_preview_skin is None and source_path.suffix.lower() == ".fbx":
            self.source_preview_skin = visual.SkinVisual.from_fbx(
                app=self,
                material=self.materials.common,
                fbx_path=str(source_path),
                path="/RetargetPreview/A/skin",
                scale=(
                    _default_scale(source_path) * self.source_edit.uniform_root_scale()
                ),
                use_materials=True,
            )
            self.source_preview_skin.set_pickable(False)
        self.source_preview.set_visible(True)
        if self.source_preview_skin is not None:
            self.source_preview_skin.set_visible(True)

    def _ensure_preview_visuals(self, source: animation.SkeletonMotion) -> None:
        if (
            self.source_edit is None
            or self.target_edit is None
            or self.result_motion is None
        ):
            return
        self.source_edit.set_visible(False)
        self.target_edit.set_visible(False)
        self.showing_result = True
        source = _scaled_motion(source, self.source_edit.uniform_root_scale())
        self.preview_target_motion = _scaled_motion(
            self.result_motion, self.target_edit.uniform_root_scale()
        )
        self._ensure_source_preview_visual(source)
        if self.target_preview is None:
            self.target_preview = visual.SkeletalVisual.define(
                app=self,
                material=self.materials.common,
                path="/RetargetPreview/B",
                state=self.preview_target_motion.frame(0),
                config=visual.SkeletalVisualConfig(
                    bone_color=ke.Vec4(1.0, 0.55, 0.2, 1.0), show_joints=True
                ),
            )
        target_path = self._path(self.target_path_text)
        if self.target_preview_skin is None and target_path.suffix.lower() == ".fbx":
            self.target_preview_skin = visual.SkinVisual.from_fbx(
                app=self,
                material=self.materials.common,
                fbx_path=str(target_path),
                path="/RetargetPreview/B/skin",
                scale=(
                    _default_scale(target_path) * self.target_edit.uniform_root_scale()
                ),
                use_materials=True,
            )
            self.target_preview_skin.set_pickable(False)
        self.source_preview.set_visible(True)
        self.target_preview.set_visible(True)
        if self.source_preview_skin is not None:
            self.source_preview_skin.set_visible(True)
        if self.target_preview_skin is not None:
            self.target_preview_skin.set_visible(True)
        self.preview_source_motion = source
        self.preview_source_rotations = source.local_rotations_wxyz()
        self.preview_target_rotations = (
            self.preview_target_motion.local_rotations_wxyz()
        )
        self._apply_preview_time(0.0)

    def _apply_preview_time(self, time: float) -> None:
        if self.source_preview is None or self.preview_source_motion is None:
            return
        source_frame = int(time * self.preview_source_motion.fps())
        if self.sequencer.loop():
            source_frame %= self.preview_source_motion.num_frames()
        else:
            source_frame = min(
                source_frame, self.preview_source_motion.num_frames() - 1
            )
        source_state = self.preview_source_motion.frame(source_frame)
        source_root = source_state.root_translation()
        source_x_offset = 0.0 if self.preview_source_only else -1.5
        source_state.set_root_translation(
            (source_root.x + source_x_offset, source_root.y, source_root.z)
        )
        self.source_preview.apply_state(source_state)
        if self.source_preview_skin is not None:
            assert self.preview_source_rotations is not None
            self.source_preview_skin.apply_pose(
                _vec3_tuple(source_state.root_translation()),
                self.preview_source_rotations[source_frame],
            )
        if (
            self.preview_source_only
            or self.preview_target_motion is None
            or self.target_preview is None
        ):
            return
        frame = int(time * self.preview_target_motion.fps())
        if self.sequencer.loop():
            frame %= self.preview_target_motion.num_frames()
        else:
            frame = min(frame, self.preview_target_motion.num_frames() - 1)
        target_state = self.preview_target_motion.frame(frame)
        target_root = target_state.root_translation()
        target_state.set_root_translation(
            (target_root.x + 1.5, target_root.y, target_root.z)
        )
        self.target_preview.apply_state(target_state)
        if self.target_preview_skin is not None:
            assert self.preview_target_rotations is not None
            self.target_preview_skin.apply_pose(
                _vec3_tuple(target_state.root_translation()),
                self.preview_target_rotations[frame],
            )

    def _save_motion(self) -> None:
        if self.result_motion is None:
            self.status = "Compute a preview or full motion before saving."
            return
        try:
            if not self.motion_path_text.strip():
                self.motion_path_text = "retargeted_motion.bvh"
            output_path = _motion_output_path(self._path(self.motion_path_text))
            self.motion_path_text = str(output_path)
            path = save_motion_bvh(output_path, self.result_motion)
            self.status = f"Motion saved: {path}"
        except Exception as error:
            self.status = f"Motion save failed: {error}"

    def on_ray_picked(self, result) -> None:
        if not result.hit or result.prim is None:
            return
        if self.edit_mode in (EditMode.FACING, EditMode.SCALE):
            manipulation_target = result.prim.resolve_manipulation_target()
            editable = (
                self.source_edit
                if self.edit_mode == EditMode.SCALE and self.scale_side == "A"
                else self.target_edit
            )
            if editable is not None and (
                manipulation_target is None
                or manipulation_target.get_path() != editable.root_prim.get_path()
            ):
                self.select_prim(editable.root_prim)
            return
        if self.source_edit is not None and not self.showing_result:
            joint = self.source_edit.joint_from_prim(result.prim)
            if joint is not None:
                self.set_gizmo_operation(ke.GizmoOperation.ROTATE)
                self.set_gizmo_space(ke.GizmoSpace.LOCAL)
                self.selected_source_joint = joint
                self.selected_side = "A"
                self.status = f"Selected A: {self.source_edit.tree.node_name(joint)}"
                return
        if self.target_edit is not None and not self.showing_result:
            joint = self.target_edit.joint_from_prim(result.prim)
            if joint is not None:
                self.set_gizmo_operation(ke.GizmoOperation.ROTATE)
                self.set_gizmo_space(ke.GizmoSpace.LOCAL)
                self.selected_target_joint = joint
                self.selected_side = "B"
                self.status = f"Selected B: {self.target_edit.tree.node_name(joint)}"

    def pre_render(self) -> None:
        if self.edit_mode == EditMode.BIND_POSE:
            # Root alignment modes use translate/scale operations. Do not let
            # those modes leak into joint editing on a later frame or pick.
            self.set_gizmo_operation(ke.GizmoOperation.ROTATE)
            self.set_gizmo_space(ke.GizmoSpace.LOCAL)
        if self.edit_mode == EditMode.SCALE:
            if self.source_edit is not None:
                self.source_edit.force_uniform_root_scale()
            if self.target_edit is not None:
                self.target_edit.force_uniform_root_scale()
        if self.source_edit is not None:
            self.source_edit.sync_handle_scale()
            self.source_edit.sync_skin()
            self.source_edit.sync_debug(self.edit_mode == EditMode.BIND_POSE)
        if self.target_edit is not None:
            self.target_edit.sync_handle_scale()
            self.target_edit.sync_skin()
            self.target_edit.sync_debug(self.edit_mode == EditMode.BIND_POSE)
        if self.showing_result:
            self.clear_debug_points("/Retarget/selection/A")
            self.clear_debug_points("/Retarget/selection/B")
        else:
            self._sync_selection_markers()
        if self.showing_result:
            if self.sequencer.is_playing():
                self.sequencer.set_current_time(
                    self.sequencer.current_time()
                    + self.get_delta_time() * self.sequencer.time_scale()
                )
            self._apply_preview_time(self.sequencer.current_time())

    def _render_load(self) -> None:
        _, self.source_path_text = imgui.input_text(
            "Source motion A", self.source_path_text
        )
        imgui.same_line()
        if _button("Browse##source_motion"):
            path, _ = self._dialog_location(self.source_path_text)
            imgui.open_file_dialog(
                "source_motion", "Select Source Motion", "Motion files{.bvh,.fbx}", path
            )
        if _button("Load Source A", "primary"):
            self._load_source()
        _, self.target_path_text = imgui.input_text(
            "Target skeleton B", self.target_path_text
        )
        imgui.same_line()
        if _button("Browse##target_skeleton"):
            path, _ = self._dialog_location(self.target_path_text)
            imgui.open_file_dialog(
                "target_skeleton",
                "Select Target Skeleton",
                "Skeleton files{.bvh,.fbx}",
                path,
            )
        if _button("Load Target B", "primary"):
            self._load_target()
        _, self.config_path_text = imgui.input_text(
            "Retarget config", self.config_path_text
        )
        imgui.same_line()
        if _button("Browse##retarget_config"):
            path, _ = self._dialog_location(self.config_path_text)
            imgui.open_file_dialog(
                "retarget_config", "Select Retarget Config", "JSON files{.json}", path
            )
        if _button("Load Existing Config", "primary"):
            self._load_config()
        imgui.same_line()
        if _button("Save As...", "success"):
            path, file_name = self._dialog_location(self.config_path_text)
            imgui.open_save_dialog(
                "save_retarget_config",
                "Save Retarget Config",
                "JSON files{.json}",
                path,
                file_name,
            )

        if self.showing_result or self.result_motion is not None:
            if _button("Clear Preview", "warning"):
                self._clear_preview()
            imgui.same_line()
        if self.source_edit is not None or self.target_edit is not None:
            if _button("Clear All", "danger"):
                self.confirm_clear_all = True
        if self.confirm_clear_all:
            imgui.text("Clear source, target, calibration, mapping, and preview?")
            if _button("Confirm Clear All", "danger"):
                self._clear_all()
            imgui.same_line()
            if _button("Cancel Clear"):
                self.confirm_clear_all = False

    def _render_file_dialogs(self) -> None:
        for key, attribute in (
            ("source_motion", "source_path_text"),
            ("target_skeleton", "target_path_text"),
            ("retarget_config", "config_path_text"),
            ("save_retarget_config", "config_path_text"),
            ("save_retarget_motion", "motion_path_text"),
        ):
            finished, selected_path = imgui.display_file_dialog(key)
            if finished and selected_path is not None:
                setattr(self, attribute, selected_path)
                if key == "save_retarget_config":
                    self._export_config()
                elif key == "save_retarget_motion":
                    self._save_motion()

    def _render_mapping_editor(self) -> None:
        if self.source_edit is None or self.target_edit is None or self.config is None:
            return

        imgui.text("Source A joints")
        imgui.same_line()
        imgui.text("                         Target B joints")
        imgui.begin_child("source_joint_list", 260.0, 220.0, True)
        for joint, name in enumerate(self.source_edit.tree.node_names()):
            mapped = self.config.joint_map.get(name)
            label = f"{name} -> {mapped}" if mapped is not None else name
            if imgui.selectable(
                f"{label}##source_{joint}", self.selected_source_joint == joint
            ):
                self.selected_source_joint = joint
                self.selected_side = "A"
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("target_joint_list", 260.0, 220.0, True)
        reverse_map = {
            target: source for source, target in self.config.joint_map.items()
        }
        for joint, name in enumerate(self.target_edit.tree.node_names()):
            mapped = reverse_map.get(name)
            label = f"{name} <- {mapped}" if mapped is not None else name
            if imgui.selectable(
                f"{label}##target_{joint}", self.selected_target_joint == joint
            ):
                self.selected_target_joint = joint
                self.selected_side = "B"
        imgui.end_child()

        source_name = (
            "<none>"
            if self.selected_source_joint is None
            else self.source_edit.tree.node_name(self.selected_source_joint)
        )
        target_name = (
            "<none>"
            if self.selected_target_joint is None
            else self.target_edit.tree.node_name(self.selected_target_joint)
        )
        imgui.text(f"Selected: {source_name} -> {target_name}")
        if _button("Map Pair", "primary"):
            self._map_pair()
        imgui.same_line()
        if _button("Unmap Selected", "warning"):
            self._unmap_pair()
        imgui.same_line()
        if _button("Auto Map Names", "success"):
            self._auto_map_names()

        imgui.text(f"Current mappings ({len(self.config.joint_map)})")
        imgui.begin_child("current_joint_mappings", 0.0, 180.0, True)
        remove_source = None
        for source_name, target_name in self.config.joint_map.items():
            imgui.text(f"{source_name} -> {target_name}")
            imgui.same_line()
            if _button(f"Remove##mapping_{source_name}", "danger"):
                remove_source = source_name
        if remove_source is not None:
            del self.config.joint_map[remove_source]
            self.status = f"Removed mapping for {remove_source}."
        imgui.end_child()

    def _render_workflow(self) -> None:
        if self.source_motion is not None:
            if _button("Preview Source Only", "primary"):
                self._preview_source()
            if self.result_motion is not None:
                imgui.same_line()
                if _button("Preview Retarget Result", "success"):
                    self._preview_retarget_result()
            if self.showing_result:
                imgui.same_line()
                if _button("Back to Calibration", "warning"):
                    self._show_calibration()
        if self.source_edit is None or self.target_edit is None:
            return

        imgui.text_disabled(
            "Suggested workflow: Facing -> Scale -> Bind Pose -> Mapping -> "
            "Preview -> Export"
        )
        for mode, label in (
            (EditMode.INSPECT, "Inspect"),
            (EditMode.FACING, "Facing"),
            (EditMode.SCALE, "Scale"),
            (EditMode.BIND_POSE, "Bind Pose"),
            (EditMode.MAPPING, "Mapping"),
        ):
            active = "* " if self.edit_mode == mode and not self.showing_result else ""
            if _button(
                f"{active}{label}",
                "primary" if active else "neutral",
            ):
                self._set_edit_mode(mode)
            if mode != EditMode.MAPPING:
                imgui.same_line()

        overlap_label = "Separate A/B" if self.overlapped else "Overlap A/B"
        if _button(overlap_label, "primary"):
            self._toggle_overlap()
        imgui.same_line()
        imgui.text_disabled("Moves B root to A root; click again to restore.")

        if self.edit_mode == EditMode.BIND_POSE and not self.showing_result:
            if _button("Reset Selected Joint", "warning"):
                if (
                    self.selected_side == "A"
                    and self.selected_source_joint is not None
                    and self.source_edit is not None
                ):
                    self.source_edit.reset_joint(self.selected_source_joint)
                elif (
                    self.selected_side == "B"
                    and self.selected_target_joint is not None
                    and self.target_edit is not None
                ):
                    self.target_edit.reset_joint(self.selected_target_joint)
                elif (
                    self.selected_source_joint is not None
                    and self.source_edit is not None
                ):
                    self.source_edit.reset_joint(self.selected_source_joint)
                elif (
                    self.selected_target_joint is not None
                    and self.target_edit is not None
                ):
                    self.target_edit.reset_joint(self.selected_target_joint)
            imgui.same_line()
            if _button("Reset All Binds", "danger"):
                if self.source_edit is not None:
                    self.source_edit.reset_all()
                if self.target_edit is not None:
                    self.target_edit.reset_all()
        elif self.edit_mode == EditMode.SCALE and not self.showing_result:
            source_prefix = "* " if self.scale_side == "A" else ""
            target_prefix = "* " if self.scale_side == "B" else ""
            if _button(
                f"{source_prefix}Scale Source A",
                "primary" if self.scale_side == "A" else "neutral",
            ):
                self._select_scale_root("A")
            imgui.same_line()
            if _button(
                f"{target_prefix}Scale Target B",
                "primary" if self.scale_side == "B" else "neutral",
            ):
                self._select_scale_root("B")
            source_scale = self.source_edit.uniform_root_scale()
            target_scale = self.target_edit.uniform_root_scale()
            imgui.text(
                f"A: {source_scale:.4f}  B: {target_scale:.4f}  "
                f"translation scale A/B: {source_scale / target_scale:.4f}"
            )
        elif self.edit_mode == EditMode.MAPPING and not self.showing_result:
            self._render_mapping_editor()

        imgui.separator()
        if _button("Export Config", "success"):
            self._export_config()
        imgui.same_line()
        if _button("Compute + Preview 100 Frames", "primary"):
            self._compute_retarget(100)
        imgui.same_line()
        if _button("Compute + Preview Full Motion", "primary"):
            self._compute_retarget(None)

    def _render_result_controls(self) -> None:
        if self.result_motion is None or self.preview_source_only:
            return
        imgui.separator()
        imgui.text(
            f"Result: {self.result_motion.num_frames()} frames @ "
            f"{self.result_motion.fps():.2f} fps"
        )
        imgui.text_disabled("Playback and frame seeking are in Motion Sequencer.")
        _, self.motion_path_text = imgui.input_text(
            "Motion output", self.motion_path_text
        )
        imgui.same_line()
        if _button("Save As...##motion", "success"):
            path, file_name = self._dialog_location(self.motion_path_text)
            imgui.open_save_dialog(
                "save_retarget_motion",
                "Save Retargeted Motion",
                "BVH motion{.bvh}",
                path,
                file_name,
            )
        if _button("Save Motion", "success"):
            self._save_motion()

    def render(self) -> None:
        imgui.begin("Retarget Editor")
        if self.showing_result:
            view = (
                "SOURCE-ONLY PREVIEW"
                if self.preview_source_only
                else "RETARGET PREVIEW"
            )
        else:
            view = self.edit_mode.name
        imgui.text(f"Mode: {view}")
        self._render_load()
        imgui.separator()
        self._render_workflow()
        self._render_result_controls()
        if self.config is not None:
            imgui.text(f"Mapped joints: {len(self.config.joint_map)}")
        imgui.text(self.status)
        imgui.end()
        if self.showing_result:
            self.sequencer.build_panel()
        self._render_file_dialogs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_motion", type=Path, nargs="?")
    parser.add_argument("target_skeleton", type=Path, nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = RetargetEditor(args.source_motion, args.target_skeleton, args.config)
    app.initialize(width=args.width, height=args.height, hide_ui=False)
    app.start()


if __name__ == "__main__":
    main()
