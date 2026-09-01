"""Visually edit pair-specific scale in an articulation IK retarget config.

The tool is deliberately separate from ``angle_retarget_editor.py``. It preserves
the shared motion and robot profiles referenced by ``*_ik_retarget.json`` and
writes only the pair configuration.

Example::

    PYTHONPATH=python python/.venv/bin/python \
        python/tools/animation/ik_retarget_editor.py \
        --config assets/retarget/ik/pairs/mixamo_to_g1_ik_retarget.json \
        --input source.fbx --output tuned_mixamo_to_g1_ik_retarget.json
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import replace
from pathlib import Path

# The editor repeatedly recompiles differently shaped preview problems. Let
# JAX grow its allocation on demand so CUDA Graph instantiation still has room.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import kangengine as ke
from kangengine.animation.retarget import build_ik_target_motion
from kangengine import imgui
from retarget_editor_common import (
    RetargetFileDialogs,
    RetargetMotionPreview,
    RetargetSourceLoader,
    RetargetSourcePanel,
    RetargetSourceSettings,
    RetargetViewMode,
    retarget_assets_directory,
    retarget_button,
    retarget_path_input,
    retarget_view_mode_buttons,
    validate_same_skeleton,
)


class IKRetargetConfigEditor(ke.App):
    """Side-by-side source scale and target articulation preview."""

    def __init__(
        self,
        source_path: Path | None = None,
        output_path: Path | None = None,
        config_path: Path | None = None,
        *,
        max_frames: int = 0,
    ) -> None:
        super().__init__()
        self.source_path = source_path
        source_path_text = "" if source_path is None else str(source_path)
        self.output_path_text = (
            "tuned_ik_retarget.json" if output_path is None else str(output_path)
        )
        self.config_path_text = "" if config_path is None else str(config_path)
        self.config = None
        self.target_profile_path_text = ""
        self.target_profile = None
        self.mjcf_path = None
        self.source_motion = None
        self.source_visual = None
        self.view_mode = RetargetViewMode.EDIT_MAPPING
        self.dialogs = RetargetFileDialogs()
        self.world = None
        self.robot = None
        self.sim_visual = None
        self.target_visual = None
        source_settings = RetargetSourceSettings(
            coordinate_system=ke.animation.CoordinateSystem.Y_UP_Z_FORWARD,
            translation_unit_scale=(
                RetargetSourceSettings.default_scale(source_path)
                if source_path is not None
                else 1.0
            ),
        )
        self.source_panel = RetargetSourcePanel(
            path_text=source_path_text,
            settings=source_settings,
            dialog_key="ik_source_motion",
        )
        self.target_coordinates = ke.animation.CoordinateSystem.Z_UP_X_FORWARD
        self.max_frames = int(max_frames)
        self.rendered_frames = 0
        self.frame = 0
        self.default_source_scale = 1.0
        self.source_scale = self.default_source_scale
        self.default_region_scales = {}
        self.region_scales = dict(self.default_region_scales)
        self.default_root_scale = None
        self.root_scale = None
        self.root_height_offset = 0.0
        self.source_root_joint = ""
        self.target_root_link = ""
        self.link_mappings = []
        self.auxiliary_points = []
        self.local_alignment_pairs = []
        self.foot_shape_pairs = []
        self.contact_links = []
        self.detect_contacts = True
        self.selected_mapping = -1
        self.selected_source_joint_name = None
        self.selected_target_link_name = None
        self.selected_auxiliary = -1
        self.new_source_joint = ""
        self.new_target_link = ""
        self.new_pair_first = ""
        self.new_pair_second = ""
        self.new_region_name = ""
        self.target_link_names = ()
        self.target_visual_link_names = ()
        self.solve_result = None
        self.solve_config = None
        self.solve_buffers = None
        self.solve_frame_start = 0
        self.pyroki_model = None
        self._solver_topology_key = None
        self._solved_request_key = None
        self.motion_output_path_text = "ik_retargeted_motion.npz"
        self.confirm_clear_all = False
        self.overlap_preview = False
        self._target_preview_key = None
        self._target_preview_positions = None
        self.sequencer = ke.MotionSequencerPanel()
        self.sequencer.set_overlay(True)
        self.sequencer.set_overlay_width_ratio(1.0)
        self.sequencer.set_playing(False)
        self.status = "Load an IK pair config, then choose a source motion."

    @property
    def source_path_text(self) -> str:
        return self.source_panel.path_text

    @source_path_text.setter
    def source_path_text(self, value: str) -> None:
        self.source_panel.path_text = str(value)

    @property
    def source_settings(self) -> RetargetSourceSettings:
        assert self.source_panel.settings is not None
        return self.source_panel.settings

    def setup(self) -> None:
        self.set_ui_layout_mode(ke.UILayoutMode.EDITOR)
        self.set_camera_view([5.0, -8.0, 3.0], [0.0, 0.0, 1.0])
        self.materials = self.create_standard_materials()
        self.scene.add_ground(scale=20.0, material=self.materials.ground)

        if self.config_path_text:
            self._try_load_config()

    def _create_source_visual(self) -> None:
        if self.source_motion is None:
            return
        self.source_visual = RetargetMotionPreview(
            self,
            self.materials.common,
            "/IKCalibration/Source/Skeleton",
            self.source_motion,
            offset=(-1.25, 0.0, 0.0),
            config=ke.visual.SkeletalVisualConfig(
                bone_color=ke.Vec4(0.25, 0.65, 1.0, 1.0),
                joint_color=ke.Vec4(1.0, 0.82, 0.2, 1.0),
                joint_radius=0.035,
                show_joints=True,
            ),
            joint_pickable=True,
        )
        self._apply_view_mode()

    def _apply_view_mode(self) -> None:
        editing = self.view_mode is RetargetViewMode.EDIT_MAPPING
        if self.source_visual is not None:
            self.source_visual.set_visible(True)
            self.source_visual.set_joint_pickable(editing)
        if self.target_visual is not None:
            self.target_visual.set_visible(
                editing or self.view_mode is RetargetViewMode.PREVIEW_RESULT
            )

    def _set_view_mode(self, mode: RetargetViewMode) -> None:
        if mode is RetargetViewMode.PREVIEW_RESULT and self.solve_result is None:
            return
        self.view_mode = mode
        self._apply_view_mode()

    def _create_target(self) -> None:
        if self.target_profile is None or self.mjcf_path is None:
            return
        self.world = ke.sim.KangSimWorld(
            num_envs=1, sim_dt=1.0 / 60.0, add_ground=False
        )
        data = self.world.load_mjcf(str(self.mjcf_path), order="DFS")
        self.layout = ke.animation.ArticulationCoordinateLayout.from_data(
            data=data, free_root=True
        )
        if self.pyroki_model is None:
            raise RuntimeError("load the PyRoki target model before its visual")
        # IK mappings target URDF/PyRoki link frames. The MJCF visual can omit
        # kinematic-only markers such as left_foot_link/right_foot_link.
        self.target_link_names = self.pyroki_model.link_names
        self.target_visual_link_names = tuple(data.skeleton_tree.node_names())
        robot = self.world.add_articulation(
            data,
            env_id=0,
            obj_id=0,
            name="ik_retarget_target",
            config=ke.physics.ArticulationConfig.free_base(),
        )
        self.robot = robot
        self.target_dof_names = tuple(self.world.state.get_obj_dof_names(0))
        missing = sorted(
            set(self.target_profile.joint_order) - set(self.target_dof_names)
        )
        if missing:
            raise ValueError("target MJCF is missing joints: " + ", ".join(missing))
        canonical_rest = dict(
            zip(
                self.target_profile.joint_order,
                self.target_profile.rest_configuration
                or (0.0,) * len(self.target_profile.joint_order),
            )
        )
        self.target_q = np.asarray(
            [canonical_rest.get(name, 0.0) for name in self.target_dof_names],
            dtype=np.float32,
        )
        self.sim_visual = ke.visual.sim.SimWorldVisualizer(self, self.world)
        self.target_visual = self.sim_visual.add_articulation_scene_graph(
            0,
            0,
            str(self.mjcf_path),
            path="/IKCalibration/Target",
            order="DFS",
            material=self.materials.pbr,
        )
        self._apply_target_pose()
        self._apply_view_mode()

    def _source_loader(self) -> RetargetSourceLoader:
        source_path = Path(self.source_path_text).expanduser()
        profile_name = (
            None if self.config is None else self.config.source_profile.name
        )
        profile = self.source_settings.profile(source_path, name=profile_name)
        if self.config is None:
            return RetargetSourceLoader(
                profile,
                self.target_coordinates,
            )
        required = [self.config.source_root_joint]
        required.extend(mapping.source_joint for mapping in self.config.link_mappings)
        required.extend(point.source_joint for point in self.config.auxiliary_points)
        return RetargetSourceLoader(
            profile,
            self.target_coordinates,
            pair_scale=self.config.translation_scale,
            required_joints=tuple(required),
        )

    def _load_source_motion(self, path: str | Path) -> None:
        source_path = Path(path).expanduser().resolve()
        motion = self._source_loader().load(source_path)
        if self.source_motion is not None:
            validate_same_skeleton(self.source_motion, motion)
        self.source_path = source_path
        self.source_path_text = str(source_path)
        self.source_motion = motion
        self.frame = 0
        self.sequencer.set_motions(
            ["Source"], [motion.num_frames()], [motion.fps()]
        )
        self.sequencer.set_current_time(0.0)
        self.sequencer.set_playing(False)
        self._target_preview_key = None
        if self.source_visual is None:
            self._create_source_visual()
        else:
            self.source_visual.set_motion(motion)
        self._set_view_mode(RetargetViewMode.EDIT_MAPPING)
        self.status = f"Loaded source motion: {source_path.name}"

    def _try_load_source_motion(self) -> None:
        try:
            self._load_source_motion(self.source_path_text)
        except Exception as error:
            self.status = f"Source load failed: {error}"

    def _release_loaded_assets(self) -> None:
        self.clear_debug_points("/IKCalibration/scaled_targets")
        if self.source_visual is not None:
            self.source_visual.remove()
        self.source_visual = None
        self.source_motion = None
        self.remove_prim("/IKCalibration/Source")
        self._release_target()

    def _release_target(self) -> None:
        if self.sim_visual is not None:
            self.sim_visual.release()
        self.sim_visual = None
        self.target_visual = None
        if self.world is not None:
            self.world.release()
        self.world = None
        self.robot = None
        self.remove_prim("/IKCalibration/Target")

    def _load_target_profile(self, path: str | Path) -> None:
        profile_path = Path(path).expanduser().resolve()
        profile = ke.animation.IKTargetProfile.load(profile_path)
        visual = profile.visual_skeleton
        if visual is None:
            raise ValueError("IK target profile requires visual_skeleton")
        for required in (profile.skeleton, visual):
            if not required.is_file():
                raise FileNotFoundError(required)

        if self.pyroki_model is not None:
            import jax

            jax.clear_caches()
            gc.collect()
        self._release_target()
        from kangengine.adapters.pyroki import load_urdf_model

        self.target_profile_path_text = str(profile_path)
        self.target_profile = profile
        self.pyroki_model = load_urdf_model(profile)
        self.mjcf_path = visual
        self.target_coordinates = ke.animation.CoordinateSystem(
            profile.coordinate_system
        )
        self._solver_topology_key = None
        self._solved_request_key = None
        self.solve_result = None
        self.solve_config = None
        self.solve_buffers = None
        self._create_target()
        self.status = f"Loaded target profile: {profile.name}"

    def _try_load_target_profile(self) -> None:
        try:
            self._load_target_profile(self.target_profile_path_text)
        except Exception as error:
            self.status = f"Target profile load failed: {error}"

    def _load_config(self, path: str | Path) -> None:
        config_path = Path(path).expanduser().resolve()
        config = ke.animation.IKRetargetConfig.load(config_path)
        mjcf = config.target_profile.visual_skeleton
        if mjcf is None:
            raise ValueError("IK target profile requires visual_skeleton")
        for required in (config.urdf_path, mjcf):
            if not required.is_file():
                raise FileNotFoundError(required)

        if self.pyroki_model is not None:
            import jax

            jax.clear_caches()
            gc.collect()

        self._release_loaded_assets()
        self.config_path_text = str(config_path)
        self.config = config
        self.target_profile = config.target_profile
        self.target_profile_path_text = str(config.target_profile_path or "")
        from kangengine.adapters.pyroki import load_urdf_model

        self.pyroki_model = load_urdf_model(config)
        self._solver_topology_key = None
        self._solved_request_key = None
        self.mjcf_path = mjcf
        self.source_settings.apply_profile(config.source_profile)
        self.target_coordinates = ke.animation.CoordinateSystem(
            config.target_profile.coordinate_system
        )
        self.default_source_scale = float(config.translation_scale)
        self.source_scale = self.default_source_scale
        self.default_region_scales = {
            name: tuple(values) for name, values in config.region_scales.items()
        }
        self.region_scales = dict(self.default_region_scales)
        self.default_root_scale = config.root_position_scale
        self.root_scale = config.root_position_scale
        self.root_height_offset = float(config.root_height_offset)
        self.source_root_joint = config.source_root_joint
        self.target_root_link = config.target_root_link
        self.link_mappings = list(config.link_mappings)
        self.auxiliary_points = list(config.auxiliary_points)
        self.local_alignment_pairs = list(config.local_alignment_pairs)
        self.foot_shape_pairs = list(config.foot_shape_pairs)
        self.contact_links = list(config.contact_links)
        self.selected_mapping = -1
        self.selected_source_joint_name = None
        self.selected_target_link_name = None
        self.selected_auxiliary = -1
        self.solve_result = None
        self.solve_config = None
        self.solve_buffers = None
        self._target_preview_key = None
        self._target_preview_positions = None
        self._create_target()
        if self.source_path_text:
            self._load_source_motion(self.source_path_text)
        else:
            self.status = f"Loaded config: {config_path.name}; choose a source motion."

    def _try_load_config(self) -> None:
        try:
            self._load_config(self.config_path_text)
        except Exception as error:
            self.status = f"Config load failed: {error}"

    def _try_save(self) -> None:
        try:
            self._save()
        except Exception as error:
            self.status = f"Config save failed: {error}"

    def _apply_target_pose(self) -> None:
        if self.robot is None or self.world is None or self.sim_visual is None:
            return
        self.robot.set_root_state(
            None,
            [self._target_x(), 0.0, 0.8],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            immediate=True,
        )
        self.robot.set_dof_state(
            None,
            self.target_q,
            np.zeros_like(self.target_q),
            immediate=True,
        )
        self.world.step(substeps=0, apply_commands=False)
        self.sim_visual.sync()

    def _save(self) -> None:
        if self.config is None:
            self.status = "Load an IK retarget config before saving."
            return
        if not self.output_path_text.strip():
            self.status = "Choose an output config path before saving."
            return
        config = self._working_config()
        output_path = Path(self.output_path_text).expanduser().resolve()
        config.save(output_path)
        self.output_path_text = str(output_path)
        self.status = f"Saved {output_path}"
        print(self.status)

    def _reset_source(self) -> None:
        self.frame = 0
        self.source_scale = self.default_source_scale
        self.status = "Source frame and scale reset to config defaults."

    def _multiply_source_scale(self, factor: float) -> None:
        self.source_scale *= float(factor)
        self._target_preview_key = None
        self._solved_request_key = None
        self.status = (
            f"Scaled source by {factor:g}; current multiplier is "
            f"{self.source_scale / self.default_source_scale:g}."
        )

    def _reset_all(self) -> None:
        self._reset_source()
        self.region_scales = dict(self.default_region_scales)
        self.root_scale = self.default_root_scale
        self.root_height_offset = float(self.config.root_height_offset)
        self.source_root_joint = self.config.source_root_joint
        self.target_root_link = self.config.target_root_link
        self.link_mappings = list(self.config.link_mappings)
        self.auxiliary_points = list(self.config.auxiliary_points)
        self.local_alignment_pairs = list(self.config.local_alignment_pairs)
        self.foot_shape_pairs = list(self.config.foot_shape_pairs)
        self.contact_links = list(self.config.contact_links)
        self.status = "All scale values reset to config defaults."

    def _working_config(self):
        if self.config is None:
            raise RuntimeError("load an IK retarget config first")
        return replace(
            self.config,
            translation_scale=self.source_scale,
            region_scales=self.region_scales,
            root_position_scale=self.root_scale,
            root_height_offset=self.root_height_offset,
            source_root_joint=self.source_root_joint,
            target_root_link=self.target_root_link,
            link_mappings=tuple(self.link_mappings),
            auxiliary_points=tuple(self.auxiliary_points),
            local_alignment_pairs=tuple(self.local_alignment_pairs),
            foot_shape_pairs=tuple(self.foot_shape_pairs),
            contact_links=tuple(self.contact_links),
        )

    def _scale3_controls(
        self, label: str, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return self._vec3_controls(label, value, minimum=0.25, maximum=2.0)

    def _vec3_controls(
        self,
        label: str,
        value: tuple[float, float, float],
        *,
        minimum: float,
        maximum: float,
    ) -> tuple[float, float, float]:
        result = list(value)
        for axis, axis_name in enumerate("xyz"):
            changed, component = imgui.slider_float(
                f"{label} {axis_name}",
                float(result[axis]),
                minimum,
                maximum,
            )
            if changed:
                result[axis] = float(component)
        return tuple(result)

    def _quat_controls(self, label: str, value) -> tuple[float, float, float, float]:
        result = list(value)
        for index, component_name in enumerate("wxyz"):
            changed, component = imgui.slider_float(
                f"{label} {component_name}", float(result[index]), -1.0, 1.0
            )
            if changed:
                result[index] = float(component)
        norm = float(np.linalg.norm(result))
        if norm <= 1.0e-8:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(float(component) / norm for component in result)

    def _source_x(self) -> float:
        return 0.0 if self.overlap_preview else -1.25

    def _target_x(self) -> float:
        return 0.0 if self.overlap_preview else 1.25

    def _update_target_preview(self, frame: int | None = None) -> None:
        key = (
            tuple(sorted(self.region_scales.items())),
            self.root_scale,
            self.source_scale,
            tuple(self.link_mappings),
            tuple(self.auxiliary_points),
        )
        if key != self._target_preview_key:
            preview_config = self._working_config()
            scale_ratio = self.source_scale / self.default_source_scale
            preview_motion = ke.animation.scale_skeleton_motion(
                self.source_motion, scale_ratio
            )
            targets = build_ik_target_motion(
                preview_motion, preview_config
            )
            self._target_preview_positions = np.asarray(
                targets.positions, dtype=np.float32
            )
            self._target_preview_key = key
        preview_frame = self.frame if frame is None else int(frame)
        points = self._target_preview_positions[preview_frame]
        points = points + np.asarray([self._source_x(), 0.0, 0.0], dtype=np.float32)
        colors = np.tile(
            np.asarray([[0.1, 0.85, 1.0, 1.0]], dtype=np.float32),
            (len(points), 1),
        )
        self.log_debug_points("/IKCalibration/scaled_targets", points, colors, 11.0)

    def _source_slice(self, start: int, count: int | None):
        end = (
            self.source_motion.num_frames()
            if count is None
            else min(start + count, self.source_motion.num_frames())
        )
        scale_ratio = self.source_scale / self.default_source_scale
        source = ke.animation.scale_skeleton_motion(self.source_motion, scale_ratio)
        return ke.animation.SkeletonMotion.from_arrays(
            skeleton_tree=source.skeleton_tree,
            root_translations=source.root_translations()[start:end],
            local_rotations_wxyz=source.local_rotations_wxyz()[start:end],
            fps=source.fps(),
            motion_name=f"{source.motion_name()}_{start:06d}_{end:06d}",
        )

    def _solve(self, *, start: int, count: int | None) -> None:
        if self.source_motion is None or self.config is None:
            self.status = "Load a config and source motion before solving."
            return
        try:
            from kangengine.adapters.physx import PhysXMotionAdapter
            from kangengine.adapters.pyroki import (
                PyrokiTrajectoryIKConfig,
                retarget_motion_offline,
                validate_profile,
            )

            config = self._working_config()
            source = self._source_slice(start, count)
            if self.pyroki_model is None:
                raise RuntimeError("load an IK retarget config before solving")
            validate_profile(config, self.pyroki_model)
            topology_key = (
                source.num_frames(),
                tuple(mapping.target_link for mapping in config.link_mappings),
                tuple(point.target_link for point in config.auxiliary_points),
                tuple(config.local_alignment_pairs),
                tuple(config.foot_shape_pairs),
                tuple(config.contact_links) if self.detect_contacts else (),
                any(mapping.rotation_weight > 0.0 for mapping in config.link_mappings),
            )
            request_key = (
                config,
                start,
                source.num_frames(),
                self.detect_contacts,
            )
            if request_key == self._solved_request_key and self.solve_result is not None:
                self.status = (
                    f"Reused solved {self.solve_result.motion.num_frames()} frame(s)."
                )
                return
            if (
                self._solver_topology_key is not None
                and topology_key != self._solver_topology_key
            ):
                import jax

                self.solve_result = None
                self.solve_config = None
                self.solve_buffers = None
                self._solved_request_key = None
                jax.clear_caches()
                gc.collect()
            self._solver_topology_key = topology_key
            self.status = f"Solving {source.num_frames()} frame(s)..."
            result = retarget_motion_offline(
                source,
                self.pyroki_model,
                config,
                self.layout,
                detect_contacts=self.detect_contacts,
                config=PyrokiTrajectoryIKConfig(
                    whole_trajectory_max_frames=1000,
                    window_size=512,
                    window_overlap=32,
                ),
            )
            adapter = PhysXMotionAdapter(self.layout)
            adapter.validate_articulation(self.robot.articulation)
            self.solve_result = result
            self.solve_config = config
            self.solve_buffers = adapter.pack(result.motion)
            self._solved_request_key = request_key
            self.solve_frame_start = start
            self._set_view_mode(RetargetViewMode.PREVIEW_RESULT)
            self.sequencer.set_motions(
                ["Source", "IK Result"],
                [self.source_motion.num_frames(), result.motion.num_frames()],
                [self.source_motion.fps(), result.motion.fps()],
            )
            solve = result.solve
            self.status = (
                f"Solved {result.motion.num_frames()} frame(s): "
                f"RMSE {solve.initial_position_rmse:.4f} → "
                f"{solve.position_rmse:.4f} m, iterations={solve.iterations}, "
                f"converged={solve.converged}"
            )
        except Exception as error:
            self.status = f"IK solve failed: {error}"

    def _save_solved_motion(self) -> None:
        if self.solve_result is None:
            self.status = "Solve an IK motion before saving it."
            return
        try:
            path = Path(self.motion_output_path_text).expanduser().resolve()
            saved = ke.animation.save_articulation_motion_npz(
                self.solve_result.motion, path
            )
            self.motion_output_path_text = str(saved)
            self.status = f"Saved IK motion: {saved}"
        except Exception as error:
            self.status = f"IK motion save failed: {error}"

    def _apply_solved_pose(self) -> bool:
        if self.solve_buffers is None:
            return False
        try:
            if self.solve_config != self._working_config():
                return False
        except (TypeError, ValueError):
            return False
        local_frame = self.frame - self.solve_frame_start
        frame_count = len(self.solve_buffers.joint_positions)
        if not 0 <= local_frame < frame_count:
            return False
        buffers = self.solve_buffers
        root_position = np.asarray(buffers.root_positions[local_frame]).copy()
        root_position[0] += self._target_x()
        self.robot.set_root_state(
            None,
            root_position,
            buffers.root_rotations_xyzw[local_frame],
            buffers.root_linear_velocities[local_frame],
            buffers.root_angular_velocities[local_frame],
            immediate=True,
        )
        self.robot.set_dof_state(
            None,
            buffers.joint_positions[local_frame],
            buffers.joint_velocities[local_frame],
            immediate=True,
        )
        self.world.step(substeps=0, apply_commands=False)
        self.sim_visual.sync()
        return True

    def pre_render(self) -> None:
        if self.source_motion is None:
            self._apply_target_pose()
            return
        scale_ratio = self.source_scale / self.default_source_scale
        if self.sequencer.is_playing():
            self.sequencer.set_current_time(
                self.sequencer.current_time()
                + self.get_delta_time() * self.sequencer.time_scale()
            )
        frame = int(self.sequencer.current_time() * self.source_motion.fps())
        if self.sequencer.loop():
            frame %= self.source_motion.num_frames()
        self.frame = max(0, min(frame, self.source_motion.num_frames() - 1))
        if self.source_visual is not None:
            self.source_visual.set_scale(scale_ratio)
            self.source_visual.offset = (self._source_x(), 0.0, 0.0)
            self.source_visual.apply_frame(self.frame)
        try:
            if (
                self.config is None
                or self.view_mode is not RetargetViewMode.EDIT_MAPPING
            ):
                self.clear_debug_points("/IKCalibration/scaled_targets")
            else:
                self._update_target_preview(self.frame)
        except Exception as error:
            self.clear_debug_points("/IKCalibration/scaled_targets")
            self.status = f"Target preview unavailable: {error}"
        if (
            self.view_mode is not RetargetViewMode.PREVIEW_RESULT
            or not self._apply_solved_pose()
        ):
            self._apply_target_pose()

    def on_ray_picked(self, result) -> None:
        if (
            self.view_mode is RetargetViewMode.EDIT_MAPPING
            and self.source_visual is not None
            and self.source_motion is not None
        ):
            joint = self.source_visual.joint_from_pick(result)
            if joint is not None:
                name = self.source_motion.skeleton_tree.node_name(joint)
                self.selected_source_joint_name = name
                self.new_source_joint = name
                self.status = f"Selected source joint: {name}"
                return
        if (
            self.view_mode is RetargetViewMode.EDIT_MAPPING
            and self.sim_visual is not None
        ):
            picked = self.sim_visual.pick_body(result)
            if picked is not None and 0 <= picked.body_id < len(
                self.target_visual_link_names
            ):
                name = self.target_visual_link_names[picked.body_id]
                if name in self.target_link_names:
                    self.selected_target_link_name = name
                    self.new_target_link = name
                    self.status = f"Selected target link: {name}"

    def _clear_all(self) -> None:
        self._release_loaded_assets()
        self.config = None
        self.target_profile = None
        self.target_profile_path_text = ""
        self.mjcf_path = None
        self.config_path_text = ""
        self.source_path = None
        self.source_path_text = ""
        self.solve_result = None
        self.solve_config = None
        self.solve_buffers = None
        self.pyroki_model = None
        self._solver_topology_key = None
        self._solved_request_key = None
        self.link_mappings = []
        self.selected_source_joint_name = None
        self.selected_target_link_name = None
        self.auxiliary_points = []
        self.local_alignment_pairs = []
        self.foot_shape_pairs = []
        self.contact_links = []
        self.sequencer.clear_motions()
        self.view_mode = RetargetViewMode.EDIT_MAPPING
        self.confirm_clear_all = False
        self.status = "Editor cleared. Load an IK pair config to begin."

    def _render_alignment_panel(self) -> None:
        imgui.separator()
        imgui.text("1. Source / Target Alignment")
        _, self.overlap_preview = imgui.checkbox(
            "overlay source and target", self.overlap_preview
        )
        changed, frame = imgui.slider_float(
            "source frame",
            float(self.frame),
            0.0,
            float(max(0, self.source_motion.num_frames() - 1)),
        )
        if changed:
            self.frame = int(round(frame))
            self.sequencer.set_current_time(
                self.frame / max(float(self.source_motion.fps()), 1.0e-6)
            )
        scale_multiplier = self.source_scale / self.default_source_scale
        changed, scale_multiplier = imgui.slider_float(
            "source scale multiplier", scale_multiplier, 0.01, 100.0
        )
        if changed:
            self.source_scale = self.default_source_scale * float(scale_multiplier)
        changed, root_height = imgui.slider_float(
            "root height offset", self.root_height_offset, -0.5, 0.5
        )
        if changed:
            self.root_height_offset = float(root_height)
        imgui.text(
            f"frame {self.frame}/{self.source_motion.num_frames() - 1} | "
            f"{self.source_motion.fps():g} fps | pair scale {self.source_scale:.6g}"
        )
        if retarget_button("Reset source alignment", "warning"):
            self._reset_source()

    def _render_mapping_panel(self) -> None:
        imgui.separator()
        imgui.text("2. Link Mapping")
        _, self.source_root_joint = imgui.input_text(
            "source root joint", self.source_root_joint
        )
        _, self.target_root_link = imgui.input_text(
            "target root link", self.target_root_link
        )
        imgui.text("Source joints")
        imgui.same_line()
        imgui.text("                              Target links")
        imgui.begin_child(
            "ik_source_joint_list",
            440.0,
            220.0,
            True,
            imgui.WindowFlags_HorizontalScrollbar,
        )
        mapped_from_source = {
            mapping.source_joint: mapping.target_link for mapping in self.link_mappings
        }
        for source_name in self.source_motion.node_names():
            mapped = mapped_from_source.get(source_name)
            label = f"{source_name} -> {mapped}" if mapped is not None else source_name
            if imgui.selectable(
                f"{label}##ik_source_{source_name}",
                self.selected_source_joint_name == source_name,
            ):
                self.selected_source_joint_name = source_name
                self.new_source_joint = source_name
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child(
            "ik_target_link_list",
            440.0,
            220.0,
            True,
            imgui.WindowFlags_HorizontalScrollbar,
        )
        mapped_to_target = {
            mapping.target_link: mapping.source_joint for mapping in self.link_mappings
        }
        for target_name in self.target_link_names:
            mapped = mapped_to_target.get(target_name)
            label = f"{target_name} <- {mapped}" if mapped is not None else target_name
            if imgui.selectable(
                f"{label}##ik_target_{target_name}",
                self.selected_target_link_name == target_name,
            ):
                self.selected_target_link_name = target_name
                self.new_target_link = target_name
        imgui.end_child()
        selected_source = self.selected_source_joint_name or "<none>"
        selected_target = self.selected_target_link_name or "<none>"
        imgui.text(f"Selected: {selected_source} -> {selected_target}")
        if retarget_button("Map Pair", "primary"):
            if (
                self.selected_source_joint_name is None
                or self.selected_target_link_name is None
            ):
                self.status = "Select one source joint and one target link first."
            else:
                source_name = self.selected_source_joint_name
                target_name = self.selected_target_link_name
                retained = [
                    mapping
                    for mapping in self.link_mappings
                    if mapping.source_joint != source_name
                    and mapping.target_link != target_name
                ]
                previous = next(
                    (
                        mapping
                        for mapping in self.link_mappings
                        if mapping.source_joint == source_name
                    ),
                    None,
                )
                retained.append(
                    ke.animation.IKEffectorMapping(source_name, target_name)
                    if previous is None
                    else replace(previous, target_link=target_name)
                )
                self.link_mappings = retained
                self.selected_mapping = len(retained) - 1
                self._target_preview_key = None
                self.status = f"Mapped {source_name} -> {target_name}"
        imgui.begin_child("ik_link_mappings", 0.0, 190.0, True)
        for index, mapping in enumerate(self.link_mappings):
            label = f"{mapping.source_joint} -> {mapping.target_link}"
            if imgui.selectable(
                f"{label}##mapping_{index}", self.selected_mapping == index
            ):
                self.selected_mapping = index
        imgui.end_child()
        if 0 <= self.selected_mapping < len(self.link_mappings):
            mapping = self.link_mappings[self.selected_mapping]
            changed_source, source_joint = imgui.input_text(
                "source joint##mapping", mapping.source_joint
            )
            changed_target, target_link = imgui.input_text(
                "target link##mapping", mapping.target_link
            )
            _, region = imgui.input_text("region##mapping", mapping.region or "")
            _, position_weight = imgui.slider_float(
                "position weight", mapping.position_weight, 0.0, 10.0
            )
            _, rotation_weight = imgui.slider_float(
                "rotation weight", mapping.rotation_weight, 0.0, 10.0
            )
            position_offset = self._vec3_controls(
                "position offset",
                mapping.position_offset,
                minimum=-1.0,
                maximum=1.0,
            )
            use_position_scale = mapping.position_scale is not None
            changed_scale, use_position_scale = imgui.checkbox(
                "custom position scale", use_position_scale
            )
            position_scale = (
                (1.0, 1.0, 1.0)
                if mapping.position_scale is None
                else mapping.position_scale
            )
            if use_position_scale:
                position_scale = self._scale3_controls("position scale", position_scale)
            else:
                position_scale = None
            rotation_offset = self._quat_controls(
                "rotation offset", mapping.rotation_offset_wxyz
            )
            if (
                changed_source
                or changed_target
                or region != (mapping.region or "")
                or (
                    position_weight != mapping.position_weight
                    or rotation_weight != mapping.rotation_weight
                    or position_offset != mapping.position_offset
                    or changed_scale
                    or position_scale != mapping.position_scale
                    or rotation_offset != mapping.rotation_offset_wxyz
                )
            ):
                try:
                    self.link_mappings[self.selected_mapping] = replace(
                        mapping,
                        source_joint=source_joint,
                        target_link=target_link,
                        region=region or None,
                        position_weight=float(position_weight),
                        rotation_weight=float(rotation_weight),
                        position_scale=position_scale,
                        position_offset=position_offset,
                        rotation_offset_wxyz=rotation_offset,
                    )
                    self._target_preview_key = None
                except ValueError as error:
                    self.status = f"Mapping edit rejected: {error}"
            if retarget_button("Remove selected mapping", "danger"):
                self.link_mappings.pop(self.selected_mapping)
                self.selected_mapping = -1
                self._target_preview_key = None
        _, self.new_source_joint = imgui.input_text(
            "new source joint", self.new_source_joint
        )
        _, self.new_target_link = imgui.input_text(
            "new target link", self.new_target_link
        )
        if retarget_button("Add mapping", "success"):
            try:
                self.link_mappings.append(
                    ke.animation.IKEffectorMapping(
                        self.new_source_joint, self.new_target_link
                    )
                )
                self.selected_mapping = len(self.link_mappings) - 1
                self._target_preview_key = None
            except Exception as error:
                self.status = f"Mapping add failed: {error}"

    def _render_scale_panel(self) -> None:
        imgui.separator()
        imgui.text("3. Target Scale")
        imgui.text("Cyan points show the current scaled IK targets.")
        remove_region = None
        for name, values in tuple(self.region_scales.items()):
            self.region_scales[name] = self._scale3_controls(name, values)
            if retarget_button(f"Remove region##{name}", "danger"):
                remove_region = name
        if remove_region is not None:
            del self.region_scales[remove_region]
        _, self.new_region_name = imgui.input_text("new region", self.new_region_name)
        imgui.same_line()
        if retarget_button("Add region", "success") and self.new_region_name:
            self.region_scales.setdefault(self.new_region_name, (1.0, 1.0, 1.0))
            self.new_region_name = ""
        use_root_scale = self.root_scale is not None
        changed, use_root_scale = imgui.checkbox(
            "custom root position scale", use_root_scale
        )
        if changed:
            self.root_scale = (1.0, 1.0, 1.0) if use_root_scale else None
        if use_root_scale and self.root_scale is not None:
            self.root_scale = self._scale3_controls("root", self.root_scale)

    def _render_auxiliary_panel(self) -> None:
        imgui.separator()
        imgui.text("4. Auxiliary Points")
        imgui.begin_child("ik_auxiliary_points", 0.0, 130.0, True)
        for index, point in enumerate(self.auxiliary_points):
            if imgui.selectable(
                f"{point.name}: {point.source_joint} -> {point.target_link}##aux_{index}",
                self.selected_auxiliary == index,
            ):
                self.selected_auxiliary = index
        imgui.end_child()
        if 0 <= self.selected_auxiliary < len(self.auxiliary_points):
            point = self.auxiliary_points[self.selected_auxiliary]
            _, name = imgui.input_text("name##aux", point.name)
            _, source_joint = imgui.input_text("source joint##aux", point.source_joint)
            _, target_link = imgui.input_text("target link##aux", point.target_link)
            _, region = imgui.input_text("region##aux", point.region or "")
            source_offset = self._vec3_controls(
                "source offset", point.source_offset, minimum=-1.0, maximum=1.0
            )
            target_offset = self._vec3_controls(
                "target offset", point.target_offset, minimum=-1.0, maximum=1.0
            )
            _, weight = imgui.slider_float("weight##aux", point.weight, 0.0, 10.0)
            try:
                self.auxiliary_points[self.selected_auxiliary] = replace(
                    point,
                    name=name,
                    source_joint=source_joint,
                    target_link=target_link,
                    region=region or None,
                    source_offset=source_offset,
                    target_offset=target_offset,
                    weight=float(weight),
                )
            except ValueError as error:
                self.status = f"Auxiliary edit rejected: {error}"
            if retarget_button("Remove selected auxiliary", "danger"):
                self.auxiliary_points.pop(self.selected_auxiliary)
                self.selected_auxiliary = -1
        if retarget_button("Add auxiliary", "success"):
            try:
                self.auxiliary_points.append(
                    ke.animation.AuxiliaryRetargetPoint(
                        name=f"aux_{len(self.auxiliary_points)}",
                        source_joint=self.new_source_joint,
                        target_link=self.new_target_link,
                        target_offset=(0.0, 0.0, 0.0),
                    )
                )
                self.selected_auxiliary = len(self.auxiliary_points) - 1
            except Exception as error:
                self.status = f"Auxiliary add failed: {error}"

    def _render_pair_list(self, label: str, values: list[tuple[str, str]]) -> None:
        imgui.text(label)
        remove = None
        for index, pair in enumerate(values):
            imgui.text(f"{pair[0]} -> {pair[1]}")
            imgui.same_line()
            if retarget_button(f"Remove##{label}_{index}", "danger"):
                remove = index
        if remove is not None:
            values.pop(remove)

    def _render_constraints_panel(self) -> None:
        imgui.separator()
        imgui.text("5. Alignment / Foot Shape")
        self._render_pair_list("Local alignment", self.local_alignment_pairs)
        self._render_pair_list("Foot shape", self.foot_shape_pairs)
        _, self.new_pair_first = imgui.input_text(
            "pair first link", self.new_pair_first
        )
        _, self.new_pair_second = imgui.input_text(
            "pair second link", self.new_pair_second
        )
        if retarget_button("Add local alignment", "success"):
            self.local_alignment_pairs.append(
                (self.new_pair_first, self.new_pair_second)
            )
        imgui.same_line()
        if retarget_button("Add foot shape", "success"):
            self.foot_shape_pairs.append((self.new_pair_first, self.new_pair_second))

    def _render_contact_panel(self) -> None:
        imgui.separator()
        imgui.text("6. Contacts")
        _, self.detect_contacts = imgui.checkbox(
            "detect contacts while solving", self.detect_contacts
        )
        imgui.begin_child("ik_contact_links", 0.0, 150.0, True)
        for link in self.target_link_names:
            enabled = link in self.contact_links
            changed, enabled = imgui.checkbox(f"{link}##contact", enabled)
            if changed and enabled:
                self.contact_links.append(link)
            elif changed:
                self.contact_links.remove(link)
        imgui.end_child()

    def _render_solve_panel(self) -> None:
        imgui.separator()
        imgui.text("7. Solve / Preview")
        if retarget_button("Preview Current Frame", "primary"):
            self._solve(start=self.frame, count=1)
        imgui.same_line()
        if retarget_button("Solve 100 Frames", "primary"):
            self._solve(start=self.frame, count=100)
        imgui.same_line()
        if retarget_button("Solve Full Motion", "success"):
            self._solve(start=0, count=None)
        if self.solve_result is not None:
            solve = self.solve_result.solve
            imgui.text(
                f"RMSE {solve.initial_position_rmse:.4f} -> "
                f"{solve.position_rmse:.4f} m | iterations {solve.iterations} | "
                f"converged {solve.converged}"
            )
            _, self.motion_output_path_text = imgui.input_text(
                "Motion output", self.motion_output_path_text
            )
            imgui.same_line()
            if retarget_button("Browse##motion_output"):
                self.dialogs.open_save(
                    "motion_output",
                    "Save IK Articulation Motion",
                    "NPZ motion{.npz}",
                    self.motion_output_path_text,
                )
            selected = self.dialogs.selected("motion_output")
            if selected is not None:
                self.motion_output_path_text = selected
                self._save_solved_motion()
            if retarget_button("Save Solved Motion", "success"):
                self._save_solved_motion()

    def render(self) -> None:
        imgui.begin("IK Retarget Config")
        if retarget_button("Clear All", "danger"):
            self.confirm_clear_all = True
        if self.confirm_clear_all:
            imgui.same_line()
            imgui.text("Remove config, source, target, preview, and solved motion?")
            if retarget_button("Confirm Clear All", "danger"):
                self._clear_all()
            imgui.same_line()
            if retarget_button("Cancel Clear"):
                self.confirm_clear_all = False
        imgui.separator()
        imgui.text("1. Source Motion")
        source_action = self.source_panel.render(
            self.dialogs,
            label="Source motion",
            loaded=self.source_motion is not None,
            current_scale=(
                None
                if self.source_motion is None
                else self.source_scale / self.default_source_scale
            ),
            filters="Motion files{.bvh,.fbx}",
            use_file_scale_on_select=self.config is None,
        )
        if source_action.load_requested:
            self._try_load_source_motion()
        if source_action.selected_path is not None:
            self.status = "Source selected. Review import settings, then load it."
        if source_action.scale_factor is not None:
            self._multiply_source_scale(source_action.scale_factor)
        imgui.separator()
        imgui.text("2. Target Profile")
        _, self.target_profile_path_text = retarget_path_input(
            "Target profile", self.target_profile_path_text
        )
        imgui.same_line()
        if retarget_button("Browse##target_profile"):
            self.dialogs.open_file(
                "target_profile",
                "Select IK Target Profile",
                "JSON files{.json}",
                str(retarget_assets_directory() / "ik" / "targets"),
            )
        imgui.same_line()
        if retarget_button("Load##target_profile", "primary"):
            self._try_load_target_profile()
        selected_path = self.dialogs.selected("target_profile")
        if selected_path is not None:
            self.target_profile_path_text = selected_path
            self.status = "Target profile selected. Load it when ready."
        if self.target_profile is not None:
            imgui.text(f"Robot: {self.target_profile.name}")
            imgui.text_disabled(f"Kinematics (URDF): {self.target_profile.skeleton}")
            imgui.text_disabled(
                f"Visual/articulation: {self.target_profile.visual_skeleton}"
            )
        imgui.separator()
        imgui.text("3. Pair Config")
        _, self.config_path_text = retarget_path_input(
            "Pair config", self.config_path_text
        )
        imgui.same_line()
        if retarget_button("Browse##pair_config"):
            self.dialogs.open_file(
                "pair_config",
                "Select IK Retarget Config",
                "JSON files{.json}",
                str(retarget_assets_directory()),
            )
        imgui.same_line()
        if retarget_button("Load##pair_config", "primary"):
            self._try_load_config()
        selected_path = self.dialogs.selected("pair_config")
        if selected_path is not None:
            self.config_path_text = selected_path
            self.status = "Pair config selected. Load it when ready."

        imgui.separator()
        selected_mode = retarget_view_mode_buttons(
            self.view_mode,
            source_available=self.source_motion is not None,
            result_available=self.solve_result is not None,
        )
        if selected_mode is not self.view_mode:
            self._set_view_mode(selected_mode)
        imgui.text_disabled("Edit Mapping enables picking; previews are read-only.")

        if self.config is None:
            imgui.text(self.status)
            imgui.end()
            if self.source_motion is not None:
                self.sequencer.build_panel()
            self._finish_frame()
            return

        imgui.text(
            f"Motion profile: {self.config.source_profile.name}  "
            f"({self.config.source_profile.coordinate_system})"
        )
        if self.source_motion is None:
            imgui.text(self.status)
            imgui.end()
            self._finish_frame()
            return

        self._render_alignment_panel()
        self._render_mapping_panel()
        self._render_scale_panel()
        self._render_auxiliary_panel()
        self._render_constraints_panel()
        self._render_contact_panel()
        self._render_solve_panel()
        imgui.separator()
        imgui.text("8. Save Pair Config")
        if retarget_button("Reset all", "warning"):
            self._reset_all()
        imgui.separator()
        _, self.output_path_text = imgui.input_text(
            "Output config", self.output_path_text
        )
        imgui.same_line()
        if retarget_button("Browse##output_config"):
            self.dialogs.open_save(
                "output_config",
                "Save IK Retarget Config",
                "JSON files{.json}",
                self.output_path_text,
            )
        selected_path = self.dialogs.selected("output_config")
        if selected_path is not None:
            self.output_path_text = selected_path
            self._try_save()
        if retarget_button("Save IK retarget config", "success"):
            self._try_save()
        imgui.separator()
        imgui.text(self.status)
        imgui.end()

        self.sequencer.build_panel()

        self._finish_frame()

    def _finish_frame(self) -> None:
        self.rendered_frames += 1
        if self.max_frames > 0 and self.rendered_frames >= self.max_frames:
            self.request_close()

    def cleanup(self) -> None:
        self._release_loaded_assets()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Close after this many rendered frames; useful for smoke checks.",
    )
    args = parser.parse_args()

    app = IKRetargetConfigEditor(
        source_path=(None if args.input is None else args.input.expanduser().resolve()),
        output_path=(
            None if args.output is None else args.output.expanduser().resolve()
        ),
        config_path=(
            None if args.config is None else args.config.expanduser().resolve()
        ),
        max_frames=args.frames,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Z, headless=args.headless)
    app.start()


if __name__ == "__main__":
    main()
