"""Inspect physics collision geometry through the shared debug renderer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._core import _ke
from ..render import TransformSource
from ..utils.math import quat_wxyz_to_matrix_numpy, quat_xyzw_to_wxyz
from .physics_inspection_source import PhysicsInspectionSource


def _rotation(quat_xyzw):
    return quat_wxyz_to_matrix_numpy(quat_xyzw_to_wxyz(quat_xyzw))


def _shape_lines(shape, plane_size, *, world_space=True):
    kind = shape["type"]
    if kind == "box":
        mesh = _ke.geometry.create_box_data(*(2 * np.asarray(shape["half_extents"])))
    elif kind == "sphere":
        mesh = _ke.geometry.create_sphere_data(shape["radius"], 12, 8)
    elif kind == "capsule":
        mesh = _ke.geometry.create_capsule_data(
            shape["radius"],
            2 * shape["half_height"],
            _ke.UpAxis.X,
            12,
        )
    elif kind == "plane":
        # Inspection planes use a local +X normal and a finite preview square.
        mesh = _ke.geometry.create_plane_data(plane_size, _ke.UpAxis.X)
    elif kind in ("convex_mesh", "triangle_mesh", "heightfield"):
        mesh = None
    else:
        return None

    if mesh is None:
        vertices = np.asarray(shape["vertices"], dtype=np.float32)
        triangles = np.asarray(shape["triangles"], dtype=np.int32).reshape(-1, 3)
        if kind != "heightfield":
            rotation = _rotation(shape["mesh_scale_rotation_xyzw"])
            # Scale in the rotated mesh axes before applying the shape pose.
            vertices = ((vertices @ rotation.T) * shape["mesh_scale"]) @ rotation
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)

    # Merge seam vertices and shared triangle edges before uploading lines.
    vertices, vertex_ids = np.unique(vertices, axis=0, return_inverse=True)
    triangles = vertex_ids[triangles]
    edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    poses = [shape["local_pose"]]
    if world_space:
        poses.append(shape["actor_pose"])
    for pose in poses:
        vertices = vertices @ _rotation(pose[3:]).T + pose[:3]
    return vertices[edges[:, 0]], vertices[edges[:, 1]]


class PhysicsInspector:
    """App-owned collider display and selection for Native or OVRTX.

    The App retains the inspector until release or App.close().
    Display is off by default.
    The App synchronizes enabled displays after pre_render(), when simulation
    fetching must be complete. Refresh after editing geometry or adding actors.
    Static poses update on Refresh; dynamic poses reuse cached local geometry.

    Args:
        app: Initialized App.
        world: Optional PhysicsWorld shorthand for a PhysXInspectionSource.
        source: PhysicsInspectionSource; omit world when supplying this.
        visualizer: Optional SimWorldVisualizer for the world shorthand.
        plane_size: Preview side length for infinite planes, in scene units.
    """

    _path = "/__physics_inspector/static"
    _dynamic_path = "/__physics_inspector/dynamic"
    _selected_path = "/__physics_inspector/selected"
    _selection_path = "/.Resources/PhysicsInspection"

    def __init__(
        self,
        app,
        world=None,
        *,
        source: PhysicsInspectionSource | None = None,
        visualizer=None,
        plane_size: float = 10.0,
    ):
        if not np.isfinite(plane_size) or plane_size <= 0:
            raise ValueError("plane_size must be finite and positive")
        previous = getattr(app, "_physics_inspector", None)
        if previous is not None and not previous._released:
            raise RuntimeError("An App can have only one physics inspector")
        if world is not None:
            if source is not None:
                raise ValueError("Supply world or source, not both")
            from ..physics.inspection import PhysXInspectionSource

            if visualizer is not None and visualizer.app is not app:
                raise ValueError("visualizer must use the same App")
            source = PhysXInspectionSource(world, visualizer=visualizer)
        elif visualizer is not None:
            raise ValueError("visualizer requires world")
        self._app = app
        self._source = None
        self._bindings = {}
        # A sentinel forces lookup after binding or Refresh, even with no selection.
        self._selection_key = object()
        self._selected_body = None
        self._selected_info = None
        self._selected_lines = None
        self._selected_pose = None
        self._plane_size = plane_size
        self._static_visible = False
        self._dynamic_visible = False
        self._collider_picking = False
        self._collider_body = None
        self._collider_shape_idx = None
        self._released = False
        self._shape_count = 0
        self._unsupported = ()
        self._dynamic_unsupported = ()
        self._dynamic_bodies = []
        self._dynamic_lines = {}
        self._dynamic_poses = None
        self._dynamic_valid = None
        self._dynamic_revision = None
        self._gpu_display = None
        self._gpu_resources = {}
        self._contacts_visible = False
        self._contact_display = None
        self._com_display = None
        self._body_com_visible = False
        self._articulation_com_visible = False
        self._frame_display = None
        self._body_frames_visible = False
        self._joint_frames_visible = False
        self._display_toggles = {}
        self.set_source(source)
        app._physics_inspector = self

    @property
    def source(self) -> PhysicsInspectionSource | None:
        return self._source

    def set_source(self, source: PhysicsInspectionSource | None) -> None:
        """Connect a provider, or disconnect with None; reset display and selection caches."""
        if self._released:
            raise RuntimeError("PhysicsInspector has been released")
        if source is self._source:
            return
        if self._collider_picking:
            self.set_collider_picking_enabled(False)
        if self._static_visible:
            self.set_static_visible(False)
        if self._dynamic_visible:
            self.set_dynamic_visible(False)
        if self._contacts_visible:
            self.set_contacts_visible(False)
        if self._body_com_visible or self._articulation_com_visible:
            self.set_com_visible(bodies=False, articulations=False)
        if self._body_frames_visible or self._joint_frames_visible:
            self.set_frames_visible(bodies=False, joints=False)
        self._clear_highlight()
        if self._source is not None:
            self._app._set_physics_inspector_callback(None)
        self._bindings.clear()
        self._selected_body = None
        self._selected_info = None
        self._selected_lines = None
        self._selection_key = object()
        self._source = source
        if source is not None:
            self._app._set_physics_inspector_callback(self._draw_menu)

    def _require_source(self):
        if self._released:
            raise RuntimeError("PhysicsInspector has been released")
        if self._source is None:
            raise RuntimeError("Connect a physics inspection source first")

    def _update_sync_callback(self):
        enabled = (
            self._static_visible
            or self._dynamic_visible
            or self._contacts_visible
            or self._body_com_visible
            or self._articulation_com_visible
            or self._body_frames_visible
            or self._joint_frames_visible
        )
        self._app._set_physics_inspector_sync_callback(self.sync if enabled else None)

    @property
    def static_visible(self) -> bool:
        return self._static_visible

    @property
    def dynamic_visible(self) -> bool:
        return self._dynamic_visible

    @property
    def collider_picking_enabled(self) -> bool:
        return self._collider_picking

    @property
    def contacts_visible(self) -> bool:
        return self._contacts_visible

    def set_contacts_visible(self, visible: bool) -> None:
        """Show contact normals, estimated modes, and normal forces; off by default."""
        self._require_source()
        if bool(visible) == self._contacts_visible:
            return
        if visible:
            if not getattr(self._source, "supports_contacts", False):
                raise RuntimeError(
                    getattr(
                        self._source,
                        "contacts_unavailable_reason",
                        "This provider does not supply contact data.",
                    )
                )
            if self._contact_display is None:
                from ._contact_display import ContactDisplay

                self._contact_display = ContactDisplay(
                    self._app, "/__debug__/physics_contacts"
                )
            self._contact_display.refresh(self._source)
        else:
            self._contact_display.release()
            self._source.clear_contact_cache()
        self._contacts_visible = bool(visible)
        self._update_sync_callback()

    @property
    def body_com_visible(self) -> bool:
        return self._body_com_visible

    @property
    def articulation_com_visible(self) -> bool:
        return self._articulation_com_visible

    def set_com_visible(self, *, bodies: bool, articulations: bool) -> None:
        """Show body/link and whole-articulation COM overlays; both default to off."""
        self._require_source()
        if bodies or articulations:
            if not getattr(self._source, "supports_com", False):
                raise RuntimeError(
                    "COM display requires a provider with body poses and masses"
                )
            if self._com_display is None:
                from ._com_display import COMDisplay

                self._com_display = COMDisplay(self._app)
        elif self._com_display is not None:
            self._com_display.release()
            self._source.clear_com_cache()
        self._body_com_visible = bool(bodies)
        self._articulation_com_visible = bool(articulations)
        self._update_sync_callback()

    def set_collider_picking_enabled(self, enabled: bool) -> None:
        """Pick visible collider categories in Inspect mode, including bodies without visuals."""
        self._require_source()
        self._collider_picking = bool(enabled)
        self._update_pick_callback()

    @property
    def body_frames_visible(self) -> bool:
        return self._body_frames_visible

    @property
    def joint_frames_visible(self) -> bool:
        return self._joint_frames_visible

    def set_frames_visible(self, *, bodies: bool, joints: bool) -> None:
        """Show body/link and child-side joint axes; both default to off."""
        self._require_source()
        if bodies or joints:
            if not getattr(self._source, "supports_frames", False):
                raise RuntimeError(
                    "Frame display requires a provider with body and joint poses"
                )
            if self._frame_display is None:
                from ._frame_display import FrameDisplay

                self._frame_display = FrameDisplay(self._app)
        elif self._frame_display is not None:
            self._frame_display.release()
            self._source.clear_frame_cache()
        self._body_frames_visible = bool(bodies)
        self._joint_frames_visible = bool(joints)
        self._update_sync_callback()

    def add_display_toggle(
        self, label: str, callback: Callable[[bool], None], *, enabled: bool = False
    ) -> None:
        """Add an application display checkbox; apply its initial value once."""
        self._require_source()
        if label in self._display_toggles:
            raise ValueError(f"Display toggle already exists: {label}")
        callback(bool(enabled))
        self._display_toggles[label] = (bool(enabled), callback)

    def set_display_toggle(self, label: str, enabled: bool) -> None:
        current, callback = self._display_toggles[label]
        if current != bool(enabled):
            callback(bool(enabled))
            self._display_toggles[label] = (bool(enabled), callback)

    def remove_display_toggle(self, label: str) -> None:
        """Remove the checkbox without changing application visibility."""
        self._display_toggles.pop(label, None)

    def _update_pick_callback(self):
        enabled = self._collider_picking and (
            self._static_visible or self._dynamic_visible
        )
        self._app._set_physics_inspector_pick_callback(
            self._pick_collider if enabled else None
        )
        if not enabled:
            self._collider_body = None
            self._collider_shape_idx = None
        self._selection_key = object()

    def _pick_collider(self, origin, direction, distance):
        picked = self._source.pick(
            origin,
            direction,
            distance,
            self._static_visible,
            self._dynamic_visible,
        )
        self._collider_body, self._collider_shape_idx = (
            picked if picked else (None, None)
        )
        self._selection_key = object()
        return picked is not None

    def bind_visual(self, visual, body, *, instance_index: int | None = None) -> None:
        """Connect a scene visual to a rigid body or Articulation.link() reference.

        ExternalBuffer visuals require an explicit instance_index.
        SimWorldVisualizer selections are resolved automatically when supplied.
        """
        self._require_source()
        component = visual.component
        external = component.transform_source == TransformSource.EXTERNAL_BUFFER
        if visual._app is not self._app or not component.attached:
            raise ValueError("visual must belong to this App and remain attached")
        if external and (instance_index is None or instance_index < 0):
            raise ValueError("ExternalBuffer selection requires an instance_index")
        if not external and instance_index is not None:
            raise ValueError("SceneGraph selection does not take an instance_index")
        self._bindings[component, instance_index] = self._source.body_reference(body)
        self._selection_key = object()

    def set_static_visible(self, visible: bool) -> None:
        """Refresh and show static colliders, or remove their debug lines."""
        self._require_source()
        if visible:
            self._refresh_static()
        else:
            self._app.debug_overlay.remove(self._path)
        self._static_visible = bool(visible)
        self._update_sync_callback()
        self._clear_highlight()
        if self._collider_picking:
            self._update_pick_callback()

    def set_dynamic_visible(self, visible: bool) -> None:
        """Show dynamic bodies or release their display cache."""
        self._require_source()
        if visible:
            if not self._source.supports_dynamic:
                raise RuntimeError(self._source.dynamic_unavailable_reason)
            self._refresh_dynamic()
        else:
            self._release_gpu_display()
            clear_cache = getattr(self._source, "clear_pose_cache", None)
            if clear_cache is not None:
                clear_cache()
            self._app.debug_overlay.remove(self._dynamic_path)
            self._dynamic_bodies = []
            self._dynamic_lines = {}
            self._dynamic_poses = None
            self._dynamic_valid = None
            self._dynamic_unsupported = ()
        self._dynamic_visible = bool(visible)
        self._selection_key = object()
        self._update_sync_callback()
        self._clear_highlight()
        if self._collider_picking:
            self._update_pick_callback()

    def refresh(self) -> None:
        """Replace visible geometry with a current snapshot without stepping physics."""
        if self._released:
            raise RuntimeError("PhysicsInspector has been released")
        if self._static_visible:
            self._refresh_static()
        if self._dynamic_visible:
            self._refresh_dynamic()
        self._selection_key = object()
        if self._contacts_visible:
            self._contact_display.refresh(self._source)
        if self._body_com_visible or self._articulation_com_visible:
            self._source.clear_com_cache()
        if self._body_frames_visible or self._joint_frames_visible:
            self._source.clear_frame_cache()

    def sync(self) -> None:
        """Synchronize enabled physics displays."""
        if self._released:
            raise RuntimeError("PhysicsInspector has been released")
        if self._dynamic_visible:
            self._sync_dynamic()
        if self._static_visible or self._dynamic_visible:
            self._sync_selection()
        if self._contacts_visible:
            self._contact_display.sync(self._source)
        if self._body_com_visible or self._articulation_com_visible:
            self._com_display.sync(
                self._source, self._body_com_visible, self._articulation_com_visible
            )
        if self._body_frames_visible or self._joint_frames_visible:
            self._frame_display.sync(
                self._source, self._body_frames_visible, self._joint_frames_visible
            )

    def _resolve_body(self, selection, key):
        if (
            selection.prim is not None
            and selection.prim.get_path() == self._selection_path
        ):
            return self._collider_body, {
                "path": "Collider",
                "picked_shape_idx": self._collider_shape_idx,
            }
        body = self._bindings.get(key)
        if body is not None:
            return body, {}

        # Native external picks can identify a batch without an owning prim.
        if selection.prim is None:
            system = self._app.get_scene_render_system()
            for (component, instance_index), body in self._bindings.items():
                if instance_index != selection.instance_index:
                    continue
                if system._handle(component, required=False) == selection.handle:
                    return body, {"path": component.owner.get_path()}

        return self._source.resolve_selection(selection)

    def _sync_selection(self):
        selection = self._app.get_selection()
        component = (
            selection.prim.get_render_component()
            if selection.hit and selection.prim is not None
            else None
        )
        external = selection.transform_source == TransformSource.EXTERNAL_BUFFER
        instance_index = selection.instance_index if external else None
        owner = component if component is not None else selection.prim
        key = (
            (owner if owner is not None else selection.handle, instance_index)
            if selection.hit
            else None
        )

        if key != self._selection_key:
            self._clear_highlight()
            self._selected_body = None
            self._selected_info = None
            self._selected_lines = None
            self._selection_key = key
            self._bindings = {
                binding: body
                for binding, body in self._bindings.items()
                if binding[0].attached and body.valid
            }
            if key is not None:
                body, identity = self._resolve_body(selection, key)
                info = self._source.body_info(body) if body is not None else None
                if info is not None:
                    path = (
                        selection.prim.get_path()
                        if selection.prim is not None
                        else f"ExternalBuffer batch {selection.handle}"
                    )
                    info.update(path=path, instance_index=instance_index)
                    info.update(identity)
                    self._selected_body = body
                    self._selected_info = info

        if self._selected_body is not None and not self._selected_body.valid:
            if self._selected_body is self._collider_body:
                self._app.clear_selection()
                self._collider_body = None
            self._selected_body = None
            self._selected_info = None
            self._selected_lines = None
            self._clear_highlight()
        self._sync_highlight()

    def _clear_highlight(self):
        if self._gpu_display is not None:
            self._gpu_display.select(None)
        if self._selected_pose is not None:
            self._app.debug_overlay.remove(self._selected_path)
        self._selected_pose = None

    def _sync_highlight(self):
        info = self._selected_info
        if info is None or not info["available"]:
            self._clear_highlight()
            return
        static = info["kind"] == "static"
        if not static and self._gpu_display is not None:
            self._gpu_display.select(info.get("display_body_idx"))
            return
        if not (self._static_visible if static else self._dynamic_visible):
            self._clear_highlight()
            return
        if self._selected_lines is None:
            picked = info.get("picked_shape_idx")
            lines = [
                _shape_lines(shape, self._plane_size, world_space=False)
                for shape in info["shapes"]
                if picked is None or shape["shape_idx"] == picked
            ]
            parts = [np.stack(line, axis=1) for line in lines if line is not None]
            self._selected_lines = (
                np.concatenate(parts) if parts else np.empty((0, 2, 3))
            )
        if not len(self._selected_lines):
            return

        if static:
            pose = np.asarray(info["shapes"][0]["actor_pose"])
        else:
            poses, valid = self._source.body_poses([self._selected_body])
            if not valid[0]:
                self._clear_highlight()
                return
            pose = poses[0]
        if np.array_equal(pose, self._selected_pose):
            return
        points = (self._selected_lines @ _rotation(pose[3:]).T + pose[:3]).astype(
            np.float32
        )
        color = _ke.ColorLibrary.get(_ke.ColorType.YELLOW)
        self._app.debug_overlay.lines(
            self._selected_path,
            points[:, 0],
            points[:, 1],
            [color.r, color.g, color.b, 1.0],
            width=3.0,
            overlay=True,
        )
        self._selected_pose = pose.copy()

    def _refresh_static(self):
        shapes = self._source.static_shapes()
        lines = []
        unsupported = set()
        for shape in shapes:
            result = _shape_lines(shape, self._plane_size)
            if result is None:
                unsupported.add(shape["type"])
            else:
                lines.append(result)

        if lines:
            starts, ends = (
                np.concatenate(parts).astype(np.float32) for parts in zip(*lines)
            )
            color = _ke.ColorLibrary.get(_ke.ColorType.PINK)
            self._app.debug_overlay.lines(
                self._path,
                starts,
                ends,
                [color.r, color.g, color.b, 1.0],
                width=1.5,
                overlay=True,
            )
        else:
            self._app.debug_overlay.remove(self._path)
        self._shape_count = len(lines)
        self._unsupported = tuple(sorted(unsupported))

    def _refresh_dynamic(self):
        self._release_gpu_display()
        shapes, bodies = self._source.dynamic_shapes()
        lines = {}
        unsupported = set()
        for shape in shapes:
            result = _shape_lines(shape, self._plane_size, world_space=False)
            if result is None:
                unsupported.add(shape["type"])
            else:
                lines.setdefault(shape["body_idx"], []).append(np.stack(result, axis=1))
        self._dynamic_lines = {
            idx: np.concatenate(parts) for idx, parts in lines.items()
        }
        self._dynamic_bodies = bodies
        self._dynamic_unsupported = tuple(sorted(unsupported))
        self._dynamic_poses = None
        self._dynamic_valid = None
        self._dynamic_revision = None
        self._sync_dynamic()

    def _sync_dynamic(self):
        revision = getattr(self._source, "pose_revision", None)
        if revision is not None and revision == self._dynamic_revision:
            return
        poses, valid = self._source.body_poses(self._dynamic_bodies)
        if getattr(poses, "is_cuda", False):
            if self._gpu_display is None:
                from ._gpu_collision_display import GPUCollisionDisplay

                self._gpu_display = GPUCollisionDisplay(
                    self._app,
                    self._dynamic_path,
                    self._dynamic_lines,
                    poses.device,
                    self._gpu_resources,
                )
            self._gpu_display.sync(poses, valid)
            self._dynamic_revision = revision
            return
        if np.array_equal(poses, self._dynamic_poses) and np.array_equal(
            valid, self._dynamic_valid
        ):
            return

        # Geometry stays actor-local; only current world poses are fetched each frame.
        lines = [
            points @ _rotation(poses[idx, 3:]).T + poses[idx, :3]
            for idx, points in self._dynamic_lines.items()
            if valid[idx]
        ]
        if lines:
            points = np.concatenate(lines).astype(np.float32)
            color = _ke.ColorLibrary.get(_ke.ColorType.GREEN)
            self._app.debug_overlay.lines(
                self._dynamic_path,
                points[:, 0],
                points[:, 1],
                [color.r, color.g, color.b, 1.0],
                width=1.5,
                overlay=True,
            )
        else:
            self._app.debug_overlay.remove(self._dynamic_path)
        self._dynamic_poses = poses
        self._dynamic_valid = valid

    def _release_gpu_display(self):
        if self._gpu_display is not None:
            self._gpu_display.release()
            self._gpu_display = None

    def _draw_menu(self):
        ui = _ke.imgui

        def color_swatch(color_type):
            color = _ke.ColorLibrary.get(color_type)
            x, y = ui.cursor_screen_pos()
            width, height = ui.calc_text_size("   ")
            ui.text("   ")
            ui.draw_rect_filled(
                x,
                y + (height - width) / 2,
                x + width,
                y + (height + width) / 2,
                _ke.Vec4(color.r, color.g, color.b, 1.0),
            )
            ui.same_line()

        color_swatch(_ke.ColorType.PINK)
        changed, visible = ui.checkbox("Static Colliders", self._static_visible)
        if changed:
            self.set_static_visible(visible)
        supports_dynamic = self._source.supports_dynamic
        color_swatch(_ke.ColorType.GREEN)
        ui.begin_disabled(not supports_dynamic)
        changed, visible = ui.checkbox("Dynamic Colliders", self._dynamic_visible)
        ui.end_disabled()
        if changed:
            self.set_dynamic_visible(visible)
        if not supports_dynamic:
            ui.text_wrapped(self._source.dynamic_unavailable_reason)
        self._draw_contacts_menu(color_swatch)
        from ._com_display import COM_COLORS

        ui.begin_disabled(not getattr(self._source, "supports_com", False))
        color_swatch(COM_COLORS[0])
        changed, bodies = ui.checkbox("Body / link COM", self._body_com_visible)
        color_swatch(COM_COLORS[1])
        changed_robot, articulations = ui.checkbox(
            "Whole robot COM", self._articulation_com_visible
        )
        ui.end_disabled()
        if changed or changed_robot:
            self.set_com_visible(bodies=bodies, articulations=articulations)
        ui.begin_disabled(not getattr(self._source, "supports_frames", False))
        changed, bodies = ui.checkbox("Body / link axes", self._body_frames_visible)
        changed_joint, joints = ui.checkbox("Joint axes", self._joint_frames_visible)
        ui.end_disabled()
        if changed or changed_joint:
            self.set_frames_visible(bodies=bodies, joints=joints)
        if bodies or joints:
            ui.text_disabled("X red / Y green / Z blue")
        if self._display_toggles:
            ui.separator()
            for label, (enabled, _) in self._display_toggles.items():
                changed, enabled = ui.checkbox(label, enabled)
                if changed:
                    self.set_display_toggle(label, enabled)
        ui.separator()
        ui.begin_disabled(not (self._static_visible or self._dynamic_visible))
        changed, enabled = ui.checkbox("Pick Colliders", self._collider_picking)
        ui.end_disabled()
        if changed:
            self.set_collider_picking_enabled(enabled)
        if self._collider_picking:
            ui.text_wrapped("Inspect mode: pick collision surfaces.")
            if not getattr(self._source, "supports_dynamic_picking", True):
                ui.text_wrapped(
                    "GPU bodies: select their visual; collider picking is static-only."
                )
        if (
            self._static_visible
            or self._dynamic_visible
            or self._contacts_visible
            or self._body_com_visible
            or self._articulation_com_visible
            or self._body_frames_visible
            or self._joint_frames_visible
        ):
            if ui.button("Refresh"):
                self.refresh()
            ui.text_wrapped("Refresh after adding actors or editing shapes.")
        if self._static_visible:
            ui.text(f"Static shapes: {self._shape_count}")
            ui.text_disabled("Static poses update on Refresh.")
            ui.text_disabled(f"Infinite plane preview: {self._plane_size:g} units")
        unsupported = set(self._unsupported if self._static_visible else ())
        unsupported.update(self._dynamic_unsupported)
        if unsupported:
            ui.text_wrapped("Not displayed: " + ", ".join(sorted(unsupported)))
        ui.separator()
        if not (self._static_visible or self._dynamic_visible):
            self._sync_selection()
        self._draw_selection()

    def _draw_contacts_menu(self, color_swatch):
        ui = _ke.imgui
        ui.separator()
        supported = getattr(self._source, "supports_contacts", False)
        ui.begin_disabled(not supported)
        changed, visible = ui.checkbox("Contacts", self._contacts_visible)
        ui.end_disabled()
        if changed:
            self.set_contacts_visible(visible)
        if not supported:
            ui.text_wrapped(
                getattr(
                    self._source,
                    "contacts_unavailable_reason",
                    "This provider does not supply contact data.",
                )
            )
        if self._contacts_visible:
            from ._contact_display import CONTACT_COLORS

            display = self._contact_display
            has_force = getattr(self._source, "supports_contact_forces", True)
            for field, label in (("normal", "Normal"), ("force", "Force")):
                if field == "force":
                    ui.same_line()
                    ui.begin_disabled(not has_force)
                color_swatch(CONTACT_COLORS[field])
                _, value = ui.checkbox(label, getattr(display, field))
                setattr(display, field, value)
                if field == "force":
                    ui.end_disabled()
            ui.begin_disabled(not has_force)
            _, display.mode = ui.checkbox("Contact Mode (estimated)", display.mode)
            ui.end_disabled()
            if not has_force:
                ui.text_disabled(
                    "Force data not supplied; showing contact normals only."
                )
            if display.mode and has_force:
                for name, label in (
                    ("open", "Unloaded"),
                    ("stick", "Sticking"),
                    ("slip", "Sliding"),
                ):
                    if name != "open":
                        ui.same_line()
                    color_swatch(CONTACT_COLORS[name])
                    ui.text(label)
            _, display.scale = ui.slider_float(
                "Contact scale", display.scale, 0.01, 100.0, True
            )
            if display.force and has_force:
                _, display.force_scale = ui.slider_float(
                    "Force scale", display.force_scale, 0.01, 100.0, True
                )
            if display.normal or (display.force and has_force):
                _, display.thickness = ui.slider_float(
                    "Arrow thickness", display.thickness, 0.25, 5.0
                )
            ui.text_disabled(f"Reported contact points: {display.count}")
            ui.text_disabled(
                "Force: normal impulse / physics step time."
                if getattr(self._source, "contact_forces_are_impulses", True)
                else "Force: normal component reported by the solver."
            )
            if getattr(self._source, "contacts_truncated", False):
                ui.text_wrapped(
                    "Contact buffer is full; some contact points are not displayed."
                )
        ui.separator()

    def _draw_selection(self):
        ui = _ke.imgui
        info = self._selected_info
        if info is None:
            ui.text_wrapped(
                "Click a collider to inspect it."
                if self._collider_picking
                else "Select a connected physics object."
            )
            return
        ui.text_wrapped(info["path"])
        ui.text("Body: " + info["kind"])
        if "env_id" in info:
            ui.text(
                f"Env {info['env_id']} / Object {info['obj_id']} / Body {info['body_id']}"
            )
        if info["instance_index"] is not None:
            ui.text(f"Instance: {info['instance_index']}")
        if not info["available"]:
            ui.text_wrapped(self._source.dynamic_unavailable_reason)
            return
        if not info["shapes"]:
            ui.text_disabled("No simulation shapes.")
        for shape in info["shapes"]:
            ui.separator()
            ui.text(f"Shape {shape['shape_idx']}: {shape['type']}")
            if shape["shape_idx"] == info.get("picked_shape_idx"):
                ui.text_disabled("Selected collider")
            if "half_extents" in shape:
                size = ", ".join(f"{2 * value:g}" for value in shape["half_extents"])
                ui.text("Size: " + size)
            if "radius" in shape:
                ui.text(f"Radius: {shape['radius']:g}")
            if "half_height" in shape:
                ui.text(f"Cylinder length: {2 * shape['half_height']:g} (X axis)")
            if "vertices" in shape:
                ui.text(
                    f"Vertices: {len(shape['vertices'])}, triangles: {len(shape['triangles'])}"
                )
            if "mesh_scale" in shape:
                ui.text(
                    "Mesh scale: " + ", ".join(f"{v:g}" for v in shape["mesh_scale"])
                )
            pose = shape["local_pose"]
            ui.text("Local position: " + ", ".join(f"{v:.3g}" for v in pose[:3]))
            ui.text("Local rotation (XYZW): " + ", ".join(f"{v:.3g}" for v in pose[3:]))

    def release(self) -> None:
        """Detach the menu and remove its debug geometry; leave physics unchanged."""
        if self._released:
            return
        self.set_source(None)
        self._display_toggles.clear()
        self._gpu_resources.clear()
        self._app._physics_inspector = None
        self._released = True
