"""Python-friendly base App for KangEngine examples and tools.

The native C++ App still owns the main loop, camera controls, renderer, ImGui
frame lifecycle, and scene rendering.  This wrapper adds default lifecycle
hooks and small input helpers so examples can inherit from `kangengine.App`
without talking directly to the pybind class.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .._core import _ke
from .._public import unwrap_native
from .. import geometry as geometry_api
from .. import input as input_api
from .. import material as material_api
from .. import render as render_api
from ..recording import VideoCaptureController

keys = _ke.keys
scene = _ke.scene
_NativeApp = _ke.App


def _safe_scene_segment(value, fallback: str = "item") -> str:
    text = str(value or fallback)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe or fallback


def _texture_path_candidate(path) -> Path:
    """Return an existing texture path, accepting OBJ/MTL Windows separators."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    text = str(path)
    if "\\" in text:
        normalized = Path(text.replace("\\", "/")).expanduser()
        if normalized.exists():
            return normalized
    return candidate


def _rotation_matrix(rotation):
    """Return a float32 3x3 matrix from a matrix or wxyz quaternion."""
    import numpy as np

    if rotation is None:
        return np.eye(3, dtype=np.float32)
    value = rotation.to_wxyz() if hasattr(rotation, "to_wxyz") else rotation
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (3, 3):
        return array
    w, x, y, z = array.reshape(4)
    norm = float(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("rotation quaternion must be non-zero")
    scale = 2.0 / norm
    return np.array(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


def _sphere_instance_data(centers, radii, colors):
    import numpy as np

    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    if len(centers) == 0:
        raise ValueError("centers must contain at least one position")
    radii = np.asarray(radii, dtype=np.float32)
    if radii.ndim == 0:
        radii = np.full(len(centers), float(radii), dtype=np.float32)
    else:
        radii = radii.reshape(-1)
        if len(radii) != len(centers):
            raise ValueError("radii must be scalar or match centers length")
    if colors is None:
        colors = np.ones((len(centers), 4), dtype=np.float32)
    else:
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 4)
        if len(colors) == 1 and len(centers) > 1:
            colors = np.repeat(colors, len(centers), axis=0)
        elif len(colors) != len(centers):
            raise ValueError("colors must have length 1 or match centers")

    transforms = np.repeat(
        np.eye(4, dtype=np.float32)[None, :, :], len(centers), axis=0
    )
    transforms[:, 0, 0] = radii
    transforms[:, 1, 1] = radii
    transforms[:, 2, 2] = radii
    # Native instance matrices use GLM's column-major memory layout.
    transforms[:, 3, :3] = centers
    return transforms, colors


class RenderablePrimView:
    """User-facing view for one scene prim and its renderer resources."""

    def __init__(self, app: "App", prim, component):
        self._app = app
        self.prim = prim
        self.component = component

    @property
    def _render_system(self):
        return self._app.get_scene_render_system()

    @property
    def path(self) -> str:
        return self.prim.get_path()

    def _require_scene_graph_transform(self):
        if self.component.transform_source != render_api.TransformSource.SCENE_GRAPH:
            raise RuntimeError(
                f"{self.path} uses an external transform buffer; "
                "Prim local/world transforms do not drive this renderable"
            )

    def set_local_translation(self, translation):
        """Set the parent-relative translation."""
        self._require_scene_graph_transform()
        self.prim.set_local_translation(translation)
        return self

    def set_local_rotation(self, rotation):
        """Set the parent-relative quaternion rotation."""
        self._require_scene_graph_transform()
        self.prim.set_local_rotation(rotation)
        return self

    def set_local_rotation_axis_angle(self, axis, angle_radians: float):
        """Set parent-relative rotation from an axis and angle in radians."""
        self._require_scene_graph_transform()
        self.prim.set_local_rotation_axis_angle(axis, angle_radians)
        return self

    def set_local_scale(self, scale):
        """Set the parent-relative scale."""
        self._require_scene_graph_transform()
        self.prim.set_local_scale(scale)
        return self

    def set_local_matrix(self, matrix):
        """Set the parent-relative transform matrix."""
        self._require_scene_graph_transform()
        self.prim.set_local_matrix(matrix)
        return self

    def set_world_translation(self, translation):
        """Set translation in world space."""
        self._require_scene_graph_transform()
        self.prim.set_world_translation(translation)
        return self

    def set_world_rotation(self, rotation):
        """Set quaternion rotation in world space."""
        self._require_scene_graph_transform()
        self.prim.set_world_rotation(rotation)
        return self

    def set_world_rotation_axis_angle(self, axis, angle_radians: float):
        """Set world rotation from an axis and angle in radians."""
        self._require_scene_graph_transform()
        self.prim.set_world_rotation_axis_angle(axis, angle_radians)
        return self

    def set_world_matrix(self, matrix):
        """Set the transform matrix in world space."""
        self._require_scene_graph_transform()
        self.prim.set_world_matrix(matrix)
        return self

    def get_local_translation(self):
        """Return the effective parent-relative translation."""
        self._require_scene_graph_transform()
        return self.prim.get_local_translation()

    def get_local_rotation(self):
        """Return the effective parent-relative quaternion rotation."""
        self._require_scene_graph_transform()
        return self.prim.get_local_rotation()

    def get_world_translation(self):
        """Return the effective world-space translation."""
        self._require_scene_graph_transform()
        return self.prim.get_world_translation()

    def get_world_rotation(self):
        """Return the effective world-space quaternion rotation."""
        self._require_scene_graph_transform()
        return self.prim.get_world_rotation()

    def compute_local_matrix(self):
        """Compute the effective parent-relative transform matrix."""
        self._require_scene_graph_transform()
        return self.prim.compute_local_matrix()

    def compute_world_matrix(self):
        """Compute the effective world-space transform matrix."""
        self._require_scene_graph_transform()
        return self.prim.compute_world_matrix()

    def set_visible(self, visible: bool):
        self.component.visible = bool(visible)
        return self

    def set_double_sided(self, enabled: bool = True):
        self._render_system.set_double_sided(self.component, bool(enabled))
        return self

    def set_casts_shadow(self, enabled: bool = True):
        self._render_system.set_casts_shadow(self.component, bool(enabled))
        return self

    def set_alpha_mode(self, mode, cutoff: float = 0.5):
        """Choose opaque, cutout-mask, or blended alpha rendering."""
        self._render_system.set_alpha_mode(self.component, mode, float(cutoff))
        return self

    def set_base_color(self, color):
        """Set the per-instance base-color multiplier for this renderable."""
        self.prim.set_display_color_alpha(color)
        return self

    def get_base_color(self):
        """Return the per-instance base-color multiplier."""
        return self.prim.get_display_color_alpha()

    def set_texture(self, texture, role_or_slot=render_api.TextureRole.BASE_COLOR):
        self._render_system.set_texture(
            self.component, unwrap_native(texture), role_or_slot
        )
        return self

    def set_material(self, material):
        """Replace this renderable's material and move it to the right batch."""
        self._render_system.set_material(self.component, unwrap_native(material))
        return self

    def set_transform_buffer(
        self,
        transforms,
        *,
        sim_device=None,
        sync_policy=None,
    ):
        """Set the ``[N, 4, 4]`` buffer of an ExternalBuffer renderable."""
        from ..utils.sim_buffer import to_external_transform_desc

        descriptor, _ = to_external_transform_desc(
            transforms,
            sim_device=sim_device,
            dtype="float32",
            name=f"{self.path}:transforms",
            sync_policy=(
                render_api.ExternalSyncPolicy.NONE
                if sync_policy is None
                else sync_policy
            ),
        )

        self._render_system.set_external_buffer(self.component, descriptor)
        return self

    def update_geometry(self, positions, normals=None):
        """Update dynamic vertex positions and optional normals."""
        self._render_system.update_geometry(self.component, positions, normals)
        return self

    def update_skinning(self, bone_matrices):
        """Update skinned bone matrices for this renderable."""
        self._render_system.update_skinning(self.component, bone_matrices)
        return self

    def remove(self):
        return self._app.remove_prim(self.prim)


class DebugPrimitiveView(RenderablePrimView):
    """Component-backed instanced debug primitive view."""

    def update_instances(self, transforms, colors=None):
        """Update raw instance transforms and optional colors."""
        self._render_system.update_instances(self.component, transforms, colors)
        return self

    def update_lines(self, starts, ends, colors=None):
        """Update this view as instanced debug lines."""
        scene.DebugDraw.update_component_lines(
            self._app,
            self.component,
            starts,
            ends,
            colors,
        )
        return self

    def update_arrows(self, starts, ends, colors=None):
        """Update this view as instanced debug arrows."""
        scene.DebugDraw.update_component_arrows(
            self._app,
            self.component,
            starts,
            ends,
            colors,
        )
        return self

    def update_spheres(self, centers, radii=0.5, colors=None):
        """Update instanced sphere positions, radii, and colors."""
        transforms, colors = _sphere_instance_data(centers, radii, colors)
        return self.update_instances(transforms, colors)


class ObjImportView:
    """Result returned by SceneContext.add_obj()."""

    def __init__(self, root, views, info):
        self.root = root
        self.views = views
        self.info = info

    def __iter__(self):
        return iter(self.views)

    def __len__(self):
        return len(self.views)


class DebugGeometry:
    """Mesh-based debug geometry owned by the SceneGraph."""

    def __init__(self, scene_context: "SceneContext"):
        self._scene = scene_context

    def add_lines(
        self,
        path: str,
        starts,
        ends,
        colors=None,
        *,
        material=None,
        radius: float = 0.005,
        segments: int = 8,
    ):
        """Add instanced line meshes and return a DebugPrimitiveView."""
        if material is None:
            material = self._scene._app.create_standard_materials().debug
        return self._scene.log_lines(
            path, material, starts, ends, colors, radius, segments
        )

    def add_arrows(
        self,
        path: str,
        starts,
        ends,
        colors=None,
        *,
        material=None,
        radius: float = 0.02,
        segments: int = 12,
    ):
        """Add instanced arrow meshes and return a DebugPrimitiveView."""
        if material is None:
            material = self._scene._app.create_standard_materials().debug
        return self._scene.log_arrows(
            path, material, starts, ends, colors, radius, segments
        )

    def add_axes(
        self,
        path: str,
        origin,
        rotation=None,
        *,
        length: float = 1.0,
        material=None,
        radius: float = 0.005,
        segments: int = 8,
    ):
        """Add RGB axis meshes and return a DebugPrimitiveView."""
        import numpy as np

        origin = np.asarray(origin, dtype=np.float32).reshape(3)
        basis = _rotation_matrix(rotation)
        starts = np.repeat(origin[None, :], 3, axis=0)
        # Rotation columns are the world-space directions of local X/Y/Z.
        ends = starts + basis.T * float(length)
        colors = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        return self.add_lines(
            path,
            starts,
            ends,
            colors,
            material=material,
            radius=radius,
            segments=segments,
        )

    def add_spheres(
        self,
        path: str,
        centers,
        radii=0.5,
        colors=None,
        *,
        material=None,
        segments: int = 16,
        rings: int = 12,
    ):
        """Add instanced solid spheres and return a DebugPrimitiveView."""
        transforms, colors = _sphere_instance_data(centers, radii, colors)
        if material is None:
            material = self._scene._app.create_standard_materials().debug
        view = self._scene.add_mesh(
            path,
            geometry_api.create_sphere_data(1.0, segments, rings),
            material,
        )
        view._render_system.update_instances(view.component, transforms, colors)
        view.set_casts_shadow(False)
        return DebugPrimitiveView(self._scene._app, view.prim, view.component)


class DebugOverlay:
    """OpenGL debug overlay that does not create SceneGraph prims."""

    def __init__(self, app: "App"):
        self._app = app

    def lines(
        self,
        path: str,
        starts,
        ends,
        colors=None,
        *,
        width: float = 1.0,
        hidden: bool = False,
    ):
        self._app.log_debug_lines(path, starts, ends, colors, width, hidden)
        return self

    def points(
        self,
        path: str,
        points,
        colors=None,
        *,
        size: float = 6.0,
        hidden: bool = False,
        overlay: bool = False,
    ):
        self._app.log_debug_points(path, points, colors, size, hidden, overlay)
        return self

    def axes(
        self,
        path: str,
        origin,
        rotation=None,
        *,
        length: float = 1.0,
        width: float = 1.0,
        hidden: bool = False,
    ):
        import numpy as np

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = _rotation_matrix(rotation)
        transform[:3, 3] = np.asarray(origin, dtype=np.float32).reshape(3)
        self._app.log_debug_axes(path, transform, length, width, hidden)
        return self

    def clear_lines(self, path: str):
        self._app.clear_debug_lines(path)
        return self

    def clear_points(self, path: str):
        self._app.clear_debug_points(path)
        return self

    def clear(self, path: str):
        """Clear both line/axis and point overlay batches at a path."""
        self._app.clear_debug_lines(path)
        self._app.clear_debug_points(path)
        return self


class WorldText:
    """Persistent screen-aligned text anchored at world-space positions."""

    def __init__(self, app: "App"):
        self._app = app

    def set(
        self,
        path: str,
        text: str,
        position,
        *,
        color=None,
        pixel_size: float = 18.0,
        alignment=None,
        depth_test: bool = True,
        hidden: bool = False,
    ):
        if color is None:
            color = _ke.Vec4(1.0, 1.0, 1.0, 1.0)
        if alignment is None:
            alignment = render_api.TextAlignment.CENTER
        depth_mode = (
            render_api.TextDepthMode.DEPTH_TESTED
            if depth_test
            else render_api.TextDepthMode.OVERLAY
        )
        self._app.set_world_text(
            path,
            text,
            position,
            color,
            float(pixel_size),
            alignment,
            depth_mode,
            bool(hidden),
        )
        return self

    def set_text(self, path: str, text: str):
        self._app.set_world_text_string(path, text)
        return self

    def set_position(self, path: str, position):
        self._app.set_world_text_position(path, position)
        return self

    def set_hidden(self, path: str, hidden: bool):
        self._app.set_world_text_hidden(path, bool(hidden))
        return self

    def remove(self, path: str):
        self._app.remove_world_text(path)
        return self

    def clear(self):
        self._app.clear_world_text()
        return self


class ScreenText:
    """Persistent text positioned in viewport pixel(screen) coordinates."""

    def __init__(self, app: "App"):
        self._app = app

    def set(
        self,
        path: str,
        text: str,
        position,
        *,
        color=None,
        pixel_size: float = 18.0,
        alignment=None,
        anchor=None,
        hidden: bool = False,
    ):
        if color is None:
            color = _ke.Vec4(1.0, 1.0, 1.0, 1.0)
        if anchor is None:
            anchor = render_api.ScreenAnchor.TOP_LEFT
        if alignment is None:
            if anchor in (
                render_api.ScreenAnchor.TOP_CENTER,
                render_api.ScreenAnchor.CENTER,
                render_api.ScreenAnchor.BOTTOM_CENTER,
            ):
                alignment = render_api.TextAlignment.CENTER
            elif anchor in (
                render_api.ScreenAnchor.TOP_RIGHT,
                render_api.ScreenAnchor.CENTER_RIGHT,
                render_api.ScreenAnchor.BOTTOM_RIGHT,
            ):
                alignment = render_api.TextAlignment.RIGHT
            else:
                alignment = render_api.TextAlignment.LEFT
        self._app.set_screen_text(
            path,
            text,
            position,
            color,
            float(pixel_size),
            alignment,
            anchor,
            bool(hidden),
        )
        return self

    def set_text(self, path: str, text: str):
        self._app.set_screen_text_string(path, text)
        return self

    def set_position(self, path: str, position):
        self._app.set_screen_text_position(path, position)
        return self

    def set_hidden(self, path: str, hidden: bool):
        self._app.set_screen_text_hidden(path, bool(hidden))
        return self

    def remove(self, path: str):
        self._app.remove_screen_text(path)
        return self

    def clear(self):
        self._app.clear_screen_text()
        return self


class SceneContext:
    """Scene-facing facade connected to the owning App renderer.

    Use this for common add/remove workflows. It keeps renderer handles inside
    the app-facing layer while preserving access to the underlying scene backend
    for lower-level operations.
    """

    def __init__(self, app: "App"):
        self._app = app
        self.debug_geometry = DebugGeometry(self)

    @property
    def native(self):
        """Return the native SceneBackend escape hatch.

        Prefer SceneContext helpers such as add_mesh(), add_ground(), and
        define_prim() for authored scene objects. Use native only when a C++ or
        pybind API explicitly requires SceneBackend.
        """
        return self._app.get_native_scene()

    def define_prim(self, path: str, prim_type):
        return self.native.define_prim(path, prim_type)

    def get_prim_at_path(self, path: str):
        return self.native.get_prim_at_path(path)

    def get_root_prim(self):
        return self.native.get_root_prim()

    def add_renderable(
        self,
        prim,
        material,
        transform_source=None,
    ):
        """Register a scene prim as renderable through RenderComponent.

        This is the preferred public path for authored scene objects. It
        returns a RenderablePrimView facade instead of exposing the native
        renderer handle. Pass a Material describing the render surface.
        """
        if transform_source is None:
            transform_source = render_api.TransformSource.SCENE_GRAPH
        self._app._add_renderable(material, prim, transform_source)
        component = prim.get_render_component()
        if component is None:
            raise RuntimeError(f"failed to register renderable prim: {prim.get_path()}")
        return RenderablePrimView(self._app, prim, component)

    def add_mesh(
        self,
        path: str,
        mesh_data,
        material,
        color=None,
        transform_source=None,
        uri=None,
    ):
        prim = self.define_prim(path, scene.PrimType.MESH)
        mesh_handle = self._app._register_mesh_resource(
            mesh_data,
            display_name=Path(path).name or "Mesh",
            uri=uri or f"scene://mesh{path}",
        )
        mesh_component = prim.get_mesh_component()
        if mesh_component is None:
            mesh_component = prim.add_mesh_component()
        # Keep the render prim on the fast path by caching the shared mesh
        # directly. The resource handle remains the editor/resource identity,
        # but renderer registration no longer depends on traversing /.Resources.
        mesh_component.mesh_data = mesh_data
        mesh_component.resource_handle = mesh_handle
        if color is not None:
            prim.set_display_color_alpha(color)
        return self.add_renderable(prim, material, transform_source)

    def add_obj(
        self,
        path: str,
        obj_path,
        *,
        transform_source=None,
        double_sided=False,
    ):
        """Load an OBJ/MTL file and add material-subset mesh prims.

        Multi-material OBJ files become one Xform root with one renderable child
        per material subset. Single-material OBJ files still use the same code
        path, which keeps resource registration and material creation
        consistent.
        """
        obj_path = Path(obj_path)
        info = _ke.asset.load_obj_with_materials(str(obj_path))
        root = self.define_prim(path, scene.PrimType.XFORM)

        subsets = list(info.subsets)
        if not subsets:
            subsets = [
                SimpleNamespace(
                    name=obj_path.stem,
                    material_index=int(info.primary_material_index),
                    mesh_data=info.mesh_data,
                )
            ]

        views = []
        used_child_names = set()
        for index, subset in enumerate(subsets):
            material = self._app._create_obj_subset_material(info, subset)
            base_child_name = _safe_scene_segment(
                subset.name if subset.name != "default" else f"subset_{index}",
                f"subset_{index}",
            )
            child_name = base_child_name
            suffix = 1
            while child_name in used_child_names:
                child_name = f"{base_child_name}_{suffix}"
                suffix += 1
            used_child_names.add(child_name)
            child_path = f"{path.rstrip('/')}/{child_name}"
            view = self.add_mesh(
                child_path,
                subset.mesh_data,
                material,
                color=self._app._obj_subset_display_color(info, subset),
                transform_source=transform_source,
                uri=f"{obj_path}#{subset.name or index}",
            )
            if double_sided:
                view.set_double_sided(True)
            if self._app._obj_subset_has_alpha_texture(info, subset):
                view.set_alpha_mode(render_api.AlphaMode.MASK)
            elif self._app._obj_subset_alpha(info, subset) < 1.0:
                view.set_alpha_mode(render_api.AlphaMode.BLEND)
            views.append(view)

        return ObjImportView(root, views, info)

    def add_ground(
        self,
        path: str = "/ground",
        scale: float = 20.0,
        material=None,
    ):
        """Add a checkerboard ground plane."""
        if material is None:
            material = self._app.create_standard_materials().ground
        return self.add_mesh(
            path,
            _ke.geometry.create_plane_data(float(scale), self._app.up_axis),
            material,
        )

    def log_lines(
        self,
        path: str,
        material,
        starts,
        ends,
        colors=None,
        radius: float = 0.005,
        segments: int = 8,
    ):
        component = scene.DebugDraw.log_component_lines(
            self._app,
            unwrap_native(material),
            path,
            starts,
            ends,
            colors,
            float(radius),
            int(segments),
        )
        if component is None or component.owner is None:
            raise RuntimeError(f"failed to register debug line prim: {path}")
        return DebugPrimitiveView(self._app, component.owner, component)

    def log_arrows(
        self,
        path: str,
        material,
        starts,
        ends,
        colors=None,
        radius: float = 0.02,
        segments: int = 12,
    ):
        component = scene.DebugDraw.log_component_arrows(
            self._app,
            unwrap_native(material),
            path,
            starts,
            ends,
            colors,
            float(radius),
            int(segments),
        )
        if component is None or component.owner is None:
            raise RuntimeError(f"failed to register debug arrow prim: {path}")
        return DebugPrimitiveView(self._app, component.owner, component)

    def remove_prim(self, path_or_prim):
        return self._app.remove_prim(path_or_prim)


class App(_NativeApp):
    """Base class for Python KangEngine apps.

    This mirrors the C++ App lifecycle:

    - `setup()` runs once before the loop starts.
    - `pre_update()` handles per-frame input before simulation updates.
    - `fixed_update(dt)` runs zero or more fixed updates per rendered frame.
    - `pre_render()` runs before scene rendering each frame.
    - `render()` runs while the ImGui frame is active.
    - `post_render()` runs after UI rendering and buffer swap setup.

    The C++ implementation still handles the default camera controls:
    WASD/mouse navigation, H to hide UI, Escape to close, framebuffer updates,
    and the built-in scene/performance panels.

    Fixed-step simulations may opt into Enter play/pause and Space
    pause/single-step controls with `set_simulation_hotkeys_enabled(True)`.
    """

    def __init__(self):
        super().__init__()
        self._previous_key_states = {}
        self.width = 1920
        self.height = 1080
        self.hide_ui = False
        self.timing_config = None
        self.run_config = None
        self._video_capture = VideoCaptureController()
        # Lazily populated after the native graphics device/context exists.
        # Pure compute/headless apps can leave this as None.
        self.shaders = None
        self.standard_materials = None
        self.materials = []
        self.textures = []
        self.up_axis = _ke.UpAxis.Y
        self.graphics_backend_type = render_api.BackendType.OPENGL
        self.scene_backend_type = scene.BackendType.NATIVE
        self.headless = False
        self.scene = SceneContext(self)
        self.input = input_api.Input(self)
        self.debug_overlay = DebugOverlay(self)
        self.world_text = WorldText(self)
        self.screen_text = ScreenText(self)
        self.resources = self.get_scene_resource_manager()
        self._resource_handles_by_object_id = {}
        self._resource_counter = 0
        self._textures_by_uri = {}
        self._scene_hook_resource_handles = {}
        self._render_hook_resource_usage = {}

    def get_native_scene(self):
        """Return the native SceneBackend escape hatch.

        Public Python code should prefer self.scene for authored scene workflows.
        Use this only when a C++ or pybind API explicitly requires SceneBackend.
        """
        return super().get_scene()

    def get_scene(self):
        """Return the Python-friendly SceneContext facade.

        This mirrors the ``app.scene`` property. Use get_native_scene() or
        scene.native only when a C++ or pybind API explicitly requires the
        native SceneBackend.
        """
        return self.scene

    def get_renderer(self):
        """Return the native renderer exposed through ``ke.render``."""
        return super().get_renderer()

    def create_scene_hook_pipeline(
        self, desc, *, shader_uris=None, shader_languages=None
    ):
        """Create a custom scene pipeline and mirror its authored definition.

        The returned backend pipeline remains caller-owned. Shader sources and
        the pipeline definition are copied into ``/.Resources`` for editor
        inspection; compiled backend objects are not scene resources.
        """
        pipeline = self.get_renderer().create_scene_hook_pipeline(desc)
        uris = list(shader_uris or ())
        languages = list(shader_languages or ())
        shader_handles = []
        for index, stage in enumerate(desc.shader.stages):
            source = scene.ShaderSourceResource()
            source.stage = stage.stage
            source.language = (
                languages[index]
                if index < len(languages)
                else scene.ShaderLanguage.GLSL
            )
            source.source = stage.source
            source.entry_point = stage.entry_point
            stage_name = str(stage.stage).rsplit(".", 1)[-1].lower()
            uri = uris[index] if index < len(uris) else ""
            shader_handles.append(
                self.resources.register_shader_source(
                    f"{desc.shader.name or desc.label}_{stage_name}", source, uri
                )
            )

        authored = scene.PipelineResource()
        authored.type = scene.PipelineType.GRAPHICS
        authored.shader_sources = shader_handles
        authored.state_summary = (
            f"topology={desc.topology}, cull={desc.cull_mode}, "
            f"blend={desc.blend}, depth_test={desc.depth_test}, "
            f"depth_write={desc.depth_write}"
        )
        pipeline_handle = self.resources.register_pipeline(
            desc.label, authored, f"python://pipeline/{desc.label}"
        )
        self._scene_hook_resource_handles[id(pipeline)] = (
            tuple(shader_handles),
            pipeline_handle,
        )
        return pipeline

    def add_render_hook(self, phase, callback, *, pipeline=None):
        """Register a custom draw callback and optionally track its pipeline."""
        pipeline_record = None
        if pipeline is not None:
            pipeline_record = self._scene_hook_resource_handles.get(id(pipeline))
            if pipeline_record is None:
                raise ValueError(
                    "pipeline was not created by App.create_scene_hook_pipeline()"
                )

        handle = self.get_renderer().add_render_hook(phase, callback)
        if pipeline_record is not None:
            _, pipeline_handle = pipeline_record
            usage_path = f"render-hook://{handle}"
            self.resources.add_external_usage(pipeline_handle, usage_path)
            self._render_hook_resource_usage[handle] = (
                pipeline_handle,
                usage_path,
            )
        return handle

    def remove_render_hook(self, handle):
        """Remove a custom scene draw callback."""
        removed = self.get_renderer().remove_render_hook(handle)
        if removed:
            usage = self._render_hook_resource_usage.pop(handle, None)
            if usage is not None:
                pipeline_handle, usage_path = usage
                self.resources.remove_external_usage(
                    pipeline_handle, usage_path
                )
        return removed

    def package_asset_path(self, *parts: str) -> str:
        return str(Path(_ke.__file__).resolve().parent / "assets" / Path(*parts))

    def create_standard_shaders(self, *, force: bool = False):
        """Compatibility alias for the built-in material bundle.

        Standard render paths no longer expose backend Shader objects. New code
        should call :meth:`create_standard_materials` directly.
        """
        self.shaders = self.create_standard_materials(force=force)
        return self.shaders

    def create_standard_materials(self, *, force: bool = False):
        """Create or return the cached standard material bundle.

        The bundle contains shared defaults for common scene and visualization
        work. Use create_phong_material() or create_pbr_material() when an
        object needs independently mutable surface parameters.
        """
        if self.standard_materials is not None and not force:
            return self.standard_materials

        self.standard_materials = SimpleNamespace(
            common=self.create_vertex_color_material(),
            common_texture=self.create_vertex_color_material(
                style=material_api.VertexColorStyle.TEXTURED
            ),
            ground=self.create_vertex_color_material(
                style=material_api.VertexColorStyle.CHECKERBOARD
            ),
            debug_checker=self.create_vertex_color_material(
                style=material_api.VertexColorStyle.DEBUG_CHECKER
            ),
            debug=self.create_vertex_color_material(),
            skinned=self.create_vertex_color_material(),
            skinned_texture=self.create_vertex_color_material(
                style=material_api.VertexColorStyle.TEXTURED
            ),
            skinned_debug=self.create_vertex_color_material(),
            skinned_debug_checker=self.create_vertex_color_material(
                style=material_api.VertexColorStyle.DEBUG_CHECKER
            ),
            phong=self.create_phong_material(),
            pbr=self.create_pbr_material(),
            skinned_phong=self.create_phong_material(),
            skinned_pbr=self.create_pbr_material(),
        )
        return self.standard_materials

    #################################################################

    ############# Helpers ###########################################
    def _remember_resource_handle(self, obj, handle):
        if obj is not None:
            # C++ resource entries for materials/textures/shaders are
            # intentionally non-owning. Holding the Python object here keeps
            # those objects alive and prevents stale id() reuse from returning
            # an unrelated handle.
            self._resource_handles_by_object_id[id(obj)] = (obj, handle)
            native = unwrap_native(obj)
            if native is not obj:
                self._resource_handles_by_object_id[id(native)] = (native, handle)
        return handle

    def _existing_resource_handle(self, obj):
        if obj is None:
            return None
        for candidate in (obj, unwrap_native(obj)):
            record = self._resource_handles_by_object_id.get(id(candidate))
            if record is None:
                continue
            recorded_obj, handle = record
            if recorded_obj is candidate:
                return handle
        return None

    def _next_resource_name(self, obj, prefix: str):
        self._resource_counter += 1
        return f"{prefix}_{type(obj).__name__}_{self._resource_counter}"

    def _register_material_resource(self, material):
        existing = self._existing_resource_handle(material)
        if existing is not None:
            return existing
        name = self._next_resource_name(material, "Material")
        handle = self.resources.register_material(
            name,
            unwrap_native(material),
            f"python://resource/Material/{name}",
        )
        return self._remember_resource_handle(material, handle)

    def _register_texture_resource(self, texture, display_name=None, uri=None):
        existing = self._existing_resource_handle(texture)
        if existing is not None:
            return existing
        name = (
            str(display_name)
            if display_name
            else self._next_resource_name(texture, "Texture")
        )
        handle = self.resources.register_texture(
            name,
            unwrap_native(texture),
            str(uri) if uri is not None else f"python://resource/Texture/{name}",
        )
        return self._remember_resource_handle(texture, handle)

    def _register_mesh_resource(self, mesh_data, display_name=None, uri=None):
        existing = self._existing_resource_handle(mesh_data)
        if existing is not None:
            return existing
        name = (
            str(display_name)
            if display_name
            else self._next_resource_name(mesh_data, "Mesh")
        )
        handle = self.resources.register_mesh(
            name,
            mesh_data,
            str(uri) if uri is not None else f"python://resource/Mesh/{name}",
        )
        return self._remember_resource_handle(mesh_data, handle)

    def _remember_material(self, material):
        """Keep Python-created native materials alive for renderer users."""
        self.materials.append(material)
        self._register_material_resource(material)
        return material

    def create_vertex_color_material(
        self, *, style=material_api.VertexColorStyle.UNTEXTURED
    ) -> material_api.VertexColorMaterial:
        """Create and retain a built-in vertex/display-color material."""
        return self._remember_material(material_api.VertexColorMaterial(style))

    def _remember_textures(self, *textures):
        """Keep Python-created native textures alive while materials use them."""
        for texture in textures:
            if texture is not None and texture not in self.textures:
                self.textures.append(texture)
                self._register_texture_resource(texture)

    def load_texture(self, path, *, flip: bool = True) -> render_api.Texture:
        """Load and retain a GPU texture, cached by normalized path."""
        texture_path = str(_texture_path_candidate(path).resolve())
        cached = self._textures_by_uri.get(texture_path)
        if cached is not None:
            return cached
        device = self.get_renderer().device()
        texture = device.create_texture(texture_path, bool(flip))
        self.textures.append(texture)
        self._register_texture_resource(
            texture,
            display_name=Path(texture_path).name,
            uri=texture_path,
        )
        self._textures_by_uri[texture_path] = texture
        return texture

    def create_phong_material(
        self,
        preset=None,
        *,
        ambient=None,
        diffuse=None,
        specular=None,
        shininess=None,
        diffuse_map=None,
        specular_map=None,
        alpha_map=None,
        normal_map=None,
    ) -> material_api.PhongMaterial:
        """Create and retain a Phong material instance.

        Each call returns a distinct material, so two meshes can share the same
        shader while carrying different colors/textures and batching keys.
        """
        material = material_api.PhongMaterial()
        if preset is not None:
            material.load_from_preset(preset)
        if ambient is not None:
            material.ambient = ambient
        if diffuse is not None:
            material.diffuse = diffuse
        if specular is not None:
            material.specular = specular
        if shininess is not None:
            material.shininess = float(shininess)
        if diffuse_map is not None:
            material.diffuse_map = diffuse_map
        if specular_map is not None:
            material.specular_map = specular_map
        if alpha_map is not None:
            material.alpha_map = alpha_map
        if normal_map is not None:
            material.normal_map = normal_map
        self._remember_textures(diffuse_map, specular_map, alpha_map, normal_map)
        return self._remember_material(material)

    def create_pbr_material(
        self,
        preset=None,
        *,
        base_color=None,
        metallic=None,
        roughness=None,
        emissive_color=None,
        emissive_strength=None,
        base_color_texture=None,
        normal_texture=None,
        metallic_roughness_texture=None,
        metallic_texture=None,
        roughness_texture=None,
        ao_texture=None,
        orm_texture=None,
        emissive_texture=None,
    ) -> material_api.PBRMaterial:
        """Create and retain a PBR material instance.

        Material identity is intentionally per instance: sharing one material
        shares its parameters, while separate materials can use the same shader
        with different factors/textures.
        """
        material = material_api.PBRMaterial()
        if preset is not None:
            material.load_from_preset(preset)
        if base_color is not None:
            material.base_color = base_color
        if metallic is not None:
            material.metallic = float(metallic)
        if roughness is not None:
            material.roughness = float(roughness)
        if emissive_color is not None:
            material.emissive_color = emissive_color
        if emissive_strength is not None:
            material.emissive_strength = float(emissive_strength)
        if base_color_texture is not None:
            material.base_color_texture = base_color_texture
        if normal_texture is not None:
            material.normal_texture = normal_texture
        if metallic_roughness_texture is not None:
            material.metallic_roughness_texture = metallic_roughness_texture
        if metallic_texture is not None:
            material.metallic_texture = metallic_texture
        if roughness_texture is not None:
            material.roughness_texture = roughness_texture
        if ao_texture is not None:
            material.ao_texture = ao_texture
        if orm_texture is not None:
            material.orm_texture = orm_texture
        if emissive_texture is not None:
            material.emissive_texture = emissive_texture
        self._remember_textures(
            base_color_texture,
            normal_texture,
            metallic_roughness_texture,
            metallic_texture,
            roughness_texture,
            ao_texture,
            orm_texture,
            emissive_texture,
        )
        return self._remember_material(material)

    def _obj_material_info(self, info, material_index):
        if material_index is None or int(material_index) < 0:
            return None
        materials = list(info.materials)
        index = int(material_index)
        return materials[index] if index < len(materials) else None

    def _obj_subset_alpha(self, info, subset):
        material = self._obj_material_info(info, subset.material_index)
        if material is None:
            return 1.0
        return float(material.diffuse_color[3])

    def _obj_subset_has_alpha_texture(self, info, subset):
        material = self._obj_material_info(info, subset.material_index)
        return bool(material is not None and material.has_alpha_texture)

    def _obj_subset_display_color(self, info, subset):
        material = self._obj_material_info(info, subset.material_index)
        alpha = 1.0 if material is None else float(material.diffuse_color[3])
        return _ke.Vec4(1.0, 1.0, 1.0, alpha)

    def _create_obj_subset_material(self, info, subset):
        material_info = self._obj_material_info(info, subset.material_index)
        if material_info is None:
            return self.create_phong_material(
                diffuse=_ke.Vec3(1.0, 1.0, 1.0),
                specular=_ke.Vec3(0.05, 0.05, 0.05),
                shininess=16.0,
            )

        diffuse_map = None
        specular_map = None
        alpha_map = None
        normal_map = None
        if material_info.has_diffuse_texture:
            diffuse_path = _texture_path_candidate(material_info.diffuse_texture_path)
            if diffuse_path.exists():
                diffuse_map = self.load_texture(diffuse_path, flip=True)
        if material_info.has_specular_texture:
            specular_path = _texture_path_candidate(material_info.specular_texture_path)
            if specular_path.exists():
                specular_map = self.load_texture(specular_path, flip=True)
        if material_info.has_alpha_texture:
            alpha_path = _texture_path_candidate(material_info.alpha_texture_path)
            if alpha_path.exists():
                alpha_map = self.load_texture(alpha_path, flip=True)
        if material_info.has_normal_texture:
            normal_path = _texture_path_candidate(material_info.normal_texture_path)
            if normal_path.exists():
                normal_map = self.load_texture(normal_path, flip=True)

        return self.create_phong_material(
            ambient=_ke.Vec3(
                float(material_info.ambient_color[0]),
                float(material_info.ambient_color[1]),
                float(material_info.ambient_color[2]),
            ),
            diffuse=_ke.Vec3(
                float(material_info.diffuse_color[0]),
                float(material_info.diffuse_color[1]),
                float(material_info.diffuse_color[2]),
            ),
            specular=_ke.Vec3(
                float(material_info.specular_color[0]),
                float(material_info.specular_color[1]),
                float(material_info.specular_color[2]),
            ),
            shininess=float(material_info.shininess),
            diffuse_map=diffuse_map,
            specular_map=specular_map,
            alpha_map=alpha_map,
            normal_map=normal_map,
        )

    def _add_renderable(self, material, prim, transform_source=None):
        """Low-level renderer handle path used by internal bridges."""
        if transform_source is None:
            transform_source = render_api.TransformSource.SCENE_GRAPH
        return super().add_renderable(unwrap_native(material), prim, transform_source)

    def _add_skinned_renderable(
        self,
        material,
        prim,
        skinned_mesh_data,
        transform_source=None,
    ):
        """Low-level skinned renderer handle path used by internal bridges."""
        if transform_source is None:
            transform_source = render_api.TransformSource.SCENE_GRAPH
        return super().add_skinned_renderable(
            unwrap_native(material),
            prim,
            skinned_mesh_data,
            transform_source,
        )

    def add_ground(
        self,
        path: str = "/ground",
        scale: float = 20.0,
        material=None,
    ):
        return self.scene.add_ground(path, scale, material)

    def add_mesh(
        self,
        path: str,
        mesh_data,
        material=None,
        color=None,
        *,
        uri=None,
    ):
        if material is None:
            raise ValueError("add_mesh requires a material")
        return self.scene.add_mesh(path, mesh_data, material, color=color, uri=uri)

    def add_obj(self, path: str, obj_path, **kwargs):
        return self.scene.add_obj(path, obj_path, **kwargs)

    def add_skinned_mesh(
        self,
        prim,
        material,
        skinned_mesh_data,
        transform_source=None,
    ):
        """Register a skinned mesh prim and return a RenderablePrimView.

        The material selects the built-in or custom RHI rendering path.
        """
        if transform_source is None:
            transform_source = render_api.TransformSource.SCENE_GRAPH
        self._add_skinned_renderable(
            material,
            prim,
            skinned_mesh_data,
            transform_source,
        )
        component = prim.get_render_component()
        if component is None:
            raise RuntimeError(
                f"failed to register skinned renderable prim: {prim.get_path()}"
            )
        return RenderablePrimView(self, prim, component)

    def as_vec3(self, value):
        if isinstance(value, _ke.Vec3):
            return value
        if hasattr(value, "tolist"):
            value = value.tolist()
        if len(value) != 3:
            raise ValueError("vec3 value expected 3 elements")
        return _ke.Vec3(float(value[0]), float(value[1]), float(value[2]))

    def set_camera_view(self, position, target):
        camera = self.get_camera()
        camera.set_camera_pos(self.as_vec3(position))
        camera.set_target_pos(self.as_vec3(target))
        return camera

    def set_render_hz(self, hz: float):
        return super().set_render_hz(float(hz))

    def configure_timing(self, config):
        """Apply a SimulationTimingConfig to this application loop."""
        from ..sim.timing import SimulationTimingConfig

        if not isinstance(config, SimulationTimingConfig):
            raise TypeError("config must be a SimulationTimingConfig")
        self.set_render_hz(config.render_hz)
        self.set_fixed_update_hz(config.fixed_update_hz)
        self.set_max_catch_up_steps(config.max_catch_up_steps)
        self.set_max_frame_delta(config.max_frame_delta)
        self.timing_config = config
        return config

    def configure_run(self, config):
        """Apply the rendering policy used by App-owned services."""
        from ..sim.run_mode import SimulationRunConfig

        if not isinstance(config, SimulationRunConfig):
            raise TypeError("config must be a SimulationRunConfig")
        self.run_config = config
        self._video_capture.configure(run_mode=config.mode)
        return config

    def set_video_recording_dir(self, output_dir):
        self._video_capture.configure(
            run_mode=self._recording_run_mode(), output_dir=output_dir
        )
        return self

    def set_video_recording_fps(self, fps: float):
        self._video_capture.configure(run_mode=self._recording_run_mode(), fps=fps)
        return self

    def set_video_recording_resolution(
        self, width: int | None, height: int | None = None
    ):
        """Set the recording resolution cap, or pass None for native size."""
        self._video_capture.set_max_resolution(width, height)
        return self

    def start_video_recording(self, output_path=None, fps: float | None = None):
        """Start framebuffer recording using the configured run mode."""
        self._video_capture.configure(run_mode=self._recording_run_mode())
        return self._video_capture.start(self, output_path, fps)

    def stop_video_recording(self):
        """Stop framebuffer recording and finalize the output file."""
        return self._video_capture.stop(self)

    def toggle_video_recording(self):
        """Toggle framebuffer recording, matching the Shift+T shortcut."""
        if self._video_capture.is_recording:
            return self._video_capture.stop(self)
        self._video_capture.configure(run_mode=self._recording_run_mode())
        return self._video_capture.start(self)

    def is_video_recording(self) -> bool:
        return self._video_capture.is_recording

    def get_video_recording_path(self):
        return self._video_capture.output_path

    def _on_frame_rendered_internal(self):
        """Internal native-loop callback; application subclasses should not override it."""
        self._video_capture.on_frame_rendered(self)

    def _recording_run_mode(self):
        if self.run_config is not None:
            return self.run_config.mode
        return "offscreen_fast" if self.headless else "paced"

    #################################################################

    def setup(self):
        pass

    def pre_update(self):
        pass

    def fixed_update(self, fixed_dt):
        pass

    def pre_render(self):
        pass

    def render(self):
        pass

    def post_render(self):
        pass

    def cleanup(self):
        pass

    def start(self):
        try:
            return super().start()
        finally:
            try:
                self.cleanup()
            finally:
                self._video_capture.close(self)

    def initialize(
        self,
        width=1920,
        height=1080,
        hide_ui=False,
        up_axis=None,
        graphics_backend_type=None,
        scene_backend_type=None,
        headless=False,
    ):
        self.initial_width = width
        self.initial_height = height
        self.hide_ui = hide_ui
        self.headless = headless

        self.up_axis = _ke.UpAxis.Y if up_axis is None else up_axis
        if graphics_backend_type is None:
            graphics_backend_type = render_api.BackendType.OPENGL
        if scene_backend_type is None:
            scene_backend_type = scene.BackendType.NATIVE
        self.graphics_backend_type = graphics_backend_type
        self.scene_backend_type = scene_backend_type

        result = super().initialize(
            self.initial_width,
            self.initial_height,
            self.hide_ui,
            self.up_axis,
            self.graphics_backend_type,
            self.scene_backend_type,
            self.headless,
        )
        # Native initialize() recreates the SceneBackend. Keep the resource
        # registry mirrored into the live scene used by ScenePanel/rendering.
        self.resources.bind_scene(self.get_native_scene())
        if self.run_config is None:
            from ..sim.run_mode import SimulationRunConfig, SimulationRunMode

            mode = (
                SimulationRunMode.OFFSCREEN_FAST
                if self.headless
                else SimulationRunMode.PACED
            )
            self.configure_run(SimulationRunConfig(mode=mode))
        return result

    def is_key_down(self, key):
        return self.is_key_pressed(key)

    def was_key_pressed(self, key):
        down = self.is_key_pressed(key)
        was_down = self._previous_key_states.get(key, False)
        self._previous_key_states[key] = down
        return down and not was_down

    def was_key_released(self, key):
        down = self.is_key_pressed(key)
        was_down = self._previous_key_states.get(key, False)
        self._previous_key_states[key] = down
        return not down and was_down

    def should_close_shortcut_pressed(self):
        return self.is_key_pressed(keys.ESCAPE)
