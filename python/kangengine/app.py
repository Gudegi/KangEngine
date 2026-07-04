"""Python-friendly base App for KangEngine examples and tools.

The native C++ App still owns the main loop, camera controls, renderer, ImGui
frame lifecycle, and scene rendering.  This wrapper adds default lifecycle
hooks and small input helpers so examples can inherit from `kangengine.App`
without talking directly to the pybind class.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ._core import _ke

keys = _ke.keys
scene = _ke.scene
NativeApp = _ke.App


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

    def set_texture(self, texture, role_or_slot=_ke.TextureRole.BaseColor):
        self._render_system.set_texture(self.component, texture, role_or_slot)
        return self

    def set_transform_buffer(
        self,
        transforms,
        *,
        sim_device=None,
        sync_policy=None,
    ):
        """Set the ``[N, 4, 4]`` buffer of an ExternalBuffer renderable."""
        from .utils.sim_buffer import to_external_transform_desc

        descriptor, _ = to_external_transform_desc(
            transforms,
            sim_device=sim_device,
            dtype="float32",
            name=f"{self.path}:transforms",
            sync_policy=(
                _ke.ExternalSyncPolicy.NONE
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


class SceneContext:
    """Scene-facing facade connected to the owning App renderer.

    Use this for common add/remove workflows. It keeps renderer handles inside
    the app-facing layer while preserving access to the underlying scene backend
    for lower-level operations.
    """

    def __init__(self, app: "App"):
        self._app = app

    @property
    def backend(self):
        return self._app.get_scene()

    def define_prim(self, path: str, prim_type):
        return self.backend.define_prim(path, prim_type)

    def get_prim_at_path(self, path: str):
        return self.backend.get_prim_at_path(path)

    def get_root_prim(self):
        return self.backend.get_root_prim()

    def add_renderable(
        self,
        prim,
        material_or_shader,
        transform_source=None,
    ):
        """Register a scene prim as renderable through RenderComponent.

        This is the preferred public path for authored scene objects. It
        returns a RenderablePrimView facade instead of exposing the native
        renderer handle.
        """
        if transform_source is None:
            transform_source = _ke.TransformSource.SceneGraph
        self._app.add_renderable(material_or_shader, prim, transform_source)
        component = prim.get_render_component()
        if component is None:
            raise RuntimeError(f"failed to register renderable prim: {prim.get_path()}")
        return RenderablePrimView(self._app, prim, component)

    def add_mesh(
        self,
        path: str,
        mesh_data,
        material_or_shader,
        color=None,
        transform_source=None,
    ):
        prim = self.define_prim(path, scene.PrimType.Mesh)
        prim.set_mesh_data(mesh_data)
        if color is not None:
            prim.set_display_color_alpha(color)
        return self.add_renderable(prim, material_or_shader, transform_source)

    def add_ground(self, path: str = "/ground", scale: float = 20.0, shader=None):
        if shader is None:
            shader = self._app.create_asset_shader("common.vs", "checkerboard.fs")
            self._app.configure_checker_shader(shader)
        return self.add_mesh(
            path,
            scene.Prim.create_plane_data(float(scale), self._app.up_axis),
            shader,
        )

    def log_lines(
        self,
        path: str,
        shader,
        starts,
        ends,
        colors=None,
        radius: float = 0.005,
        segments: int = 8,
    ):
        component = scene.DebugDraw.log_component_lines(
            self._app,
            shader,
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
        shader,
        starts,
        ends,
        colors=None,
        radius: float = 0.02,
        segments: int = 12,
    ):
        component = scene.DebugDraw.log_component_arrows(
            self._app,
            shader,
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


class App(NativeApp):
    """Base class for Python KangEngine apps.

    This mirrors the C++ App lifecycle:

    - `setup()` runs once before the loop starts.
    - `preRender()` runs before scene rendering each frame.
    - `render()` runs while the ImGui frame is active.
    - `postRender()` runs after UI rendering and buffer swap setup.

    The C++ implementation still handles the default camera controls:
    WASD/mouse navigation, H to hide UI, Escape to close, framebuffer updates,
    and the built-in scene/performance panels.
    """

    def __init__(self):
        super().__init__()
        self._previous_key_states = {}
        self.width = 1920
        self.height = 1080
        self.hide_ui = False
        self.up_axis = _ke.UpAxis.Y
        self.graphics_backend_type = _ke.BackendType.OpenGL
        self.scene_backend_type = scene.BackendType.Native
        self.headless = False
        self.scene = SceneContext(self)

    def package_asset_path(self, *parts: str) -> str:
        return str(Path(_ke.__file__).resolve().parent / "assets" / Path(*parts))

    ############# Shaders ###########################################
    def _bind_common_ubos(self, *shaders):
        for shader in shaders:
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)
            shader.set_uniform_block_binding("shadowUBO", 2)
        return shaders[0] if len(shaders) == 1 else shaders

    def create_asset_shader(self, vertex_shader: str, fragment_shader: str):
        shader = self.get_renderer().device().create_shader_from_file(
            self.package_asset_path("shaders", vertex_shader),
            self.package_asset_path("shaders", fragment_shader),
        )
        self._bind_common_ubos(shader)
        return shader

    def set_texture_uniform(self, shader, unit: int = 0, name: str = "uTexture"):
        shader.use()
        shader.set_int(name, int(unit))
        return shader

    def configure_checker_shader(
        self,
        shader,
        color1=None,
        color2=None,
    ):
        if color1 is None:
            color1 = _ke.vec4(1.0, 1.0, 1.0, 1.0)
        if color2 is None:
            preset = _ke.ColorLibrary.get(_ke.ColorType.DARK_GREEN)
            color2 = _ke.vec4(preset.r, preset.g, preset.b, preset.a)
        shader.use()
        shader.set_vec4("checkerColor1", color1)
        shader.set_vec4("checkerColor2", color2)
        return shader

    def create_standard_shaders(self):
        shaders = SimpleNamespace(
            common=self.create_asset_shader("common.vs", "common.fs"),
            common_texture=self.create_asset_shader("common.vs", "commonTex.fs"),
            common_debug=self.create_asset_shader("common.vs", "debug_checker.fs"),
            skinned=self.create_asset_shader("skinned_mesh.vs", "common.fs"),
            skinned_texture=self.create_asset_shader(
                "skinned_mesh.vs",
                "commonTex.fs",
            ),
            skinned_debug=self.create_asset_shader(
                "skinned_mesh.vs",
                "debug_checker.fs",
            ),
            ground=self.create_asset_shader("common.vs", "checkerboard.fs"),
        )
        self.set_texture_uniform(shaders.common_texture)
        self.set_texture_uniform(shaders.skinned_texture)
        self.configure_checker_shader(shaders.ground)
        return shaders

    #################################################################

    ############# Helpers ###########################################
    def add_ground(self, path: str = "/ground", scale: float = 20.0, shader=None):
        return self.scene.add_ground(path, scale, shader)

    def add_mesh(self, path: str, mesh_data, shader, color=None):
        return self.scene.add_mesh(path, mesh_data, shader, color=color)

    def as_vec3(self, value):
        if isinstance(value, _ke.vec3):
            return value
        if hasattr(value, "tolist"):
            value = value.tolist()
        if len(value) != 3:
            raise ValueError("vec3 value expected 3 elements")
        return _ke.vec3(float(value[0]), float(value[1]), float(value[2]))

    def set_camera_view(self, position, target):
        camera = self.get_camera()
        camera.set_camera_pos(self.as_vec3(position))
        camera.set_target_pos(self.as_vec3(target))
        return camera

    def set_render_hz(self, hz: float):
        return super().set_render_hz(float(hz))
    
    #################################################################

    def setup(self):
        pass

    def preRender(self):
        pass

    def render(self):
        pass

    def postRender(self):
        pass

    def cleanup(self):
        pass

    def start(self):
        try:
            return super().start()
        finally:
            self.cleanup()

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
            graphics_backend_type = _ke.BackendType.OpenGL
        if scene_backend_type is None:
            scene_backend_type = scene.BackendType.Native
        self.graphics_backend_type = graphics_backend_type
        self.scene_backend_type = scene_backend_type

        return super().initialize(
            self.initial_width,
            self.initial_height,
            self.hide_ui,
            self.up_axis,
            self.graphics_backend_type,
            self.scene_backend_type,
            self.headless,
        )

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
