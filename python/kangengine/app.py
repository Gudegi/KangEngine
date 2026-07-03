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

    def __init__(self, app: "App", prim, handles):
        self._app = app
        self.prim = prim
        self._handles = tuple(int(handle) for handle in handles)
        self._external_buffers = {}

    @property
    def path(self) -> str:
        return self.prim.get_path()

    def set_visible(self, visible: bool):
        self.prim.set_visible(bool(visible))
        return self

    def set_double_sided(self, enabled: bool = True):
        for handle in self._handles:
            self._app.set_renderable_double_sided(handle, bool(enabled))
        return self

    def set_casts_shadow(self, enabled: bool = True):
        for handle in self._handles:
            self._app.set_renderable_casts_shadow(handle, bool(enabled))
        return self

    def set_alpha_mode(self, mode, cutoff: float = 0.5):
        """Choose opaque, cutout-mask, or blended alpha rendering."""
        for handle in self._handles:
            self._app.set_renderable_alpha_mode(handle, mode, float(cutoff))
        return self

    def set_texture(self, texture, role_or_slot=_ke.TextureRole.BaseColor):
        for handle in self._handles:
            self._app.set_renderable_texture(handle, texture, role_or_slot)
        return self

    def set_transform_buffer(
        self,
        transforms,
        *,
        sim_device=None,
        sync_policy=None,
    ):
        """Use a float32 column-major ``[N, 4, 4]`` transform buffer."""
        from .utils.sim_buffer import to_external_transform_desc

        descriptor, buffer = to_external_transform_desc(
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

        renderer = self._app.get_renderer()
        for handle in self._handles:
            renderer.set_renderable_external_buffer(handle, descriptor)
            self._external_buffers[handle] = (buffer, descriptor)
        return self

    def remove(self):
        return self._app.remove_prim(self.prim)


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
        if transform_source is None:
            transform_source = _ke.TransformSource.SceneGraph
        handle = self._app.add_renderable(material_or_shader, prim, transform_source)
        return RenderablePrimView(self._app, prim, [handle])

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
