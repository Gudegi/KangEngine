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
            preset = _ke.ColorLibrary.get(_ke.ColorType.PASTEL_GREEN)
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
        if shader is None:
            shader = self.create_asset_shader("common.vs", "checkerboard.fs")
            self.configure_checker_shader(shader)
        ground = self.get_scene().define_prim(path, scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(float(scale), self.up_axis))
        handle = self.add_renderable(shader, ground)
        return ground, handle

    def add_mesh(self, path: str, mesh_data, shader, color=None):
        prim = self.get_scene().define_prim(path, scene.PrimType.Mesh)
        prim.set_mesh_data(mesh_data)
        if color is not None:
            prim.set_display_color_alpha(color)
        handle = self.add_renderable(shader, prim)
        return prim, handle

    def set_shape_textures(self, handle, diffuse=None, normal=None):
        if diffuse is not None:
            self.set_renderable_texture(
                handle,
                diffuse,
                _ke.TextureRole.BaseColor,
            )
        if normal is not None:
            self.set_renderable_texture(handle, normal, _ke.TextureRole.Normal)
        return handle

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
        self.set_render_hz(float(hz))
    
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
