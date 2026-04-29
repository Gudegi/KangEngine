"""Python-friendly base App for KangEngine examples and tools.

The native C++ App still owns the main loop, camera controls, renderer, ImGui
frame lifecycle, and scene rendering.  This wrapper adds default lifecycle
hooks and small input helpers so examples can inherit from `kangengine.App`
without talking directly to the pybind class.
"""

from . import _kangengine as _ke

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
    ):
        if up_axis is None:
            up_axis = _ke.UpAxis.Y
        if graphics_backend_type is None:
            graphics_backend_type = _ke.BackendType.OpenGL
        if scene_backend_type is None:
            scene_backend_type = scene.BackendType.Native

        return super().initialize(
            width,
            height,
            hide_ui,
            up_axis,
            graphics_backend_type,
            scene_backend_type,
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
