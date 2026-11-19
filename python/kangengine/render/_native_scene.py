"""Native renderer resources for evaluated static scene inputs."""

from ..app import App
from .. import scene


class NativeSceneAdapter:
    def __init__(self, width, height):
        self._app = App()
        try:
            self._app.initialize(width=width, height=height, hide_ui=True, headless=True)
            self._app.set_frame_capture_active(True)
            self._camera = self._app.scene.define_prim("/camera", scene.PrimType.CAMERA)
            self._camera.add_camera_component()
            self._app.set_active_scene_camera(self._camera)
        except BaseException:
            self._app.close()
            raise

    def render(self, frame):
        self._app.get_renderer()._sync_scene(frame)
        camera = frame["camera"]
        self._camera.set_world_matrix(camera["world"].T.copy())
        self._camera.get_camera_component().set_perspective(
            camera["vertical_fov"], camera["near"], camera["far"])
        light = frame["lights"][0] if frame["lights"] else None
        self._app.set_light_intensity(light["intensity"] if light else 0)
        self._app.set_light_ambient([0, 0, 0])
        if light:
            self._app.set_light_direction(light["direction"])
            self._app.set_light_color(light["color"])
        self._app.draw_frame()
        return self._app.read_rgb_pixels()

    def close(self):
        self._app.close()
