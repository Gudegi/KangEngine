"""Check real window clicks, surface hits, and occluded selection outlines."""

import ctypes
from pathlib import Path
import sys

import numpy as np
import kangengine as ke
from kangengine._core import _ke

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinning_backend_smoke import cube_asset


class PickApp(ke.App):
    picked = None

    def on_ray_picked(self, result):
        self.picked = result


def main():
    app = PickApp()
    app.initialize(width=320, height=240, hide_ui=True, renderer="ovrtx")
    app.set_vsync(False)
    try:
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        app.set_light_ambient([0, 0, 0])
        material = app.create_pbr_material(base_color=[0.1, 0.8, 0.2, 1])
        front = app.scene.add_mesh("/front", ke.geometry.create_cube_data(1.5), material)
        back = app.scene.add_mesh("/back", ke.geometry.create_cube_data(0.8), material)
        back.prim.set_local_translation(ke.Vec3(0, 0, -2))

        def capture():
            for _ in range(12):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        capture()
        glfw = ctypes.CDLL(_ke.__file__)
        glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
        window = glfw.glfwGetCurrentContext()
        mouse_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        cursor_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_double, ctypes.c_double)
        glfw.glfwSetMouseButtonCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        glfw.glfwSetMouseButtonCallback.restype = ctypes.c_void_p
        glfw.glfwSetCursorPosCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        glfw.glfwSetCursorPosCallback.restype = ctypes.c_void_p
        mouse = glfw.glfwSetMouseButtonCallback(window, None)
        cursor = glfw.glfwSetCursorPosCallback(window, None)
        glfw.glfwSetMouseButtonCallback(window, mouse)
        glfw.glfwSetCursorPosCallback(window, cursor)

        def click():
            app.picked = None
            cursor_type(cursor)(window, 160, 120)
            mouse_type(mouse)(window, 0, 1, 0)
            mouse_type(mouse)(window, 0, 0, 0)
            capture()
            assert app.picked is not None
            return app.picked

        click()
        assert app.picked is not None and app.picked.hit
        assert app.picked.prim.get_path() == "/front"
        assert abs(app.picked.position.z - 0.75) < 0.02
        front.prim.set_pickable(False)
        assert click().prim.get_path() == "/back"
        front.prim.set_pickable(True)
        front.prim.set_display_color_alpha(ke.Vec4(1, 1, 1, 0))
        front.set_alpha_mode(ke.render.AlphaMode.MASK)
        capture()
        assert click().prim.get_path() == "/back"
        front.prim.set_display_color_alpha(ke.Vec4(1, 1, 1, 1))
        front.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
        capture()
        app.select_prim(back.prim)
        outlined = capture()
        red, green, blue = outlined.transpose(2, 0, 1)
        assert np.count_nonzero((red > 200) & (green > 80) & (blue < 30)) > 30
        back.remove()
        front.remove()
        capture()
        batch = app.scene.add_mesh("/batch", ke.geometry.create_cube_data(0.6), material,
                                   transform_source=ke.render.TransformSource.EXTERNAL_BUFFER)
        matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        matrices[0, 3, 0] = 1.5
        app.get_scene_render_system().update_instances(batch.component, matrices)
        capture()
        picked = click()
        assert picked.hit and picked.prim.get_path() == "/batch" and picked.instance_index == 1
        assert picked.transform_source == ke.render.TransformSource.EXTERNAL_BUFFER
        batch.remove()
        skin = ke.visual.SkinnedSurface.create(app, "/skin", cube_asset(), material=material)
        capture()
        assert click().prim.get_path() == "/skin"
        joints = np.eye(4, dtype=np.float32)[None]
        joints[0, 0, 3] = 1.5
        skin.update_joint_matrices(joints)
        capture()
        assert not click().hit
        skin.remove()
        capture()
        print("PASS: window click, surface hit, occluded outline, instance identity, posed skin, and removal")
    finally:
        app.close()


if __name__ == "__main__":
    main()
