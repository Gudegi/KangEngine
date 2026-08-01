"""Display the reusable gamepad input visualizer."""

import numpy as np
import kangengine as ke


class GamepadVisualizerApp(ke.App):
    def setup(self):
        self.orbit_camera = True
        self.attach_camera = True
        self.gamepad = self.input.gamepad()
        self.gamepad_visualizer = ke.input.GamepadVisualizer(
            self, width=400, anchor=ke.render.ScreenAnchor.BottomCenter
        )
        self.gamepad_state = self.gamepad.state()
        self.standard_materials = self.create_standard_materials()
        self.scene.add_ground("/ground", scale=20.0)

        self.sphere = self.scene.add_mesh(
            "/controlled_sphere",
            ke.geometry.create_sphere_data(0.5, 32, 16),
            self.standard_materials.common,
            color=ke.vec4(0.25, 0.72, 0.95, 1.0),
        )
        self.sphere_pos = np.array([0.0, -0.3, 0.7])
        self.previous_sphere_pos = self.sphere_pos.copy()
        self.sphere.set_local_translation(self.sphere_pos)

        self.scene.debug_geometry.add_axes(
            "/debug_geometry/axes",
            origin=np.array([0.0, 0.0, 0.01]),
            rotation=np.eye(3),
            length=1.0,
            radius=0.05,
        )
        if self.attach_camera:
            offset = (
                np.array([5.0, 0.0, 2.0])
                if self.up_axis == ke.UpAxis.Z
                else np.array([5.0, 2.0, 0.0])
            )
            camera = self.get_camera()
            camera.set_camera_pos(self.sphere_pos + offset)
            camera.set_target_pos(self.sphere_pos)

    def preRender(self):
        self.gamepad_state = self.gamepad.state()
        joystick = self.gamepad.get_left_joystick(
            camera_relative=True,
            state=self.gamepad_state,
        )
        if joystick is not None:
            direction, strength = joystick
            distance = 3.0 * strength * min(self.get_delta_time(), 0.1)
            self.sphere_pos += direction * distance
            self.sphere.set_local_translation(self.sphere_pos)

        if self.attach_camera:
            delta = self.sphere_pos - self.previous_sphere_pos
            camera = self.get_camera()
            camera.set_camera_pos(camera.get_camera_pos() + delta)
            camera.set_target_pos(camera.get_target_pos() + delta)
        self.previous_sphere_pos = self.sphere_pos.copy()

        look = self.gamepad.get_right_joystick(
            orbit=self.orbit_camera,
            state=self.gamepad_state,
        )
        if look is not None:
            yaw_pitch, strength = look
            self._rotate_camera(yaw_pitch * strength)

    def _rotate(self, vector, axis, angle):
        axis = axis / np.linalg.norm(axis)
        return (
            vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
        )

    def _rotate_camera(self, yaw_pitch):
        camera = self.get_camera()
        camera_pos = np.array(camera.get_camera_pos(), dtype=float)
        target_pos = np.array(camera.get_target_pos(), dtype=float)
        up = (
            np.array([0.0, 0.0, 1.0])
            if self.up_axis == ke.UpAxis.Z
            else np.array([0.0, 1.0, 0.0])
        )
        offset = (
            camera_pos - target_pos if self.orbit_camera else target_pos - camera_pos
        )
        angle_scale = np.deg2rad(90.0) * min(self.get_delta_time(), 0.1)
        offset = self._rotate(offset, up, -yaw_pitch[0] * angle_scale)
        right = np.cross(offset, up)
        right /= np.linalg.norm(right)
        pitched = self._rotate(offset, right, yaw_pitch[1] * angle_scale)
        if abs(np.dot(pitched / np.linalg.norm(pitched), up)) < 0.98:
            offset = pitched

        if self.orbit_camera:
            camera.set_camera_pos(target_pos + offset)
        else:
            camera.set_target_pos(camera_pos + offset)

    def render(self):
        self.gamepad_visualizer.draw(self.gamepad_state)
        if ke.imgui.begin("Gamepad Camera"):
            _, self.orbit_camera = ke.imgui.checkbox("Orbit camera", self.orbit_camera)
            _, self.attach_camera = ke.imgui.checkbox(
                "Attach camera", self.attach_camera
            )
        ke.imgui.end()


if __name__ == "__main__":
    app = GamepadVisualizerApp()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
