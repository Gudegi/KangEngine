"""Debug geometry, overlay, and text in one scene."""

import math

import numpy as np

import kangengine as ke


class DebugVisualizationApp(ke.App):
    def setup(self):
        self.count = 0
        self.orbit_angle = 0.0
        materials = self.create_standard_materials()
        self.scene.add_ground("/ground", scale=20.0)

        self.box = self.scene.add_mesh(
            "/box",
            ke.geometry.create_cube_data(1.0),
            materials.common,
            color=ke.vec4(0.8, 0.3, 0.05, 1.0),
        )
        self.box.set_local_translation(ke.vec3(0.0, 0.5, 0.0))

        # Lightweight overlay: no SceneGraph prims are created.
        self.debug_overlay.lines(
            "/overlay/lines",
            np.array([[-3.0, 0.05, -1.0], [-3.0, 0.05, 0.0]]),
            np.array([[-1.5, 1.0, -1.0], [-1.5, 1.0, 0.0]]),
            np.array(
                [
                    [1.0, 0.25, 0.2, 1.0],
                    [0.2, 0.9, 0.35, 1.0],
                ]
            ),
            width=3.0,
        )
        self.debug_overlay.points(
            "/overlay/points",
            np.array([[-3.0, 0.1, 1.0], [-2.5, 0.5, 1.0], [-2.0, 0.9, 1.0]]),
            np.array(
                [
                    [1.0, 0.3, 0.3, 1.0],
                    [0.3, 1.0, 0.3, 1.0],
                    [0.3, 0.5, 1.0, 1.0],
                ]
            ),
            size=10.0,
        )
        self.debug_overlay.axes(
            "/overlay/axes",
            origin=np.array([-2.0, 0.05, 2.0]),
            rotation=np.eye(3),
            length=1.0,
            width=3.0,
        )
        # Mesh-based debug geometry: visible as ordinary SceneGraph objects.
        self.scene.debug_geometry.add_lines(
            "/debug_geometry/lines",
            np.array([[1.5, 0.05, -1.0], [1.5, 0.05, 0.0]]),
            np.array([[3.0, 1.0, -1.0], [3.0, 1.0, 0.0]]),
            np.array(
                [
                    [1.0, 0.6, 0.15, 1.0],
                    [0.2, 0.75, 1.0, 1.0],
                ]
            ),
            material=materials.debug,
            radius=0.025,
        )
        self.scene.debug_geometry.add_arrows(
            "/debug_geometry/arrows",
            np.array([[1.5, 0.05, 1.0], [1.5, 0.05, 2.0]]),
            np.array([[3.0, 1.0, 1.0], [3.0, 1.0, 2.0]]),
            np.array(
                [
                    [0.95, 0.35, 0.75, 1.0],
                    [0.45, 0.95, 0.35, 1.0],
                ]
            ),
            material=materials.debug,
            radius=0.04,
        )
        self.scene.debug_geometry.add_axes(
            "/debug_geometry/axes",
            origin=np.array([2.0, 0.05, 3.0]),
            rotation=np.eye(3),
            material=materials.debug,
            length=1.0,
            radius=0.02,
        )
        self.debug_sphere_centers = np.array(
            [
                [0.5, 0.4, 4.5],
                [1.5, 0.6, 4.5],
                [2.9, 0.8, 4.5],
            ],
            dtype=np.float32,
        )
        self.debug_sphere_radii = np.array([0.35, 0.55, 0.75])
        self.debug_sphere_colors = np.array(
            [
                [1.0, 0.35, 0.25, 1.0],
                [0.3, 0.8, 1.0, 1.0],
                [0.5, 1.0, 0.35, 1.0],
            ]
        )
        self.debug_spheres = self.scene.debug_geometry.add_spheres(
            "/debug_geometry/spheres",
            centers=self.debug_sphere_centers,
            radii=self.debug_sphere_radii,
            colors=self.debug_sphere_colors,
            material=materials.debug,
        )

        self.world_text.set(
            "/labels/overlay",
            "Debug overlay",
            ke.vec3(-2.3, 1.5, 0.0),
            pixel_size=24.0,
            depth_test=False,
        )
        self.world_text.set(
            "/labels/geometry",
            "Scene debug geometry",
            ke.vec3(2.3, 1.5, 0.0),
            pixel_size=24.0,
        )
        self.world_text.set(
            "/labels/box",
            "Moving box",
            ke.vec3(0.0, 1.7, 0.0),
            pixel_size=24.0,
        )
        self.screen_text.set(
            "/screen/title",
            "Debug Visualization {}".format(self.count),
            ke.vec2(30.0, 30.0),
            pixel_size=28.0,
        )
        self.screen_text.set(
            "/screen/anchor",
            "TopCenter",
            ke.vec2(30.0, 30.0),
            alignment=ke.render.TextAlignment.Center,
            anchor=ke.render.ScreenAnchor.TopCenter,
            color=ke.vec4(0.3, 0.9, 1.0, 1.0),
            pixel_size=25.0,
        )
        self.screen_text.set(
            "/screen/anchor2",
            "CenterLeft",
            ke.vec2(0.0, 0.0),
            anchor=ke.render.ScreenAnchor.CenterLeft,
            color=ke.vec4(0.3, 0.9, 1.0, 1.0),
            pixel_size=25.0,
        )

        # self.set_camera_view([7.0, -9.0, 6.0], [0.0, 0.0, 0.8])

    def preRender(self):
        self.orbit_angle = (
            self.orbit_angle + math.radians(120.0) * self.get_delta_time()
        ) % (2.0 * math.pi)
        self.box.set_local_translation(
            ke.vec3(
                math.cos(self.orbit_angle),
                0.5,
                math.sin(self.orbit_angle),
            )
        )

        box_pos = self.box.get_world_translation()
        box_quat = self.box.get_world_rotation()
        self.world_text.set_position(
            "/labels/box",
            box_pos + box_quat * ke.vec3(0.0, 1.2, 0.0),
        )

        sphere_centers = self.debug_sphere_centers.copy()
        sphere_centers[:, 1] += 0.1 * np.sin(
            self.orbit_angle + np.arange(len(sphere_centers))
        )
        self.debug_spheres.update_spheres(
            sphere_centers,
            self.debug_sphere_radii,
            self.debug_sphere_colors,
        )

        self.screen_text.set_text(
            "/screen/title",
            "Debug Visualization {}".format(self.count),
        )
        self.count += 1


if __name__ == "__main__":
    app = DebugVisualizationApp()
    app.initialize(1920, 1080, False, ke.UpAxis.Y)
    app.start()
