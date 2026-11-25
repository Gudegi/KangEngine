"""Measure scene surfaces with a world-space ray; one scene unit is one meter."""

import math

import numpy as np

import kangengine as ke
from kangengine.utils._example_args import render_example_parser
from kangengine import imgui


MISS_LENGTH = 9.0


def rgba(preset):
    color = ke.ColorLibrary.get(preset)
    return (color.r, color.g, color.b, color.a)


class RaycastDistanceApp(ke.App):
    def setup(self):
        self.origin_x = -4.0
        self.heading = 0.0
        self.height = 1.0
        self.distance = None
        self.target_path = ""
        # self.set_ui_layout_mode(mode=ke.UILayoutMode.OVERLAY)
        self.set_interaction_mode(mode=ke.InteractionMode.INSPECT)
        materials = self.create_standard_materials()
        ground = self.scene.add_ground(path="/ground", scale=18.0)
        ground.prim.set_pickable(pickable=False)

        box = self.scene.add_mesh(
            path="/box",
            mesh_data=ke.geometry.create_box_data(
                x_scale=1.5,
                y_scale=1.5,
                z_scale=2.0,
            ),
            material=materials.common,
            color=ke.Vec4(*rgba(ke.ColorType.PASTEL_BLUE)),
        )
        box.set_local_translation(ke.Vec3(0.0, 0.0, 1.0))
        sphere = self.scene.add_mesh(
            path="/sphere",
            mesh_data=ke.geometry.create_sphere_data(
                radius=1.0,
                num_longitudes=48,
                num_latitudes=24,
            ),
            material=materials.common,
            color=ke.Vec4(*rgba(ke.ColorType.PASTEL_ORANGE)),
        )
        sphere.set_local_translation(ke.Vec3(2.5, 2.5, 1.0))
        self.set_camera_view(position=(-8.0, -10.0, 8.0), target=(0.0, 0.7, 0.6))

        self.start_color = rgba(ke.ColorType.CYAN)
        self.hit_color = rgba(ke.ColorType.YELLOW)
        self.miss_color = rgba(ke.ColorType.ORANGE)

    def pre_render(self):
        angle = math.radians(self.heading)
        origin = np.array([self.origin_x, 0.0, self.height], dtype=np.float32)
        direction = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)
        hit = self.raycast(origin=ke.Vec3(*origin), direction=ke.Vec3(*direction))
        self.distance = hit.distance if hit.hit else None
        self.target_path = hit.prim.get_path() if hit.hit else ""

        # Miss length is only a display limit, not a raycast hit distance.
        length = hit.distance if hit.hit else MISS_LENGTH
        end = (
            np.asarray(hit.position, dtype=np.float32)
            if hit.hit
            else origin + direction * length
        )
        color = self.hit_color if hit.hit else self.miss_color
        self.debug.overlay.lines(
            path="/ray/segment",
            starts=origin[None],
            ends=end[None],
            colors=color,
            width=3.0,
        )
        self.debug.overlay.points(
            path="/ray/endpoints",
            points=np.stack([origin, end]),
            colors=np.array([self.start_color, color]),
            size=12.0,
            overlay=True,
        )

        # One-meter marks make the numeric distance visible along the ray.
        ticks = origin + np.arange(1.0, length)[:, None] * direction
        side = np.array([-direction[1], direction[0], 0.0]) * 0.1
        self.debug.overlay.lines(
            path="/ray/ticks",
            starts=ticks - side,
            ends=ticks + side,
            colors=color,
            width=2.0,
        )
        label = f"{length:.3f} m" if hit.hit else "No hit"
        self.debug.world_text.set(
            path="/ray/distance",
            text=label,
            position=ke.Vec3(*((origin + end) * 0.5 + [0.0, 0.0, 0.3])),
            pixel_size=28.0,
            depth_test=False,
        )
        self.debug.world_text.set(
            path="/ray/origin",
            text="Origin",
            position=ke.Vec3(*(origin + [0.0, 0.0, 0.35])),
            pixel_size=20.0,
            depth_test=False,
        )

    def render(self):
        # imgui.set_next_window_pos(x=20.0, y=20.0)
        # imgui.set_next_window_size(width=380.0, height=245.0)
        imgui.begin("Raycast distance")
        _, self.origin_x = imgui.slider_float("Origin X (m)", self.origin_x, -6.0, -2.0)
        _, self.heading = imgui.slider_float("Heading (deg)", self.heading, -40.0, 50.0)
        _, self.height = imgui.slider_float("Height (m)", self.height, 0.2, 3.0)
        if self.distance is not None:
            imgui.text(f"Surface distance: {self.distance:.3f} m")
            imgui.text(f"Hit: {self.target_path}")
        else:
            imgui.text(f"No hit; showing {MISS_LENGTH:g} m of the ray.")
        imgui.text("Cyan: origin. Yellow: hit. Orange: miss.")
        imgui.text("Each tick is 1 m. Default box distance: 3.250 m.")
        if imgui.button("Reset ray"):
            self.origin_x, self.heading, self.height = -4.0, 0.0, 1.0
        imgui.end()


def main():
    args = render_example_parser(description=__doc__).parse_args()
    app: RaycastDistanceApp = RaycastDistanceApp()
    app.initialize(width=1280, height=900, up_axis=ke.UpAxis.Z, renderer=args.renderer)
    app.start()


if __name__ == "__main__":
    main()
