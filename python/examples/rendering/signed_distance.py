"""Move a sphere through a closed wall and inspect native BVH signed distances.

Run from the repository root: python python/examples/rendering/signed_distance.py
Coordinates are Y-up and measured in meters. No physics world or simulation is
used. The sphere test uses d(center) - radius; arbitrary robot shapes still need
their own surface-overlap tests in addition to center-point containment.
"""

import numpy as np

import kangengine as ke
from kangengine import imgui
from kangengine.utils._example_args import render_example_parser


WALL_CENTER = np.array([0.0, 1.4, 0.0])
WALL_HALF_EXTENTS = np.array([1.0, 1.2, 0.4])
PRESETS = {
    "Outside": [0.3, 1.4, 1.0],
    "Surface overlap": [0.3, 1.4, 0.55],
    "Fully inside": [0.3, 1.4, 0.1],
}
SLICE_PLANES = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}
SAMPLE_MIN = np.array([-1.5, 0.0, -1.2])
SAMPLE_MAX = np.array([1.5, 2.8, 1.2])
SLICE_RESOLUTION = 41


def wall_mesh():
    # Shared vertices keep the distance mesh topologically closed; render-box
    # generators may split vertices at hard normal seams.
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return corners * WALL_HALF_EXTENTS + WALL_CENTER, faces


def rgba(preset):
    color = ke.ColorLibrary.get(preset)
    return (color.r, color.g, color.b, color.a)


def distance_colors(distances, color_range):
    """Map negative/zero/positive distances to red/white/blue, clamped in meters."""
    normalized = np.clip(np.asarray(distances) / color_range, -1.0, 1.0)
    colors = np.ones(normalized.shape + (4,), dtype=np.float32)
    colors[..., 0] -= np.maximum(normalized, 0)
    colors[..., 1] -= np.abs(normalized)
    colors[..., 2] += np.minimum(normalized, 0)
    return colors


class SignedDistanceApp(ke.App):
    def setup(self):
        self.center = np.array(PRESETS["Surface overlap"])
        self.radius = 0.25
        self.show_solid = False
        self.show_slice = True
        self.slice_plane = "XZ"
        self.slice_offset = 1.4
        self.color_range = 0.5
        self.show_sample = False
        self.sample_idx = SLICE_RESOLUTION * 30 + 32
        self._last_query = None
        self._last_slice = None
        self._last_display = None
        self.query_count = 0
        self.slice_query_count = 0
        self._panel_initialized = False
        self.red = rgba(ke.ColorType.RED)
        self.green = rgba(ke.ColorType.GREEN)
        self.yellow = rgba(ke.ColorType.YELLOW)
        self.cyan = rgba(ke.ColorType.CYAN)
        self.magenta = rgba(ke.ColorType.MAGENTA)
        self.set_ui_layout_mode(mode=ke.UILayoutMode.OVERLAY)
        self.set_interaction_mode(mode=ke.InteractionMode.INSPECT)
        self.set_background_color(color=rgba(ke.ColorType.BLACK))

        vertices, faces = wall_mesh()
        self.sdf = ke.geometry.SignedDistanceMesh(vertices, faces)
        edges = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)
                          if np.count_nonzero(vertices[a] != vertices[b]) == 1])
        self.debug.overlay.lines(
            path="/sdf/wall", starts=vertices[edges[:, 0]], ends=vertices[edges[:, 1]],
            colors=self.cyan, width=2.0, overlay=True,
        )
        material = self.create_color_material(style=ke.material.ColorStyle.UNLIT)
        self.wall_visual = self.scene.add_mesh(
            path="/wall", mesh_data=ke.geometry.create_box_data(*(2 * WALL_HALF_EXTENTS)),
            material=material, color=ke.Vec4(*rgba(ke.ColorType.PASTEL_BLUE)),
        )
        self.wall_visual.set_world_translation(WALL_CENTER.tolist())
        self.wall_visual.set_casts_shadow(False)
        self.wall_visual.set_visible(False)
        self.set_camera_view(position=[3.2, 2.9, 4.8], target=[0, 1.4, 0])
        self.update_query()

    def update_query(self):
        # Camera movement and ordinary redraws reuse the previous BVH result.
        key = tuple(self.center)
        if key != self._last_query:
            distance, closest, gradient = self.sdf.query(self.center[None])
            self.distance = float(distance[0])
            self.closest = closest[0]
            self.gradient = gradient[0]
            self._last_query = key
            self.query_count += 1
        self.clearance = self.distance - self.radius

    def pre_render(self):
        self.update_query()
        self.update_slice_query()
        display_key = (tuple(self.center), self.radius, self.color_range, self.show_solid,
                       self.show_slice, self._last_slice, self.show_sample, self.sample_idx)
        if display_key == self._last_display:
            return
        self._last_display = display_key
        center_color = distance_colors(self.distance, self.color_range)
        sphere_color = self.red if self.clearance < 0 else self.green
        self.wall_visual.set_visible(self.show_solid)
        self.debug.overlay.points(
            path="/sdf/points", points=np.stack([self.center, self.closest]),
            colors=np.array([center_color, self.yellow]), size=13, overlay=True,
        )
        self.debug.overlay.lines(
            path="/sdf/distance", starts=self.center[None], ends=self.closest[None],
            colors=self.yellow, width=3, overlay=True,
        )
        self.debug.overlay.arrows(
            path="/sdf/gradient", starts=self.center[None],
            ends=(self.center + 0.5 * self.gradient)[None],
            colors=self.magenta, width=3, overlay=True,
        )

        angle = np.linspace(0, 2 * np.pi, 65)
        rings = []
        for axes in ((0, 1), (0, 2), (1, 2)):
            ring = np.zeros((len(angle), 3))
            ring[:, axes[0]] = self.radius * np.cos(angle)
            ring[:, axes[1]] = self.radius * np.sin(angle)
            rings.append(ring + self.center)
        self.debug.overlay.lines(
            path="/sdf/sphere", starts=np.concatenate([r[:-1] for r in rings]),
            ends=np.concatenate([r[1:] for r in rings]), colors=sphere_color,
            width=2, overlay=True,
        )
        self.debug.world_text.set(
            path="/sdf/label", text=f"d = {self.distance:+.3f} m",
            position=ke.Vec3(*(self.center + [0, self.radius + 0.15, 0])),
            pixel_size=22, depth_test=False,
        )

        if self._last_slice is not None:
            self.slice_colors = distance_colors(self.slice_distances, self.color_range)
            self.debug.overlay.points(
                path="/sdf/slice", points=self.slice_points, colors=self.slice_colors,
                size=5, overlay=True, hidden=not self.show_slice,
            )
            point = self.slice_points[self.sample_idx]
            closest = self.slice_closest[self.sample_idx]
            hidden = not (self.show_slice and self.show_sample)
            self.debug.overlay.points(
                path="/sdf/sample_points", points=np.stack([point, closest]),
                colors=np.array([self.slice_colors[self.sample_idx], self.yellow]),
                size=13, overlay=True, hidden=hidden,
            )
            self.debug.overlay.lines(
                path="/sdf/sample_distance", starts=point[None], ends=closest[None],
                colors=self.yellow, width=3, overlay=True, hidden=hidden,
            )
            self.debug.overlay.arrows(
                path="/sdf/sample_gradient", starts=point[None],
                ends=(point + 0.5 * self.slice_gradients[self.sample_idx])[None],
                colors=self.magenta, width=3, overlay=True, hidden=hidden,
            )

    def update_slice_query(self):
        """Keep geometry results independent of colors, selection, and camera."""
        key = (self.slice_plane, self.slice_offset)
        if not self.show_slice or key == self._last_slice:
            return
        first, second, fixed = SLICE_PLANES[self.slice_plane]
        u, v = np.meshgrid(
            np.linspace(SAMPLE_MIN[first], SAMPLE_MAX[first], SLICE_RESOLUTION),
            np.linspace(SAMPLE_MIN[second], SAMPLE_MAX[second], SLICE_RESOLUTION),
        )
        self.slice_points = np.empty((u.size, 3))
        self.slice_points[:, first] = u.ravel()
        self.slice_points[:, second] = v.ravel()
        self.slice_points[:, fixed] = self.slice_offset
        self.slice_distances, self.slice_closest, self.slice_gradients = self.sdf.query(self.slice_points)
        self._last_slice = key
        self.slice_query_count += 1

    def render(self):
        if not self._panel_initialized:
            imgui.set_next_window_pos(x=15, y=45)
            imgui.set_next_window_size(width=445, height=690)
            self._panel_initialized = True
        imgui.begin("Signed distance / sphere collision")
        imgui.text("Closed wall mesh | Y-up | meters")
        imgui.separator()
        for axis, low, high in ((0, -1.6, 1.6), (1, 0.0, 2.8), (2, -1.2, 1.2)):
            _, self.center[axis] = imgui.slider_float(
                f"Center {'XYZ'[axis]}", float(self.center[axis]), low, high,
            )
        _, self.radius = imgui.slider_float("Sphere radius", self.radius, 0.05, 0.35)
        for name, center in PRESETS.items():
            if imgui.button(name):
                self.center = np.array(center)
                self.radius = 0.25
            if name != "Fully inside":
                imgui.same_line()
        self.update_query()
        imgui.separator()
        imgui.text(f"Center signed distance: {self.distance:+.4f} m")
        imgui.text("Center: INSIDE" if self.distance < 0 else "Center: OUTSIDE / SURFACE")
        imgui.text(f"Sphere clearance (d - radius): {self.clearance:+.4f} m")
        imgui.text("Sphere: OVERLAPPING SOLID" if self.clearance < 0 else "Sphere: CLEAR / TOUCHING")
        imgui.text("Yellow: closest surface point and distance segment")
        imgui.text("Magenta arrow: increasing signed distance")
        imgui.text("Point colors: red (inside), white (zero), blue (outside)")
        imgui.text("Green / red sphere: clear / overlapping solid")
        _, self.show_solid = imgui.checkbox("Show solid wall (X-ray annotations)", self.show_solid)
        imgui.separator()
        imgui.text("Distance field slice")
        _, self.color_range = imgui.slider_float("Color range (m)", self.color_range, 0.02, 1.0)
        imgui.text(f"Red <= {-self.color_range:.2f} m | White = 0 | Blue >= {self.color_range:.2f} m")
        _, self.show_slice = imgui.checkbox("Show distance slice", self.show_slice)
        for plane, axes in SLICE_PLANES.items():
            if imgui.button(f"{'* ' if plane == self.slice_plane else ''}{plane}"):
                self.slice_plane = plane
                self.slice_offset = float(WALL_CENTER[axes[2]])
            if plane != "YZ":
                imgui.same_line()
        fixed = SLICE_PLANES[self.slice_plane][2]
        _, self.slice_offset = imgui.slider_float(
            f"Slice {'XYZ'[fixed]} (m)", self.slice_offset,
            float(SAMPLE_MIN[fixed]), float(SAMPLE_MAX[fixed]),
        )
        _, self.show_sample = imgui.checkbox("Inspect slice sample", self.show_sample)
        if self.show_sample and self.show_slice:
            _, self.sample_idx = imgui.slider_int("Sample index", self.sample_idx, 0, SLICE_RESOLUTION ** 2 - 1)
            if self._last_slice == (self.slice_plane, self.slice_offset):
                point = self.slice_points[self.sample_idx]
                imgui.text(f"Sample XYZ: {point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f} m")
                imgui.text(f"Sample distance: {self.slice_distances[self.sample_idx]:+.4f} m")
        imgui.separator()
        imgui.text("Queries update only when center or slice inputs change.")
        imgui.text("Gradient arrow is not a separate triangle-normal query.")
        imgui.text("No physics stepping or automatic collision response.")
        imgui.text("Center distance alone is not collider penetration depth.")
        imgui.end()


def main():
    args = render_example_parser(description=__doc__).parse_args()
    app: SignedDistanceApp = SignedDistanceApp()
    try:
        app.initialize(width=1280, height=850, up_axis=ke.UpAxis.Y, renderer=args.renderer)
        app.start()
    finally:
        app.close()


if __name__ == "__main__":
    main()
