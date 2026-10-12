"""Texture keyboard overlay plus independently cropped WASD/Shift keys."""

import kangengine as ke


class KeyboardVisualizerApp(ke.App):
    def setup(self):
        self.camera_keyboard_enabled = False
        self.keyboard_visualizer = ke.input.KeyboardVisualizer(
            self, full_keyboard=True, anchor=ke.render.ScreenAnchor.BOTTOM_CENTER
        )
        materials = self.create_standard_materials()
        self.add_ground(material=materials.ground)

    def render(self):
        self.keyboard_visualizer.draw()
        ke.imgui.begin("Keyboard texture regions")
        ke.imgui.text("Press WASD / Shift. All keys share one texture.")
        for index, key in enumerate(("W", "A", "S", "D", "LEFT_SHIFT")):
            if index:
                ke.imgui.same_line()
            self.keyboard_visualizer.draw_key(
                key, width=110 if key == "LEFT_SHIFT" else 48
            )
        ke.imgui.end()


if __name__ == "__main__":
    app = KeyboardVisualizerApp()
    app.initialize(width=1920, height=1080, up_axis=ke.UpAxis.Z)
    app.start()
