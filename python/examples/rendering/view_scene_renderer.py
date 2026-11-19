"""View the same PBR scene with Native or ovrtx rendering."""

import argparse
from pathlib import Path

import kangengine as ke


class SceneRendererApp(ke.App):
    def setup(self):
        self.set_ui_layout_mode(mode=ke.UILayoutMode.EDITOR)
        self.set_camera_view(position=[0, 1, 5], target=[0, 0, 0])
        light = ke.scene.DirectionalLight()
        light.direction = ke.Vec3(0, -0.3, -1)
        light.intensity = 3
        light.ambient = ke.Vec3(0, 0, 0)
        self.scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
        texture = self.load_texture(
            Path(ke.__file__).parent / "assets/textures/awesomeface.png", srgb=True,
        )
        material: ke.material.PBRMaterial = self.create_pbr_material(roughness=0.8)
        material.set_base_color_texture(texture)
        self.scene.add_mesh(
            path="/cube", mesh_data=ke.geometry.create_cube_data(1.5), material=material,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = SceneRendererApp()
    try:
        app.initialize(width=960, height=640, render_backend=args.backend)
        app.start()
    finally:
        app.close()


if __name__ == "__main__":
    main()
