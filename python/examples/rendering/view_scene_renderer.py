"""Compare PBR materials, textures, shadows, and moving hierarchies across renderers."""

import argparse
import math
from pathlib import Path

import kangengine as ke


def rgba(preset):
    color = ke.ColorLibrary.get(preset)
    return (color.r, color.g, color.b, color.a)


class SceneRendererApp(ke.App):
    hdri_path = None

    def setup(self):
        self.animate = True
        self.angle = 0.0
        self.controls_docked = False
        self.set_fixed_update_hz(update_hz=60)
        self.set_ui_layout_mode(mode=ke.UILayoutMode.EDITOR)
        self.set_camera_view(position=[10, 8, 14], target=[0, 1, 0])
        light = ke.scene.DirectionalLight()
        light.direction = ke.Vec3(-0.5, -1, -0.4)
        light.intensity = 3
        light.ambient = ke.Vec3(0, 0, 0)
        self.scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
        self.environment_enabled = True
        self.environment: ke.scene.Prim = self.scene.define_prim(
            path="/lights/environment", prim_type=ke.scene.PrimType.LIGHT,
        )
        self.environment_light: ke.scene.DomeLight = ke.scene.DomeLight()
        self.environment_light.intensity = 0.3
        if self.hdri_path is not None:
            self.environment_light.texture_path = str(self.hdri_path)
            self.environment_light.texture_format = "latlong"
        self.environment.set_dome_light(light=self.environment_light)

        # Shared geometry and materials keep repeated objects comparable.
        cube: ke.scene.MeshData = ke.geometry.create_cube_data(scale=1.0)
        sphere: ke.scene.MeshData = ke.geometry.create_sphere_data(
            radius=0.5, num_longitudes=48, num_latitudes=24,
        )
        floor_material: ke.material.PBRMaterial = self.create_pbr_material(
            base_color=rgba(ke.ColorType.LIGHT_GRAY), roughness=0.7,
        )
        wall_material: ke.material.PBRMaterial = self.create_pbr_material(
            base_color=rgba(ke.ColorType.PASTEL_BLUE), roughness=0.85,
        )
        floor = self.scene.add_mesh(path="/room/floor", mesh_data=cube, material=floor_material)
        floor.set_local_scale(ke.Vec3(14, 0.2, 12))
        floor.set_local_translation(ke.Vec3(0, -0.1, 0))
        wall = self.scene.add_mesh(path="/room/back", mesh_data=cube, material=wall_material)
        wall.set_local_scale(ke.Vec3(14, 4, 0.2))
        wall.set_local_translation(ke.Vec3(0, 2, -5.5))

        # Left -> right: roughness; front -> back: metallic factor.
        for row, metallic in enumerate((0.0, 0.5, 1.0)):
            for column, roughness in enumerate((0.05, 0.2, 0.4, 0.65, 1.0)):
                material: ke.material.PBRMaterial = self.create_pbr_material(
                    base_color=rgba(ke.ColorType.PASTEL_ORANGE),
                    metallic=metallic, roughness=roughness,
                )
                surface = self.scene.add_mesh(
                    path=f"/materials/row_{row}/sphere_{column}",
                    mesh_data=sphere, material=material,
                )
                surface.set_local_translation(ke.Vec3((column - 2) * 1.5, 0.5, 1 - row * 1.7))

        texture = self.load_texture(
            Path(ke.__file__).parent / "assets/textures/awesomeface.png", srgb=True,
        )
        textured: ke.material.PBRMaterial = self.create_pbr_material(
            base_color=rgba(ke.ColorType.WHITE), roughness=0.6, base_color_texture=texture,
        )
        for index in range(3):
            surface = self.scene.add_mesh(
                path=f"/textures/cube_{index}", mesh_data=cube, material=textured,
            )
            surface.set_local_translation(ke.Vec3(-5, 0.6 + index * 1.1, -2))
            surface.set_local_rotation_axis_angle(axis=ke.Vec3(0, 1, 0), angle_radians=index * 0.3)

        self.rotor: ke.scene.Prim = self.scene.define_prim("/moving", ke.scene.PrimType.XFORM)
        self.rotor.set_local_translation(ke.Vec3(0, 1.1, 3.5))
        arm = self.scene.add_mesh(path="/moving/arm", mesh_data=cube, material=wall_material)
        arm.set_local_scale(ke.Vec3(4, 0.2, 0.3))
        for index, x in enumerate((-2.0, 2.0)):
            surface = self.scene.add_mesh(
                path=f"/moving/cube_{index}", mesh_data=cube, material=textured,
            )
            surface.set_local_translation(ke.Vec3(x, 0, 0))

    def fixed_update(self, fixed_dt):
        if self.animate:
            self.angle = (self.angle + fixed_dt * 0.6) % math.tau
            self.rotor.set_local_rotation_axis_angle(
                axis=ke.Vec3(0, 1, 0), angle_radians=self.angle,
            )

    def pre_render(self):
        if not self.controls_docked:
            self.controls_docked = ke.imgui.dock_window("Scene Controls", "Inspector")
        if ke.imgui.begin("Scene Controls"):
            _, self.animate = ke.imgui.checkbox("Animate", self.animate)
            changed, self.environment_enabled = ke.imgui.checkbox("Environment", self.environment_enabled)
            if changed:
                self.environment.set_visible(visible=self.environment_enabled)
            changed, intensity = ke.imgui.slider_float(
                "Environment intensity", self.environment_light.intensity, 0.0, 2.0,
            )
            if changed:
                self.environment_light.intensity = intensity
                self.environment.set_dome_light(light=self.environment_light)
            ke.imgui.text("Spheres: roughness increases left to right.")
            ke.imgui.text("Metallic increases front to back.")
        ke.imgui.end()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    parser.add_argument("--hdri-path", type=Path, help="Latlong environment image for ovrtx")
    args = parser.parse_args()
    if args.hdri_path is not None and args.backend != "ovrtx":
        parser.error("--hdri-path requires --backend ovrtx; Native supports uniform dome lighting")
    app = SceneRendererApp()
    app.hdri_path = args.hdri_path
    try:
        app.initialize(width=1280, height=800, up_axis=ke.UpAxis.Y, render_backend=args.backend)
        app.start()
    finally:
        app.close()


if __name__ == "__main__":
    main()
