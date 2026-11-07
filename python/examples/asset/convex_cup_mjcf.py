"""Drop balls into a single convex hull and a cup reloaded from exported MJCF.

Export the Blender cup with python/tools/mesh_to_mjcf.py first.
Cups are held still so only the cavity's collision behavior is compared.
"""

from pathlib import Path

import numpy as np
import kangengine as ke
from kangengine import imgui, keys


ROOT = Path(__file__).resolve().parents[3]
OBJ_PATH = ROOT / "assets/objects/cup/cup.obj"
MJCF_PATH = ROOT / "build/convex/cup/mjcf/cup.xml"
BALL_RADIUS = 0.15


class CupComparison(ke.App):
    def setup(self):
        config = ke.physics.PhysicsConfig.y_up()
        self.physics = ke.physics.PhysicsWorld(config)
        self.configure_timing(ke.SimulationTimingConfig.from_dt(
            physics_dt=config.dt, fixed_dt=config.dt, render_hz=60.0,
        ))
        self.set_simulation_hotkeys_enabled(True)
        self.physics.add_default_ground()
        self.scene.add_ground(scale=10.0, material=self.create_standard_materials().ground)
        cup_material = self.create_pbr_material(
            base_color=ke.Vec4(1, 1, 1, 1), metallic=0.0, roughness=0.2,
        )
        ball_material = self.create_pbr_material(
            base_color=ke.Vec4(1, 1, 1, 1), metallic=0.0, roughness=0.55,
        )

        mesh: ke.scene.MeshData = ke.asset.load_obj(path=str(OBJ_PATH))
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
        self.bottom_offset = -float(vertices[:, 1].min())
        single = self.physics.create_dynamic_convex_compound(
            [ke.physics.ConvexMeshPart(vertices, faces)],
            pos=(-2.0, self.bottom_offset, 0.0),
        )
        single.set_kinematic(True)
        data: ke.asset.ArticulationDesc = ke.asset.MJCFLoader.load(mjcf_path=str(MJCF_PATH))
        compound = self.physics.create_dynamic_rigid(
            ke.asset.RigidDesc.from_articulation(data),
            pos=(2.0, self.bottom_offset, 0.0), kinematic=True,
        )
        self.cups = [single, compound]
        for cup_idx, (cup, color) in enumerate(zip(
            self.cups, (ke.ColorType.BLUE, ke.ColorType.PASTEL_GREEN), strict=True,
        )):
            for part_idx, collision in enumerate(cup.get_convex_collision_meshes()):
                view = self.scene.add_mesh(
                    path=f"/cups/cup_{cup_idx}/part_{part_idx}", mesh_data=collision,
                    material=cup_material, color=ke.preset_rgba(color),
                )
                view.set_local_translation(ke.Vec3(*cup.get_root_position()))

        self.balls = [
            self.physics.create_dynamic_sphere(radius=BALL_RADIUS, pos=(x, 3.0, 0.0))
            for x in (-2.0, 2.0)
        ]
        ball_mesh = ke.geometry.create_sphere_data(BALL_RADIUS, 32, 16)
        self.ball_views = [
            self.scene.add_mesh(
                path=f"/balls/ball_{i}", mesh_data=ball_mesh, material=ball_material,
                color=ke.preset_rgba(ke.ColorType.YELLOW),
            )
            for i in range(2)
        ]
        self.set_light_direction(ke.Vec3(0.35, -0.82, 0.45))
        self.set_light_ambient(ke.Vec3(0.3, 0.3, 0.3))
        camera = self.get_camera()
        camera.set_target_pos(ke.Vec3(0.0, 0.65, 0.0))
        camera.set_camera_pos(ke.Vec3(0.5, 9.0, 2.0))
        camera.set_fov(45.0)

    def pre_update(self):
        if self.was_key_pressed(keys.R):
            for x, ball in zip((-2.0, 2.0), self.balls, strict=True):
                ball.set_root_state((x, 3.0, 0.0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0))

    def fixed_update(self, fixed_dt):
        self.physics.step()

    def pre_render(self):
        for view, ball in zip(self.ball_views, self.balls, strict=True):
            view.set_local_translation(ke.Vec3(*ball.get_root_position()))

    def render(self):
        imgui.begin("Cup collision comparison")
        imgui.text("Left: single hull blocks the opening")
        imgui.text(f"Right: exported MJCF, {self.cups[1].num_shapes()} convex parts")
        imgui.text("Cups are held still; balls fall under gravity.")
        imgui.text("Enter: play/pause    Space: pause/step    R: drop again")
        imgui.end()


if __name__ == "__main__":
    app = CupComparison()
    app.initialize(width=1440, height=900)
    app.start()
