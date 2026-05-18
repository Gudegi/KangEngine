"""Generic MJCF articulation DOF position-control viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import imgui, keys, scene


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def default_mjcf_path() -> Path:
    return Path(package_asset_path("characters", "kw", "kw.xml"))


class MjcfDofControlApp(ke.App):
    """Load an MJCF articulation and expose every DOF as an ImGui slider."""

    window_title = "MJCF DOF Control"
    object_name = "mjcf"
    prim_base_path = "/mjcf"
    camera_pos = (3.8, -5.4, 1.2)
    camera_target = (0.0, 0.0, 0.45)
    visual_color = (0.22, 0.48, 0.95, 1.0)
    ground_size = 10.0
    root_pos = (0.0, 0.0, 0.0)
    root_rot_xyzw = (0.0, 0.0, 0.0, 1.0)
    fixed_base = False
    order = "DFS"
    sim_dt = 1.0 / 240.0
    step_substeps = 4
    default_kp = 120.0
    default_kd = 12.0
    default_anim_amp = 0.35
    default_anim_speed = 1.0
    default_contact_force_scale = 0.002
    contact_force_threshold = 1e-3
    visual_alpha_with_collision = 0.1

    def __init__(self, mjcf_path: str | Path):
        super().__init__()
        self.mjcf_path = str(Path(mjcf_path).expanduser().resolve())

    def setup(self):
        self.paused = False
        self.elapsed = 0.0
        self.kp = float(self.default_kp)
        self.kd = float(self.default_kd)
        self.animate = True
        self.anim_amp = float(self.default_anim_amp)
        self.anim_speed = float(self.default_anim_speed)
        self.show_collision = False
        self.show_contact_forces = False
        self.contact_force_scale = float(self.default_contact_force_scale)
        self.contact_force_handle = None
        self.contact_force_color = np.array([[1.0, 0.25, 0.05, 1.0]], dtype=np.float32)
        self.empty_vec3 = np.empty((0, 3), dtype=np.float32)
        self.empty_vec4 = np.empty((0, 4), dtype=np.float32)

        self.configure_camera()
        self.create_shaders()
        self.create_world()
        self.load_articulation()
        self._reset()
        self.print_summary()

    def configure_camera(self):
        self.getCamera().set_camera_pos(ke.vec3(*self.camera_pos))
        self.getCamera().set_target_pos(ke.vec3(*self.camera_target))

    def create_shaders(self):
        device = self.getGraphicsDevice()
        vs = package_asset_path("shaders", "common.vs")
        fs = package_asset_path("shaders", "common.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.robot_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)
        for shader in (self.robot_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

    def create_world(self):
        self.world = ke.KangSimWorld(
            num_envs=1,
            sim_dt=float(self.sim_dt),
            add_ground=True,
        )
        self.visual = ke.KangWorldVisualBridge(self, self.world)

        ground = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(float(self.ground_size), ke.UpAxis.Z))
        self.addShape(self.ground_shader, ground)

    def load_articulation(self):
        data = self.world.load_mjcf(self.mjcf_path, order=self.order)
        self.obj_id = 0
        config = (
            ke.ArticulationConfig.fixed_base()
            if self.fixed_base
            else ke.ArticulationConfig.free_base()
        )
        self.robot = self.world.add_articulation(
            data,
            env_id=0,
            obj_id=self.obj_id,
            name=self.object_name,
            config=config,
        ).articulation

        self.articulation_visual_record = self.visual.add_articulation(
            0,
            self.obj_id,
            self.mjcf_path,
            prim_base_path=self.prim_base_path,
            order=self.order,
            shader=self.robot_shader,
            collision_base_path=f"{self.prim_base_path}_collision",
            show_collision=self.show_collision,
            color=np.array(self.visual_color, dtype=np.float32),
        )
        self.visual_body_prims = self.articulation_visual_record.body_prims
        self.visual_body_handles = self.articulation_visual_record.body_handles
        # self.collision_body_prims = self.articulation_visual_record.collision_prims

        self.num_dofs = self.robot.num_dofs()
        self.dof_names = self.world.state.get_obj_dof_names(self.obj_id)
        self.dof_limits = np.asarray(
            self.world.state.get_obj_dof_limits(self.obj_id), dtype=np.float32
        )
        if self.dof_limits.shape != (self.num_dofs, 2):
            self.dof_limits = np.tile(
                np.array([-math.pi, math.pi], dtype=np.float32),
                (self.num_dofs, 1),
            )
        self.targets = self.initial_targets()

    def initial_targets(self) -> np.ndarray:
        return np.zeros(self.num_dofs, dtype=np.float32)

    def print_summary(self):
        print(
            f"{self.object_name} loaded: "
            f"links={self.robot.num_links()} dofs={self.num_dofs}"
        )
        print("DOFs:", ", ".join(self.dof_names))
        print("XML:", self.mjcf_path)

    def _reset(self):
        self.elapsed = 0.0
        self.targets[:] = self.initial_targets()
        self.world.set_root_state(
            None,
            self.obj_id,
            np.array(self.root_pos, dtype=np.float32),
            np.array(self.root_rot_xyzw, dtype=np.float32),
        )
        self.world.set_dof_state(None, self.obj_id, self.targets)
        self.world.set_cmd(
            None,
            self.obj_id,
            self.targets,
            mode=ke.ControlMode.POS,
            kp=self.kp,
            kd=self.kd,
        )
        self.world.step(substeps=0, apply_commands=False)
        self.visual.sync()
        self._update_contact_force_arrows()

    def _animated_targets(self):
        out = np.zeros_like(self.targets)
        for i in range(self.num_dofs):
            lo, hi = self.slider_limits(i)
            span = max(0.0, min(float(hi - lo) * 0.35, self.anim_amp))
            center = 0.5 * float(lo + hi)
            phase = self.elapsed * self.anim_speed + i * 0.75
            out[i] = np.clip(center + span * math.sin(phase), lo, hi)
        return out

    def slider_limits(self, dof_index: int) -> tuple[float, float]:
        lo, hi = self.dof_limits[dof_index]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return -math.pi, math.pi
        return float(lo), float(hi)

    def preRender(self):
        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused
        if self.was_key_pressed(keys.R):
            self._reset()
        if self.was_key_pressed(keys.A):
            self.animate = not self.animate

        if self.paused:
            return

        self.elapsed += 1.0 / 60.0
        if self.animate:
            self.targets[:] = self._animated_targets()

        self.world.set_cmd(
            None,
            self.obj_id,
            self.targets,
            mode=ke.ControlMode.POS,
            kp=self.kp,
            kd=self.kd,
        )
        self.world.step(substeps=int(self.step_substeps))
        self.visual.sync()
        self._update_contact_force_arrows()
        self.checkError()

    def _clear_contact_force_arrows(self):
        if self.contact_force_handle is None:
            return
        ke.scene.DebugDraw.update_arrows(
            self,
            self.contact_force_handle,
            self.empty_vec3,
            self.empty_vec3,
            self.empty_vec4,
        )

    def _update_contact_force_arrows(self):
        if not self.show_contact_forces:
            self._clear_contact_force_arrows()
            return

        # Body-aggregated force visualization. 
        # the arrows start at link origins instead of real contact points.
        # body_pos = np.asarray(
        #     self.world.state.get_body_pos(self.obj_id)[0], dtype=np.float32
        # )
        # forces = np.asarray(
        #     self.world.state.get_contact_forces(self.obj_id)[0], dtype=np.float32
        # )
        # active = np.linalg.norm(forces, axis=1) > float(self.contact_force_threshold)
        # starts = body_pos[active]
        # ends = starts + forces[active] * float(self.contact_force_scale)

        contacts = self.world.physics.get_contacts()
        starts = []
        ends = []
        dt = max(float(self.world.sim_dt), 1e-8)
        for contact in contacts:
            position = self._vec3_to_np(contact.position)
            force = self._vec3_to_np(contact.impulse) / dt
            if np.linalg.norm(force) <= float(self.contact_force_threshold):
                continue
            starts.append(position)
            ends.append(position + force * float(self.contact_force_scale))

        if not starts:
            self._clear_contact_force_arrows()
            return

        starts = np.asarray(starts, dtype=np.float32)
        ends = np.asarray(ends, dtype=np.float32)
        colors = np.repeat(self.contact_force_color, starts.shape[0], axis=0)

        if self.contact_force_handle is None:
            self.contact_force_handle = ke.scene.DebugDraw.log_arrows(
                self,
                self.robot_shader,
                "/debug/contact_forces",
                starts,
                ends,
                colors,
                0.015,
                12,
            )
        else:
            ke.scene.DebugDraw.update_arrows(
                self,
                self.contact_force_handle,
                starts,
                ends,
                colors,
            )

    @staticmethod
    def _vec3_to_np(value) -> np.ndarray:
        return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float32)

    def _set_visual_alpha(self, alpha: float):
        color = np.array(self.visual_color, dtype=np.float32).reshape(-1)
        if color.size == 3:
            color = np.concatenate([color, np.ones(1, dtype=np.float32)])
        color = color[:4].copy()
        color[3] = float(alpha)
        rgba = ke.vec4(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        for prim in self.visual_body_prims:
            prim.set_display_color_alpha(rgba)
        if self.visual_body_handles:
            colors = color.reshape(1, 4)
            for handle in self.visual_body_handles:
                self.setShapeColors(handle, colors)

    def _set_collision_visible(self, visible: bool):
        self.show_collision = bool(visible)
        self.visual.set_collision_visible(self.show_collision)
        self._set_visual_alpha(
            self.visual_alpha_with_collision if self.show_collision else 1.0
        )

    def render(self):
        imgui.begin(self.window_title)
        imgui.text(f"State: {'paused' if self.paused else 'running'}")
        imgui.text("Space: pause/resume    R: reset    A: auto motion")
        imgui.text(f"Links: {self.robot.num_links()}  DOFs: {self.num_dofs}")
        imgui.text(Path(self.mjcf_path).name)
        imgui.separator()
        _, self.kp = imgui.slider_float("kp", self.kp, 0.0, 1000.0)
        _, self.kd = imgui.slider_float("kd", self.kd, 0.0, 80.0)
        _, self.animate = imgui.checkbox("Animate targets", self.animate)
        _, self.anim_amp = imgui.slider_float("anim amplitude", self.anim_amp, 0.0, 1.5)
        _, self.anim_speed = imgui.slider_float("anim speed", self.anim_speed, 0.0, 6.0)
        changed, self.show_collision = imgui.checkbox(
            "Show collision prims",
            self.show_collision,
        )
        if changed:
            self._set_collision_visible(self.show_collision)
        changed, self.show_contact_forces = imgui.checkbox(
            "Show contact forces",
            self.show_contact_forces,
        )
        if changed:
            self._update_contact_force_arrows()
        _, self.contact_force_scale = imgui.slider_float(
            "contact force scale",
            self.contact_force_scale,
            0.0,
            0.02,
        )
        imgui.separator()

        for i, name in enumerate(self.dof_names):
            lo, hi = self.slider_limits(i)
            changed, value = imgui.slider_float(name, float(self.targets[i]), lo, hi)
            if changed:
                self.targets[i] = value
                self.animate = False

        pos = self.world.state.get_dof_pos(self.obj_id)[0]
        imgui.separator()
        imgui.text("Current DOF positions")
        for name, value in zip(self.dof_names, pos):
            imgui.text(f"{name}: {float(value): .3f}")
        imgui.end()

    def cleanup(self):
        if hasattr(self, "world"):
            self.world.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mjcf_path",
        nargs="?",
        default=str(default_mjcf_path()),
        help="Path to an MJCF XML file",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        help="Use a fixed root instead of the default free root.",
    )
    parser.add_argument("--order", default="DFS", choices=("DFS", "BFS"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    class CliMjcfDofControlApp(MjcfDofControlApp):
        fixed_base = args.fixed_base
        order = args.order

    app = CliMjcfDofControlApp(args.mjcf_path)
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
