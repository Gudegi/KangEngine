"""
This example shows simple pd target control of walking motion.
KW has defined for human like, so it has several spherical joints (hips, shoulders, ankles..). 
But in kw.xml definition, all joints are defined hinge joint(1~3 hinge). It means articulation can't fully follow the original motion.
# TODO: support spherical 
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import animation, asset, imgui, keys, scene

from view_motion import default_char_file, default_motion_file, load_motion


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def twist_angle(q_wxyz: np.ndarray, axis: np.ndarray) -> float:
    """Return the signed twist angle around a local joint axis."""
    w = float(q_wxyz[0])
    xyz = np.array([float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])])
    if w < 0.0:
        w, xyz = -w, -xyz
    proj = float(np.dot(xyz, axis))
    return float(2.0 * np.arctan2(proj, w))


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> list[float]:
    return [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]


class KwMotionTrackingApp(ke.App):
    def __init__(
        self,
        char_file: Path,
        motion_file: Path,
        order: str = "DFS",
    ):
        super().__init__()
        self.char_file = str(char_file)
        self.motion_file = str(motion_file)
        self.order = str(order)

    def setup(self):
        self.root_pos, self.local_rot, self.fps = load_motion(Path(self.motion_file))
        self.frame_idx = 0
        self.paused = False
        self.kp = 200.0
        self.kd = 10.0
        self.track_root = True
        self.show_collision = False

        device = self.getGraphicsDevice()
        vs = package_asset_path("shaders", "common.vs")
        fs = package_asset_path("shaders", "common.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.robot_shader = device.createShaderFromFile(vs, fs)
        self.ghost_shader = device.createShaderFromFile(vs, fs)
        self.ground_shader = device.createShaderFromFile(vs, checker_fs)

        for shader in (self.robot_shader, self.ghost_shader, self.ground_shader):
            shader.use()
            shader.setUniformBlockBinding("cameraUBO", 0)
            shader.setUniformBlockBinding("lightUBO", 1)
            shader.setUniformBlockBinding("shadowUBO", 2)

        self.ground_shader.use()
        self.ground_shader.setVec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.setVec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

        self.sim_world = ke.KangSimWorld(add_ground=True)

        ground = self.getScene().define_prim("/ground", scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(100.0, ke.UpAxis.Z))
        self.addShape(self.ground_shader, ground)

        mjcf_data = asset.MJCFLoader.load(self.char_file, order=self.order)
        sim_record = self.sim_world.add_articulation(
            mjcf_data,
            env_id=0,
            obj_id=0,
            name="kw",
            config=ke.ArticulationConfig.free_base(),
        )
        self.articulation = sim_record.articulation
        self.num_dofs = self.articulation.num_dofs()

        self.visual_bridge = ke.KangWorldVisualBridge(self, self.sim_world)
        self.visual_bridge.add_articulation(
            0,
            0,
            self.char_file,
            prim_base_path="/robot",
            order=self.order,
            shader=self.robot_shader,
            collision_base_path="/collision",
            show_collision=self.show_collision,
        )

        self.ghost = animation.SkeletonBridge.from_mjcf(
            self.char_file,
            self.getScene(),
            "/ghost",
            1.0,
            self.order,
        )
        for prim in self.ghost.body_prims():
            prim.set_display_color_alpha(ke.vec4(0.2, 0.6, 1.0, 0.35))
            self.addShape(self.ghost_shader, prim)

        self.body_axes: dict[int, list[np.ndarray]] = {}
        for body_idx, joints in self.articulation.joints().items():
            axes = []
            for joint in joints:
                axis = joint.axis
                axes.append(np.array([axis.x, axis.y, axis.z], dtype=np.float64))
            if axes:
                self.body_axes[int(body_idx)] = axes

        if self.local_rot.shape[1] != self.articulation.num_links():
            print(
                "Warning: motion/body count mismatch: "
                f"{self.local_rot.shape[1]} motion bodies vs "
                f"{self.articulation.num_links()} articulation links"
            )

        self._reset()
        print(
            f"KW motion tracking loaded: {self.articulation.num_links()} links, "
            f"{self.num_dofs} DOFs, {self.local_rot.shape[0]} frames, {self.fps} fps"
        )
        print(f"character: {self.char_file}")
        print(f"motion: {self.motion_file}")
        self.checkError()

    def _motion_dof_targets(self, local_rot_frame: np.ndarray) -> list[float]:
        targets: list[float] = []
        num_links = min(self.articulation.num_links(), local_rot_frame.shape[0])
        for body_idx in range(1, num_links):
            axes = self.body_axes.get(body_idx)
            if not axes:
                continue
            q = local_rot_frame[body_idx]
            for axis in axes:
                targets.append(twist_angle(q, axis))
        return targets

    def _apply_ghost(self, idx: int):
        root = self.root_pos[idx]
        self.ghost.set_root_translation(
            ke.vec3(float(root[0]), float(root[1]), float(root[2]))
        )
        num_bodies = min(self.ghost.num_bodies(), self.local_rot.shape[1])
        for body_idx in range(num_bodies):
            q = self.local_rot[idx][body_idx]
            self.ghost.set_joint_rotation(
                body_idx,
                ke.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            )
        self.ghost.apply_pose()

    def _reset(self):
        self.frame_idx = 0
        self._apply_ghost(0)
        self.sim_world.set_root_state(
            None,
            0,
            self.root_pos[0],
            wxyz_to_xyzw(self.local_rot[0][0]),
        )
        targets = self._motion_dof_targets(self.local_rot[0])
        if len(targets) == self.num_dofs:
            self.sim_world.set_dof_state(None, 0, targets)
            self.sim_world.set_cmd(None, 0, targets, ke.ControlMode.POS, self.kp, self.kd)
        self.sim_world.step(substeps=0, apply_commands=False)
        self.visual_bridge.sync()

    def preRender(self):
        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused
        if self.was_key_pressed(keys.R):
            self._reset()

        if self.paused:
            return

        idx = self.frame_idx
        self._apply_ghost(idx)

        if self.track_root:
            self.sim_world.set_root_state(
                None,
                0,
                self.root_pos[idx],
                wxyz_to_xyzw(self.local_rot[idx][0]),
            )

        targets = self._motion_dof_targets(self.local_rot[idx])
        if len(targets) == self.num_dofs:
            self.sim_world.set_cmd(None, 0, targets, ke.ControlMode.POS, self.kp, self.kd)

        self.sim_world.step()
        self.visual_bridge.sync()

        self.frame_idx = (self.frame_idx + 1) % self.root_pos.shape[0]
        self.checkError()

    def render(self):
        state = "PAUSED" if self.paused else "running"
        imgui.begin("KW Motion Tracking")
        imgui.text(
            f"Links: {self.articulation.num_links()}  "
            f"DOFs: {self.num_dofs}  |  {state}"
        )
        imgui.text(f"Frame: {self.frame_idx} / {self.root_pos.shape[0] - 1}")
        imgui.text("Space: pause/resume    R: reset")
        imgui.separator()
        _, self.kp = imgui.slider_float("kp (stiffness)", self.kp, 0.0, 1000.0)
        _, self.kd = imgui.slider_float("kd (damping)", self.kd, 0.0, 100.0)
        imgui.separator()
        _, self.track_root = imgui.checkbox("Track root (kinematic)", self.track_root)
        changed, self.show_collision = imgui.checkbox(
            "Show collision",
            self.show_collision,
        )
        if changed:
            self.visual_bridge.set_collision_visible(self.show_collision)
        imgui.end()

    def postRender(self):
        pass

    def cleanup(self):
        if hasattr(self, "visual_bridge"):
            self.visual_bridge = None
        if hasattr(self, "sim_world"):
            self.sim_world.release()
            self.sim_world = None
        if hasattr(self, "articulation"):
            self.articulation = None


def main():
    missing = [
        name
        for name in (
            "PhysicsConfig",
            "PhysicsWorld",
            "ArticulationConfig",
            "Articulation",
            "PhysicsBridge",
        )
        if not hasattr(ke, name)
    ]
    if missing:
        raise RuntimeError(
            "KangEngine built without PhysX. Missing: " + ", ".join(missing)
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--char-file", default=str(default_char_file()))
    parser.add_argument("--motion-file", default=str(default_motion_file()))
    parser.add_argument("--order", default="DFS", choices=("DFS", "BFS"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    char_file = Path(args.char_file).resolve()
    motion_file = Path(args.motion_file).resolve()
    if not char_file.exists():
        raise FileNotFoundError(char_file)
    if not motion_file.exists():
        raise FileNotFoundError(motion_file)

    app = KwMotionTrackingApp(char_file, motion_file, order=args.order)
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
