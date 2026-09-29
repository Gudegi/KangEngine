"""Procedural three-stage scissor: inbound hinges/slider plus external D6 pivots.

This compact planar mechanism demonstrates the construction pattern used by
examples/physics/physx_articulation.cpp; it is not a dimension-for-dimension port.
"""

import argparse
import math

import numpy as np
import kangengine as ke
from kangengine import imgui


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def make_scissor():
    p = ke.physics
    Builder = p.ArticulationBuilder
    builder = Builder()
    poses, shapes, pivots = {}, {}, []

    def link(name, parent, position, angle, half, mass=1., joint=None):
        position = np.asarray(position)
        parent_position, parent_angle = poses[parent] if parent else (np.zeros(3), 0.)
        local_position = rotation_y(parent_angle).T @ (position - parent_position)
        delta = angle - parent_angle
        inertia = mass / 3 * np.array([half[1]**2 + half[2]**2,
                                      half[0]**2 + half[2]**2,
                                      half[0]**2 + half[1]**2])
        builder.add_link(name, parent=parent, position=local_position,
                         rotation_xyzw=(0., math.sin(delta/2), 0., math.cos(delta/2)),
                         joint=joint, mass=mass, inertia_diagonal=inertia,
                         shapes=[Builder.BoxShape(half)])
        poses[name], shapes[name] = (position, angle), half

    link("base", None, (0., 0., .2), 0., (1.2, .3, .2), mass=10.)
    link("left_root", "base", (-.9, 0., .5), 0., (.08, .12, .08))
    link("slider", "base", (.9, 0., .5), 0., (.08, .12, .08),
         joint=Builder.PrismaticJoint(axis=(1., 0., 0.), limits=(-1.3, .15)))
    angle, height = math.asin(.9), math.sqrt(1-.9**2)
    left_parent, right_parent = "left_root", "slider"
    for level in range(3):
        left, right = f"left_{level}", f"right_{level}"
        center = (0., 0., .5 + (2*level+1)*height)
        for name, parent, tilt in ((left, left_parent, angle), (right, right_parent, -angle)):
            link(name, parent, center, tilt, (.04, .04, 1.),
                 joint=Builder.RevoluteJoint(axis=(0., 1., 0.), joint_offset=(0., 0., -1.),
                                       limits=(-2.8, 2.8)))
        pivots.append((left, right))
        left_parent, right_parent = right, left
    return builder, shapes, pivots


class Scissor:
    def __init__(self, gpu=False):
        self.world = ke.sim.KangSimWorld(num_envs=1, sim_device="cuda:0" if gpu else "cpu",
                                       sim_dt=1/240, add_ground=False)
        self.joints = []
        try:
            builder, self.shapes, pivots = make_scissor()
            config = ke.physics.ArticulationConfig.fixed_base()
            config.solver_position_iteration_count = 64
            self.robot = self.world.add_articulation(builder.build_template(config=config), config=config)
            self.names = list(self.robot.articulation.template.body_names)
            self.pivots = [(self.names.index(a), self.names.index(b)) for a, b in pivots]
            config = ke.physics.D6JointConfig()
            config.enabled = True
            for a, b in pivots:
                self.joints.append(self.world.physics.create_d6_joint(
                    self.robot.articulation.link(a), self.robot.articulation.link(b), config=config))
            self.slider_dof = list(self.robot.articulation.get_dof_names()).index("slider/0")
            self.targets = np.zeros(self.robot.num_dofs, dtype=np.float32)
            self.kp = self.targets.copy()
            self.kd = np.full_like(self.targets, 2.)
            self.kp[self.slider_dof], self.kd[self.slider_dof] = 20000., 1000.
            self.robot.articulation.set_kps(self.kp)
            self.robot.articulation.set_kds(self.kd)
            self.robot.articulation.set_drive_targets(self.targets)
            if gpu:
                self.world.init_gpu_system(cuda_device_id=0)
            self.target = 0.
            self.world.step(substeps=0)
        except Exception:
            self.close()
            raise

    def step(self):
        self.targets[self.slider_dof] = self.target
        self.world.set_cmd(None, 0, self.targets, mode="pos", kp=self.kp, kd=self.kd)
        self.world.step(refresh=False)

    def state(self):
        self.world.state.refresh()
        positions = self.world.state.get_body_pos(0)[0].detach().cpu().numpy().copy()
        rotations = self.world.state.get_body_rot(0)[0].detach().cpu().numpy().copy()
        return positions, rotations

    def close(self):
        for joint in self.joints:
            joint.release()
        self.world.release()


class Viewer(ke.App):
    def __init__(self, gpu=False):
        super().__init__()
        self.gpu = gpu

    def setup(self):
        self.demo = Scissor(self.gpu)
        self.configure_timing(ke.SimulationTimingConfig(
            physics_hz=240., fixed_update_hz=240., render_hz=60.))
        self.set_simulation_hotkeys_enabled(True)
        self.materials = self.create_standard_materials()
        self.set_camera_view([5., -8., 4.], [0., 0., 2.])
        self.views = []
        for name in self.demo.names:
            half = self.demo.shapes[name]
            mesh = ke.geometry.create_box_data(*[2*x for x in half])
            self.views.append(self.scene.add_mesh(f"/scissor/{name}", mesh, self.materials.common))

    def fixed_update(self, fixed_dt):
        self.demo.step()

    def pre_render(self):
        positions, rotations = self.demo.state()
        for view, position, rotation in zip(self.views, positions, rotations):
            view.prim.set_local_translation(ke.Vec3(*map(float, position)))
            view.prim.set_local_rotation(ke.Quat(float(rotation[3]), *map(float, rotation[:3])))

    def render(self):
        imgui.begin("Procedural scissor")
        _, self.demo.target = imgui.slider_float("Slider target", self.demo.target, -1., 0.)
        imgui.text("Internal slider/hinges + individual external D6 pivots")
        imgui.end()

    def cleanup(self):
        if hasattr(self, "demo"):
            self.demo.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        app = Viewer(args.gpu)
        app.initialize(1280, 800, False, ke.UpAxis.Z)
        app.start()
        return
    demo = Scissor(args.gpu)
    try:
        heights, max_error = [], 0.
        for target in (0., -.8, 0.):
            demo.target = target
            for _ in range(720):
                demo.step()
            positions, _ = demo.state()
            assert np.isfinite(positions).all()
            for a, b in demo.pivots:
                max_error = max(max_error, float(np.linalg.norm(positions[a]-positions[b])))
            heights.append(float(positions[-1, 2]))
        assert max_error < .02, max_error
        assert heights[1] > heights[0] + .5, heights
        assert abs(heights[2] - heights[0]) < .1, heights
        print(f"SCISSOR PASS gpu={args.gpu} heights={heights} max_pivot_error={max_error:.6f}")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
