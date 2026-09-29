"""KW5 world-anchor D6 playground. Run with --gpu or --headless for a smoke check."""

import argparse
from pathlib import Path

import numpy as np
import kangengine as ke
from kangengine import imgui, keys


LIMBS = ("LeftWrist", "RightWrist", "LeftToe", "RightToe")


class Suspension:
    def __init__(self, gpu=False):
        self.gpu = gpu
        self.xml = str(Path(ke.__file__).resolve().parent / "assets/characters/kw/kw5.xml")
        self.world = ke.sim.KangSimWorld(num_envs=1, sim_device="cuda:0" if gpu else "cpu",
                                       sim_dt=1 / 120, add_ground=True)
        self.joints = None
        try:
            data = self.world.load_mjcf(self.xml, order="DFS")
            self.robot = self.world.add_articulation(
                data, env_id=0, obj_id=0, config=ke.physics.ArticulationConfig.free_base())
            body_names = list(self.robot.articulation.template.body_names)
            self.link_ids = [body_names.index(name) for name in LIMBS]
            # No pose spring; velocity damping stabilizes the suspended articulation.
            self.robot.articulation.set_kps([0.] * self.robot.num_dofs)
            self.robot.articulation.set_kds([50.] * self.robot.num_dofs)
            if gpu:
                self.world.init_gpu_system(cuda_device_id=0)
            self.force = np.zeros(3, dtype=np.float32)
            self.reset()
        except Exception:
            self.close()
            raise

    def positions(self):
        self.world.state.refresh()
        return self.world.state.get_body_pos(0)[0].detach().cpu().numpy().copy()

    def reset(self):
        if self.joints is not None:
            self.joints.set_enabled([0, 1, 2, 3], False)
        self.world.set_root_state(None, 0, [0., 0., 3.], [0., 0., 0., 1.],
                                  linear_velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.])
        self.world.set_dof_state(None, 0, np.zeros(self.robot.num_dofs, dtype=np.float32),
                                 velocities=np.zeros(self.robot.num_dofs, dtype=np.float32))
        self.world.step(substeps=0, apply_commands=False)
        self.anchors = self.positions()[self.link_ids].copy()
        frames = np.column_stack((self.anchors, np.tile([0., 0., 0., 1.], (4, 1))))
        if self.joints is None:
            # Identity link frames attach at each link's origin.
            self.joints = ke.physics.D6Batch.create_world_links(
                self.world.physics, [self.robot.articulation] * 4, self.link_ids,
                world_frames=frames,
                link_frames=np.tile([0., 0., 0., 0., 0., 0., 1.], (4, 1)))
        else:
            self.joints.set_world_frames([0, 1, 2, 3], frames)
        self.joints.set_enabled([0, 1], True)
        self.force[:] = 0

    def toggle(self, limb):
        enabled = not self.joints.enabled[limb]
        if enabled:
            # Local attachment is the link origin. Therefore its current world
            # position is the matching world frame; never reuse the old anchor.
            self.anchors[limb] = self.positions()[self.link_ids[limb]]
            self.joints.set_world_frames([limb], [[*self.anchors[limb], 0., 0., 0., 1.]])
        self.joints.set_enabled([limb], enabled)

    def step(self):
        if self.gpu:
            forces = self.world.get_gpu_articulation_link_forces()
            forces.zero_()
            forces[:, 0, :] = forces.new_tensor(self.force)
            self.world.apply_gpu_articulation_link_wrenches(forces=True, torques=False)
        else:
            self.world.set_body_force(None, 0, 0, self.force)
        self.world.step(refresh=False)

    def close(self):
        if self.joints is not None:
            self.joints.release()
        self.world.release()


class SuspensionViewer(ke.App):
    def __init__(self, gpu):
        super().__init__()
        self.gpu = gpu

    def setup(self):
        self.demo = Suspension(self.gpu)
        self.configure_timing(ke.SimulationTimingConfig(
            physics_hz=120., fixed_update_hz=120., render_hz=60.))
        self.set_simulation_hotkeys_enabled(True)
        self.materials = self.create_standard_materials()
        self.add_ground(material=self.materials.ground)
        self.set_camera_view([5., -6., 4.], [0., 0., 2.5])
        self.visual = ke.visual.sim.SimWorldVisualizer(self, self.demo.world)
        self.robot_visual = self.visual.add(self.demo.robot, self.demo.xml, path="/kw_d6",
                                            material=self.materials.common)

    def pre_update(self):
        for limb, key in enumerate((keys.J, keys.K, keys.N, keys.M)):
            if self.was_key_pressed(key):
                self.demo.toggle(limb)
        if self.was_key_pressed(keys.R):
            self.demo.reset()

    def fixed_update(self, fixed_dt):
        self.demo.step()

    def pre_render(self):
        self.visual.sync()
        positions = self.demo.positions()[self.demo.link_ids]
        colors = np.array([[0.1, 1., 0.2, 1.] if on else [0.5, 0.5, 0.5, 1.]
                           for on in self.demo.joints.enabled], dtype=np.float32)
        self.log_debug_points("/d6/anchors", self.demo.anchors, colors, 12.)
        self.log_debug_lines("/d6/errors", positions, self.demo.anchors, colors, 2.)

    def render(self):
        imgui.begin("KW D6 suspension")
        imgui.text("J/K: left/right hand   N/M: left/right foot")
        imgui.text("R: reset   Enter: play/pause   Space: step")
        imgui.text("Green: attached. Reattach captures the current link position.")
        for limb, name in enumerate(LIMBS):
            changed, _ = imgui.checkbox(name, self.demo.joints.enabled[limb])
            if changed:
                self.demo.toggle(limb)
        if imgui.button("Release all"):
            self.demo.joints.set_enabled([0, 1, 2, 3], False)
        imgui.same_line()
        if imgui.button("Reset (hands attached)"):
            self.demo.reset()
        imgui.separator()
        imgui.text("Anchors stay fixed until release and reattach.")
        imgui.text("Persistent root force (N)")
        for axis, label in enumerate("XYZ"):
            _, value = imgui.slider_float(f"Force {label}", float(self.demo.force[axis]), -300., 300.)
            self.demo.force[axis] = value
        if imgui.button("Clear force"):
            self.demo.force[:] = 0
        imgui.end()

    def cleanup(self):
        if hasattr(self, "visual"):
            self.visual.release()
        if hasattr(self, "demo"):
            self.demo.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.headless:
        demo = Suspension(args.gpu)
        try:
            demo.force[0] = 100.
            fixed_anchors = demo.anchors.copy()
            for _ in range(600):
                demo.step()
            np.testing.assert_array_equal(demo.anchors, fixed_anchors)
            assert np.isfinite(demo.positions()).all()
            np.testing.assert_allclose(demo.positions()[demo.link_ids[:2]], demo.anchors[:2], atol=0.04)
            for limb in range(4):
                demo.toggle(limb)
                demo.step()
                demo.toggle(limb)
            demo.toggle(2)
            demo.toggle(3)
            for _ in range(240):
                demo.step()
            assert np.isfinite(demo.positions()).all()
            np.testing.assert_allclose(demo.positions()[demo.link_ids], demo.anchors, atol=0.04)
            demo.toggle(2)
            demo.toggle(3)
            demo.joints.set_enabled([0, 1, 2, 3], False)
            old_positions = demo.positions()[demo.link_ids[:2]].copy()
            for _ in range(30):
                demo.step()
            current_positions = demo.positions()[demo.link_ids[:2]].copy()
            assert np.linalg.norm(current_positions - old_positions, axis=1).min() > 0.05
            demo.toggle(0)
            demo.toggle(1)
            np.testing.assert_allclose(demo.anchors[:2], current_positions, atol=1e-5)
            fixed_anchors = demo.anchors.copy()
            for _ in range(120):
                demo.step()
            np.testing.assert_array_equal(demo.anchors, fixed_anchors)
            assert np.isfinite(demo.positions()).all()
            np.testing.assert_allclose(demo.positions()[demo.link_ids[:2]], current_positions, atol=0.04)
            demo.reset()
            assert demo.joints.enabled == [True, True, False, False]
            print(f"KW D6 PASS gpu={args.gpu} links={dict(zip(LIMBS, demo.link_ids))}")
        finally:
            demo.close()
    else:
        app = SuspensionViewer(args.gpu)
        app.initialize(1440, 900, False, ke.UpAxis.Z)
        app.start()


if __name__ == "__main__":
    main()
