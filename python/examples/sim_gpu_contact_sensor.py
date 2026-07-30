"""High-level PhysX GPU rigid batch with CUDA contact/force sensors.

This example keeps the public path small:

- create a batched ``KangSimWorld`` on CUDA,
- spawn two rigid views across many environments,
- attach ``ContactSensor`` and ``ForceSensor`` to the views,
- read the sensor outputs as Torch CUDA tensors,
- optionally draw both rigid batches through GPU ExternalBuffer visuals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
import numpy as np
import torch
from kangengine import imgui, keys


LEFT_OBJ_ID = 0
RIGHT_OBJ_ID = 1


def asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def make_env_origins(num_envs: int, device) -> torch.Tensor:
    env_ids = torch.arange(num_envs, dtype=torch.float32, device=device)
    columns = min(4, max(1, num_envs))
    x = torch.remainder(env_ids, float(columns)) * 1.6
    y = torch.div(env_ids, columns, rounding_mode="floor") * 1.8
    return torch.stack((x, y, torch.zeros_like(x)), dim=1)


def make_identity_quaternions(count: int, device) -> torch.Tensor:
    rotations = torch.zeros((count, 4), dtype=torch.float32, device=device)
    rotations[:, 3] = 1.0
    return rotations


def env_group_colors(num_envs: int):
    palette = (
        (0.18, 0.52, 0.92, 1.0),
        (0.92, 0.42, 0.18, 1.0),
        (0.42, 0.72, 0.28, 1.0),
        (0.74, 0.36, 0.82, 1.0),
        (0.88, 0.72, 0.22, 1.0),
        (0.25, 0.68, 0.68, 1.0),
        (0.86, 0.38, 0.58, 1.0),
        (0.52, 0.52, 0.58, 1.0),
    )
    colors = []
    for env_id in range(num_envs):
        colors.append(list(palette[env_id % len(palette)]))
    return colors


def contact_asset(args) -> str:
    filename = "ball.xml" if args.shape == "sphere" else "box.xml"
    return asset_path("objects", filename)


def initial_x_offset(args) -> float:
    # ball.xml radius is 0.12, box.xml x half-extent is 0.18.
    return 0.135 if args.shape == "sphere" else 0.205


class GpuContactSensorDemo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.cuda_device}")
        self.world = None
        self.peak_impulse = 0.0
        self.peak_force = 0.0

    def setup(self):
        args = self.args
        physics_config = ke.physics.PhysicsConfig.z_up()
        physics_config.enable_contact_reports = False  # Turn off CPU contact report callback
        physics_config.restitution = 0.3
        self.world = ke.sim.KangSimWorld(
            num_envs=args.num_envs,
            physics_config=physics_config,
            sim_device=self.device,
            sim_dt=1.0 / 120.0,
            add_ground=True,
        )
        rigid_xml = contact_asset(args)
        rigid_data = self.world.load_mjcf(rigid_xml)

        for env_id in range(args.num_envs):
            self.world.add_rigid(
                rigid_data,
                env_id=env_id,
                obj_id=LEFT_OBJ_ID,
                name=f"left_{args.shape}",
                density=200.0,
            )
            self.world.add_rigid(
                rigid_data,
                env_id=env_id,
                obj_id=RIGHT_OBJ_ID,
                name=f"right_{args.shape}",
                density=200.0,
            )

        self.left = self.world.get_rigid_batch(obj_id=LEFT_OBJ_ID)
        self.right = self.world.get_rigid_batch(obj_id=RIGHT_OBJ_ID)
        self.left_contact = self.left.add_contact_sensor(
            body_ids=[0], name="left_contact"
        )
        self.left_force = self.left.add_force_sensor(body_ids=[0], name="left_force")
        self.right_contact = self.right.add_contact_sensor(
            body_ids=[0], name="right_contact"
        )

        self._build_reset_tensors()
        self.world.init_gpu_system(cuda_device_id=args.cuda_device)
        self.reset()
        return self

    def _build_reset_tensors(self):
        args = self.args
        origins = make_env_origins(args.num_envs, self.device)
        self.rotations = make_identity_quaternions(args.num_envs, self.device)
        self.zeros3 = torch.zeros((args.num_envs, 3), dtype=torch.float32, device=self.device)

        offset = initial_x_offset(self.args)
        self.left_pos = origins + torch.tensor(
            [-offset, 0.0, 1.0], dtype=torch.float32, device=self.device
        )
        self.right_pos = origins + torch.tensor(
            [offset, 0.0, 1.0], dtype=torch.float32, device=self.device
        )
        self.left_vel = self.zeros3.clone()
        self.right_vel = self.zeros3.clone()
        self.left_vel[:, 0] = float(self.args.speed)
        self.right_vel[:, 0] = -float(self.args.speed)

    def reset(self):
        self.left.set_root_state(
            None,
            self.left_pos,
            self.rotations,
            linear_velocity=self.left_vel,
            angular_velocity=self.zeros3,
        )
        self.right.set_root_state(
            None,
            self.right_pos,
            self.rotations,
            linear_velocity=self.right_vel,
            angular_velocity=self.zeros3,
        )
        self.world.step(substeps=0, refresh=False, apply_commands=False)
        self.peak_impulse = 0.0
        self.peak_force = 0.0

    def step(self, substeps: int):
        self.world.step(substeps=substeps, refresh=False)
        self._update_peak_metrics()
        return self.left_contact.data

    def _current_metrics(self):
        impulse = self.left_contact.net_impulse[:, 0]
        force = self.left_force.force[:, 0]
        max_impulse = float(torch.linalg.vector_norm(impulse, dim=1).max().item())
        max_force = float(torch.linalg.vector_norm(force, dim=1).max().item())
        return max_impulse, max_force

    def _update_peak_metrics(self):
        max_impulse, max_force = self._current_metrics()
        self.peak_impulse = max(self.peak_impulse, max_impulse)
        self.peak_force = max(self.peak_force, max_force)
        return max_impulse, max_force

    def report(self, step: int):
        torch.cuda.synchronize(self.device)
        counts = self.left_contact.contact_count[:, 0]
        hit_count = int(torch.count_nonzero(counts).item())
        max_impulse, max_force = self._current_metrics()
        print(
            f"step {step:04d} | contacts={hit_count}/{self.args.num_envs} "
            f"impulse={max_impulse:.6f} force={max_force:.2f} "
            f"peak_impulse={self.peak_impulse:.6f} peak_force={self.peak_force:.2f}"
        )

    def release(self):
        if self.world is not None:
            self.world.release()
            self.world = None


def run_headless(args):
    demo = GpuContactSensorDemo(args).setup()
    try:
        print("KangSimWorld high-level GPU contact sensor example")
        print(f"  envs       : {args.num_envs}")
        print(f"  device     : cuda:{args.cuda_device}")
        print("  tensors    : ContactSensor/ForceSensor outputs stay on CUDA")
        for step in range(args.steps):
            if args.reset_every and step and step % args.reset_every == 0:
                demo.reset()
                demo.report(step)
                continue
            demo.step(args.substeps)
            if step % args.report_every == 0 or step == args.steps - 1:
                demo.report(step)
    finally:
        demo.release()


class GpuContactSensorViewer(ke.App):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self):
        self._skip_fixed_updates_this_frame = False
        self.show_contact_debug = True
        self.contact_marker_view = None
        self.force_arrow_view = None
        self.demo = GpuContactSensorDemo(self.args).setup()
        self.timing = self.configure_timing(
            ke.SimulationTimingConfig.from_dt(
                physics_dt=self.demo.world.sim_dt,
                fixed_dt=self.demo.world.sim_dt * self.args.substeps,
                render_hz=60.0,
            )
        )
        self.set_simulation_hotkeys_enabled(True)
        self.shaders = self.create_standard_shaders()
        self.add_ground(scale=16.0, shader=self.shaders.ground)
        self.set_camera_view([3.5, -5.5, 3.4], [2.0, 1.2, 0.8])

        self.visual = ke.visual.sim.SimWorldVisualizer(self, self.demo.world)
        rigid_xml = contact_asset(self.args)
        group_colors = env_group_colors(self.args.num_envs)
        self.visual.add(
            self.demo.left,
            rigid_xml,
            prim_base_path="/gpu_contact/left",
            shader=self.shaders.common,
            color=group_colors,
        )
        self.visual.add(
            self.demo.right,
            rigid_xml,
            prim_base_path="/gpu_contact/right",
            shader=self.shaders.common,
            color=group_colors,
        )
        self.check_error()

    def preUpdate(self):
        if self.was_key_pressed(keys.C):
            self.show_contact_debug = not self.show_contact_debug
        if self.was_key_pressed(keys.R):
            self.demo.reset()
            self._skip_fixed_updates_this_frame = True

    def fixedUpdate(self, fixed_dt):
        if not self._skip_fixed_updates_this_frame:
            self.demo.step(self.args.substeps)

    def preRender(self):
        self.visual.sync()
        self._update_contact_debug()
        self._skip_fixed_updates_this_frame = False
        self.check_error()

    def _clear_contact_debug(self):
        empty3 = np.zeros((0, 3), dtype=np.float32)
        empty4 = np.zeros((0, 4), dtype=np.float32)
        if self.contact_marker_view is not None:
            self.contact_marker_view.update_lines(empty3, empty3, empty4)
        if self.force_arrow_view is not None:
            self.force_arrow_view.update_arrows(empty3, empty3, empty4)

    def _update_contact_debug(self):
        if not self.show_contact_debug:
            self._clear_contact_debug()
            return

        points = self._contact_points_cpu()
        if points.size == 0:
            self._clear_contact_debug()
            return
        # 0:3  = contact position xyz
        # 3:6  = contact normal xyz
        # 6:9  = normal impulse vector xyz
        # 9    = separation
        positions = points[:, 0:3]
        impulses = points[:, 6:9]
        self._update_contact_markers(positions)
        self._update_force_arrows(positions, impulses)

    def _contact_points_cpu(self):
        gpu_system = self.demo.world.gpu_system
        count_tensor = torch.as_tensor(
            gpu_system.contact_point_count(), device=self.demo.device
        )
        count = int(count_tensor[0].item())
        if count <= 0:
            return np.zeros((0, 10), dtype=np.float32)
        points = torch.as_tensor(gpu_system.contact_points(), device=self.demo.device)
        count = min(count, int(points.shape[0]), int(self.args.max_debug_contacts))
        return points[:count].cpu().numpy().astype(np.float32, copy=False)

    def _update_contact_markers(self, positions):
        half = float(self.args.contact_marker_size) * 0.5
        offsets = np.array(
            (
                (half, 0.0, 0.0),
                (0.0, half, 0.0),
                (0.0, 0.0, half),
            ),
            dtype=np.float32,
        )
        starts = []
        ends = []
        for pos in positions:
            for offset in offsets:
                starts.append(pos - offset)
                ends.append(pos + offset)
        starts = np.asarray(starts, dtype=np.float32)
        ends = np.asarray(ends, dtype=np.float32)
        colors = np.repeat(
            np.array([[0.05, 0.95, 1.0, 1.0]], dtype=np.float32),
            starts.shape[0],
            axis=0,
        )
        if self.contact_marker_view is None:
            self.contact_marker_view = self.scene.log_lines(
                "/debug/gpu_contact_points",
                self.shaders.common,
                starts,
                ends,
                colors,
                0.006,
                8,
            )
        else:
            self.contact_marker_view.update_lines(starts, ends, colors)

    def _update_force_arrows(self, positions, impulses):
        dt = max(float(self.demo.world.sim_dt), 1e-8)
        forces = impulses / dt
        norms = np.linalg.norm(forces, axis=1)
        active = norms > float(self.args.force_threshold)
        if not np.any(active):
            empty3 = np.zeros((0, 3), dtype=np.float32)
            empty4 = np.zeros((0, 4), dtype=np.float32)
            if self.force_arrow_view is not None:
                self.force_arrow_view.update_arrows(empty3, empty3, empty4)
            return

        starts = positions[active].astype(np.float32, copy=False)
        ends = (
            starts + forces[active] * float(self.args.force_arrow_scale)
        ).astype(np.float32, copy=False)
        colors = np.repeat(
            np.array([[1.0, 0.86, 0.05, 1.0]], dtype=np.float32),
            starts.shape[0],
            axis=0,
        )
        if self.force_arrow_view is None:
            self.force_arrow_view = self.scene.log_arrows(
                "/debug/gpu_contact_forces",
                self.shaders.common,
                starts,
                ends,
                colors,
                0.018,
                12,
            )
        else:
            self.force_arrow_view.update_arrows(starts, ends, colors)

    def render(self):
        counts = self.demo.left_contact.contact_count[:, 0]
        force = self.demo.left_force.force[:, 0]
        hit_count = int(torch.count_nonzero(counts).item())
        max_force = float(torch.linalg.vector_norm(force, dim=1).max().item())

        imgui.begin("GPU Contact Sensor")
        state = "paused" if self.is_simulation_paused() else "running"
        imgui.text(f"State: {state}")
        imgui.text(f"Envs: {self.args.num_envs}")
        imgui.text(f"Contacts: {hit_count}/{self.args.num_envs}")
        imgui.text(f"Max normal force: {max_force:.2f}")
        imgui.text(f"Peak normal force: {self.demo.peak_force:.2f}")
        imgui.text(f"Contact debug: {'on' if self.show_contact_debug else 'off'}")
        imgui.separator()
        imgui.text(
            "Enter: play/pause    Space: pause/step    "
            "R: reset    C: contact debug"
        )
        imgui.end()

    def cleanup(self):
        if hasattr(self, "visual"):
            self.visual.release()
        if hasattr(self, "demo"):
            self.demo.release()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=30)
    parser.add_argument("--reset-every", type=int, default=0)
    parser.add_argument("--shape", choices=("box", "sphere"), default="box")
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--max-debug-contacts", type=int, default=128)
    parser.add_argument("--contact-marker-size", type=float, default=0.08)
    parser.add_argument("--force-arrow-scale", type=float, default=0.001)
    parser.add_argument("--force-threshold", type=float, default=1e-4)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.substeps < 1:
        parser.error("--substeps must be positive")
    if args.report_every < 1:
        parser.error("--report-every must be positive")
    if args.reset_every < 0:
        parser.error("--reset-every must be non-negative")
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.max_debug_contacts < 1:
        parser.error("--max-debug-contacts must be positive")
    if args.contact_marker_size <= 0.0:
        parser.error("--contact-marker-size must be positive")
    if args.force_arrow_scale < 0.0:
        parser.error("--force-arrow-scale must be non-negative")
    if args.force_threshold < 0.0:
        parser.error("--force-threshold must be non-negative")
    return args


def main():
    args = parse_args()
    if args.viewer:
        app = GpuContactSensorViewer(args)
        app.initialize(args.width, args.height, False, ke.UpAxis.Z)
        app.start()
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
