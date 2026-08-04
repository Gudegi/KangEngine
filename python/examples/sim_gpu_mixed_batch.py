"""Run rigid bodies and MJCF articulations together through KangSimWorld GPU APIs.

The default scene creates ten environments. Each environment contains one
dynamic ball and one fixed-base KW5 articulation. Articulations receive batched
random joint position targets while rigid bodies are launched and sparsely
reset through the same high-level ``env_ids`` API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kangengine as ke
import torch
from kangengine import imgui, keys


RIGID_OBJ_ID = 0
ROBOT_OBJ_ID = 1


def asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def identity_quaternions(count: int) -> torch.Tensor:
    rotations = torch.zeros((count, 4), dtype=torch.float32)
    rotations[:, 3] = 1.0
    return rotations


def print_gpu_state(world, step: int, cuda_device: int):
    rigid_state = world.get_gpu_rigid_data()
    joint_pos = world.get_gpu_articulation_joint_positions()
    joint_vel = world.get_gpu_articulation_joint_velocities()
    torch.cuda.synchronize(cuda_device)

    rigid_first = world.rigid_gpu_row(0, RIGID_OBJ_ID)
    rigid_last = world.rigid_gpu_row(world.num_envs - 1, RIGID_OBJ_ID)
    robot_rows = [
        world.articulation_gpu_row(env_id, ROBOT_OBJ_ID)
        for env_id in range(world.num_envs)
    ]

    first_pos = rigid_state[rigid_first, :3].detach().cpu().tolist()
    last_pos = rigid_state[rigid_last, :3].detach().cpu().tolist()
    qpos = joint_pos[robot_rows, :]
    qvel = joint_vel[robot_rows, :]
    max_joint = float(qpos.abs().max().item())
    mean_joint_vel = float(qvel.abs().mean().item())
    print(
        f"step {step:04d} | "
        f"ball[0]=[{first_pos[0]: .2f}, {first_pos[1]: .2f}, {first_pos[2]: .2f}] "
        f"ball[-1]=[{last_pos[0]: .2f}, {last_pos[1]: .2f}, {last_pos[2]: .2f}] "
        f"|q|_max={max_joint: .3f} "
        f"|dq|_mean={mean_joint_vel: .3f}"
    )


def create_simulation(num_envs: int, cuda_device: int):
    world = ke.sim.KangSimWorld(
        num_envs=num_envs,
        sim_device=f"cuda:{cuda_device}",
        sim_dt=1.0 / 120.0,
        add_ground=True,
    )
    try:
        ball_data = world.load_mjcf(asset_path("objects", "ball.xml"))
        robot_data = world.load_mjcf(
            asset_path("characters", "kw", "kw5.xml"), order="DFS"
        )
        robot_config = ke.physics.ArticulationConfig.free_base()
        cloner = ke.sim.GridCloner(
            world,
            spacing=(2.2, 3.0),
            columns=5,
        )
        zeros3 = torch.zeros((num_envs, 3), dtype=torch.float32)
        ball_velocity = zeros3.clone()
        ball_velocity[:, 1] = -1.0
        balls = cloner.add_rigid(
            ball_data,
            obj_id=RIGID_OBJ_ID,
            name="ball",
            density=600.0,
            initial_root_pos=(0.0, 1.0, 1.5),
            initial_linear_velocity=ball_velocity,
            initial_angular_velocity=zeros3,
        )
        robots = cloner.add_articulation(
            robot_data,
            obj_id=ROBOT_OBJ_ID,
            name="kw5",
            config=robot_config,
            initial_root_pos=(0.0, 0.0, 1.0),
            initial_linear_velocity=zeros3,
            initial_angular_velocity=zeros3,
        )
        origins = cloner.env_origins.cpu()
        rotations = identity_quaternions(num_envs)

        ball_positions = origins + torch.tensor([0.0, 1.0, 1.5], dtype=torch.float32)
        robot_positions = origins + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        robots.set_dof_state(
            None,
            torch.zeros((num_envs, robots.num_dofs), dtype=torch.float32),
        )
        configure_position_drives(robots, kp=100.0, kd=10.0)

        world.init_gpu_system(cuda_device_id=cuda_device)
        world.step(substeps=0, refresh=False, apply_commands=False)
        return (
            world,
            balls,
            robots,
            ball_positions,
            robot_positions,
            rotations,
            zeros3,
        )
    except Exception:
        world.release()
        raise


def create_cuda_commands(
    num_envs: int,
    num_dofs: int,
    cuda_device: int,
    seed: int,
    base_frequency: float,
):
    device = torch.device(f"cuda:{cuda_device}")
    targets = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)
    kp = torch.empty_like(targets)
    kd = torch.empty_like(targets)
    noise = torch.empty_like(targets)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    phase = torch.rand(
        (num_envs, num_dofs),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ) * (2.0 * torch.pi)
    frequency = torch.empty_like(targets)
    frequency.uniform_(
        base_frequency * 0.65,
        base_frequency * 1.35,
        generator=generator,
    )
    return device, targets, kp, kd, phase, frequency, noise, generator


def configure_position_drives(robots, kp: float, kd: float):
    for record in robots.records:
        record.articulation.set_kps([float(kp)] * robots.num_dofs)
        record.articulation.set_kds([float(kd)] * robots.num_dofs)


def update_random_position_targets(
    robots,
    targets,
    kp,
    kd,
    phase,
    frequency,
    noise,
    generator,
    sim_time: float,
    noise_scale: float,
    control_mode: str,
):
    targets.copy_(
        noise_scale * torch.sin(2.0 * torch.pi * frequency * float(sim_time) + phase)
    )
    noise.uniform_(-noise_scale * 0.08, noise_scale * 0.08, generator=generator)
    targets.add_(noise).clamp_(-noise_scale, noise_scale)
    if control_mode == "pos":
        robots.set_cmd(None, targets, mode=ke.sim.ControlMode.POS, kp=None, kd=None)
    elif control_mode == "pd-explicit":
        robots.set_cmd(
            None,
            targets,
            mode=ke.sim.ControlMode.PD_EXPLICIT,
            kp=kp,
            kd=kd,
        )
    else:
        raise ValueError(f"unsupported control mode: {control_mode}")


def step_with_random_position_control(
    world,
    robots,
    targets,
    kp,
    kd,
    phase,
    frequency,
    noise,
    generator,
    noise_scale: float,
    control_mode: str,
    substeps: int = 1,
):
    for _ in range(int(substeps)):
        update_random_position_targets(
            robots,
            targets,
            kp,
            kd,
            phase,
            frequency,
            noise,
            generator,
            world.sim_time,
            noise_scale,
            control_mode,
        )
        world.step(substeps=1, refresh=False)


def reset_scene(
    world,
    balls,
    robots,
    ball_positions,
    robot_positions,
    rotations,
    zeros3,
    targets=None,
):
    selected_envs = list(range(0, balls.num_envs))
    count = len(selected_envs)
    balls.set_root_state(
        selected_envs,
        ball_positions[selected_envs]
        + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        rotations[selected_envs],
        linear_velocity=torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).repeat(
            count, 1
        ),
        angular_velocity=zeros3[selected_envs],
    )
    robots.set_root_state(
        selected_envs,
        robot_positions[selected_envs],
        rotations[selected_envs],
        linear_velocity=zeros3[selected_envs],
        angular_velocity=zeros3[selected_envs],
    )
    robots.set_dof_state(
        selected_envs,
        torch.zeros((count, robots.num_dofs), dtype=torch.float32),
        velocities=torch.zeros((count, robots.num_dofs), dtype=torch.float32),
    )
    if targets is not None:
        targets.zero_()
    robots.clear_cmd(selected_envs)
    world.step(substeps=0, refresh=False, apply_commands=False)


def env_color(env_id: int, num_envs: int, warm: bool = False):
    t = env_id / max(1, num_envs - 1)
    if warm:
        return [0.95, 0.25 + 0.5 * t, 0.12, 1.0]
    return [0.15 + 0.65 * t, 0.45, 0.95 - 0.55 * t, 1.0]


def run_headless(args):
    (
        world,
        balls,
        robots,
        ball_positions,
        robot_positions,
        rotations,
        zeros3,
    ) = create_simulation(args.num_envs, args.cuda_device)
    try:
        (
            device,
            targets,
            pd_kp,
            pd_kd,
            command_phase,
            command_frequency,
            command_noise,
            command_rng,
        ) = create_cuda_commands(
            args.num_envs,
            robots.num_dofs,
            args.cuda_device,
            args.seed,
            args.motion_frequency,
        )
        print("KangSimWorld mixed PhysX GPU batch")
        print(f"  rigid bodies  : {balls.num_envs}")
        print(f"  articulations : {robots.num_envs}")
        print(f"  robot DOFs    : {robots.num_dofs}")
        pd_kp.fill_(float(args.pd_kp))
        pd_kd.fill_(float(args.pd_kd))

        print(
            f"  Torch cmd     : {device} ({args.control_mode}, continuous randomized targets)"
        )

        reset_step = args.steps // 2

        for step in range(args.steps):
            if step == reset_step:
                reset_scene(
                    world,
                    balls,
                    robots,
                    ball_positions,
                    robot_positions,
                    rotations,
                    zeros3,
                    targets,
                )

            step_with_random_position_control(
                world,
                robots,
                targets,
                pd_kp,
                pd_kd,
                command_phase,
                command_frequency,
                command_noise,
                command_rng,
                args.joint_noise,
                args.control_mode,
                substeps=args.substeps,
            )
            if step % args.report_every == 0 or step == args.steps - 1:
                print_gpu_state(world, step, args.cuda_device)
    finally:
        world.release()


class MixedGpuBatchViewer(ke.App):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self):
        self.standard_materials = self.create_standard_materials()
        self.add_ground(scale=16.0, material=self.standard_materials.ground)
        self.set_camera_view([6.0, -9.0, 5.5], [4.0, 1.5, 0.8])

        (
            self.world,
            self.balls,
            self.robots,
            self.ball_positions,
            self.robot_positions,
            self.rotations,
            self.zeros3,
        ) = create_simulation(self.args.num_envs, self.args.cuda_device)
        self.timing = self.configure_timing(
            ke.SimulationTimingConfig.from_dt(
                physics_dt=self.world.sim_dt,
                fixed_dt=self.world.sim_dt * self.args.substeps,
                render_hz=60.0,
            )
        )
        self.set_simulation_hotkeys_enabled(True)
        (
            self.device,
            self.targets,
            self.pd_kp,
            self.pd_kd,
            self.command_phase,
            self.command_frequency,
            self.command_noise,
            self.command_rng,
        ) = create_cuda_commands(
            self.args.num_envs,
            self.robots.num_dofs,
            self.args.cuda_device,
            self.args.seed,
            self.args.motion_frequency,
        )
        self.pd_kp.fill_(float(self.args.pd_kp))
        self.pd_kd.fill_(float(self.args.pd_kd))

        self.visual = ke.visual.sim.SimWorldVisualizer(self, self.world)
        self.ball_visual = self.visual.add(
            self.balls,
            asset_path("objects", "ball.xml"),
            prim_base_path="/gpu_batch/balls",
            material=self.standard_materials.common,
            color=[
                env_color(env_id, self.args.num_envs, warm=True)
                for env_id in range(self.args.num_envs)
            ],
        )
        self.robot_visual = self.visual.add(
            self.robots,
            asset_path("characters", "kw", "kw5.xml"),
            prim_base_path="/gpu_batch/robots",
            material=self.standard_materials.common,
            color=[
                env_color(env_id, self.args.num_envs)
                for env_id in range(self.args.num_envs)
            ],
        )
        self.check_error()

    def pre_update(self):
        if self.was_key_pressed(keys.R):
            reset_scene(
                self.world,
                self.balls,
                self.robots,
                self.ball_positions,
                self.robot_positions,
                self.rotations,
                self.zeros3,
                self.targets,
            )

    def fixed_update(self, fixed_dt):
        step_with_random_position_control(
            self.world,
            self.robots,
            self.targets,
            self.pd_kp,
            self.pd_kd,
            self.command_phase,
            self.command_frequency,
            self.command_noise,
            self.command_rng,
            self.args.joint_noise,
            self.args.control_mode,
            substeps=self.args.substeps,
        )

    def pre_render(self):
        self.visual.sync()
        self.check_error()

    def render(self):
        imgui.begin("Mixed GPU Simulation")
        state = "paused" if self.is_simulation_paused() else "running"
        imgui.text(f"State: {state}")
        imgui.text(f"Rigid instances: {self.balls.num_envs}")
        imgui.text(f"Articulation instances: {self.robots.num_envs}")
        imgui.text(f"Links per articulation: {self.robots.num_bodies}")
        imgui.text(f"PhysX GPU -> CUDA {self.args.control_mode} -> ExternalBuffer")
        imgui.separator()
        imgui.text("Enter: play/pause    Space: pause/step    R: reset scene")
        imgui.end()

    def cleanup(self):
        if hasattr(self, "visual"):
            self.visual.release()
        if hasattr(self, "world"):
            self.world.release()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=60)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--joint-noise", type=float, default=0.6)
    parser.add_argument("--motion-frequency", type=float, default=1.2)
    parser.add_argument("--pd-kp", type=float, default=60.0)
    parser.add_argument("--pd-kd", type=float, default=6.0)
    parser.add_argument(
        "--control-mode",
        choices=("pd-explicit", "pos"),
        default="pd-explicit",
        help="Articulation command path to test.",
    )
    parser.add_argument("--seed", type=int, default=7)
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
    if args.joint_noise < 0.0:
        parser.error("--joint-noise must be non-negative")
    if args.motion_frequency <= 0.0:
        parser.error("--motion-frequency must be positive")
    if args.pd_kp < 0.0:
        parser.error("--pd-kp must be non-negative")
    if args.pd_kd < 0.0:
        parser.error("--pd-kd must be non-negative")
    return args


def main():
    args = parse_args()
    if args.viewer:
        app = MixedGpuBatchViewer(args)
        app.initialize(args.width, args.height, False, ke.UpAxis.Z)
        app.start()
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
