"""Load and simulate an MJCF articulation with Newton and KangEngine.

Newton imports the MJCF and advances it with Newton's MuJoCo solver backend.
KangEngine renders Newton's visual/collision batches through the Newton
ViewerBase adapter. Use Shift + Left drag to apply a force to a Newton body or
articulation link.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kangengine.adapters.newton import ViewerKE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjcf",
        type=Path,
        help="MJCF XML path; defaults to Newton's bundled nv_humanoid.xml.",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument(
        "--max-worlds",
        type=int,
        default=0,
        help="Limit rendered worlds; zero renders every simulated world.",
    )
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--show-joints", action="store_true")
    parser.add_argument("--show-contacts", action="store_true")
    args = parser.parse_args()

    import newton
    import newton.examples
    import warp as wp

    newton.use_coord_layout_targets = True
    mjcf_path = (
        Path(newton.examples.get_asset("nv_humanoid.xml"))
        if args.mjcf is None
        else args.mjcf.expanduser().resolve()
    )
    if not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)
    if args.worlds < 1:
        raise ValueError("--worlds must be at least 1")
    if args.max_worlds < 0:
        raise ValueError("--max-worlds cannot be negative")

    articulation = newton.ModelBuilder()
    articulation.add_mjcf(
        str(mjcf_path),
        floating=True,
        ignore_names=["floor", "ground"],
        xform=wp.transform((0.0, 0.0, 1.5), wp.quat_identity()),
        enable_self_collisions=False,
    )

    builder = newton.ModelBuilder()
    builder.replicate(articulation, world_count=args.worlds)
    builder.add_ground_plane()

    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    substeps = 10
    # Native MuJoCo is substantially faster on CPU, but Newton only supports
    # its single-world mapping there. Multi-world models use MuJoCo Warp.
    use_mujoco_cpu = not wp.get_device().is_cuda and args.worlds == 1
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=use_mujoco_cpu,
    )

    viewer = ViewerKE(
        width=args.width,
        height=args.height,
        headless=args.headless,
    )
    viewer.show_ground = False
    viewer.app.scene.add_ground("/Ground", scale=20.0)
    viewer.set_model(
        model,
        max_worlds=None if args.max_worlds == 0 else args.max_worlds,
    )
    viewer.set_world_offsets((3.0, 3.0, 0.0))
    viewer.show_collision = bool(args.show_collision)
    viewer.show_joints = bool(args.show_joints)
    viewer.show_contacts = bool(args.show_contacts)
    viewer.set_camera(wp.vec3(5.0, -7.0, 3.0), pitch=-8.0, yaw=125.0)

    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / substeps
    sim_time = 0.0
    rendered_frames = 0

    print(
        f"Newton MJCF: {mjcf_path}\n"
        f"bodies={model.body_count} shapes={model.shape_count} "
        f"joints={model.joint_count} worlds={model.world_count}\n"
        f"solver=MuJoCo backend={'CPU' if use_mujoco_cpu else 'Warp'} "
        f"substeps={substeps}"
    )

    try:
        while viewer.is_running():
            if viewer.should_step():
                for _ in range(substeps):
                    state_0.clear_forces()
                    viewer.apply_forces(state_0)
                    solver.step(
                        state_0,
                        state_1,
                        control,
                        None,
                        sim_dt,
                    )
                    state_0, state_1 = state_1, state_0
                sim_time += frame_dt

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.log_mujoco_contacts(solver, state_0)
            viewer.end_frame()

            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
