"""Simulate a volumetric Newton soft body

Newton owns the tetrahedral particles, contacts, and VBD solver. KangEngine
renders Newton's boundary triangles as one dynamic, double-sided PBR mesh.
The supported deformable render path is CPU-backed; CUDA can be used only with
explicit per-frame readback for debugging. Use the viewer panel to toggle
particles and particle constraints.
"""

# TODO : CUDA deformable geometry zero-copy rendering

from __future__ import annotations

import argparse

from kangengine.adapters.newton import ViewerKE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Newton simulation device (default: cpu).",
    )
    parser.add_argument(
        "--allow-cuda-readback",
        action="store_true",
        help="Allow slow CUDA-to-CPU readback of deformable vertices every rendered frame.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many rendered frames; zero runs until closed.",
    )
    parser.add_argument("--show-particles", action="store_true")
    parser.add_argument("--show-constraints", action="store_true")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not args.allow_cuda_readback:
        parser.error(
            "CUDA deformable rendering is not device-resident yet; use the "
            "default --device cpu, or pass --allow-cuda-readback for debugging."
        )

    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_soft_grid(
        pos=wp.vec3(-0.6, -0.2, 1.6),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=12,
        dim_y=4,
        dim_z=4,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1.0e3,
        k_mu=1.0e5,
        k_lambda=1.0e5,
        k_damp=1.0e-2,
        fix_left=True,
    )
    # VBD uses this graph coloring for conflict-free particle updates.
    builder.color()

    model = builder.finalize(device=args.device)
    model.soft_contact_ke = 1.0e2
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 1.0

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=10,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )

    viewer = ViewerKE(
        width=args.width,
        height=args.height,
        headless=args.headless,
        allow_cuda_readback=args.allow_cuda_readback,
    )
    viewer.show_ground = False
    viewer.app.scene.add_ground("/Ground", scale=20.0)
    viewer.set_model(model)
    viewer.show_triangles = True
    viewer.show_particles = bool(args.show_particles)
    viewer.show_springs = bool(args.show_constraints)
    viewer.show_contacts = False
    viewer.set_camera(wp.vec3(3.5, -4.5, 2.8), pitch=-12.0, yaw=130.0)

    frame_dt = 1.0 / 60.0
    substeps = 10
    sim_dt = frame_dt / substeps
    sim_time = 0.0
    rendered_frames = 0

    print(
        f"Newton soft body: particles={model.particle_count} "
        f"tetrahedra={model.tet_count} surface_triangles={model.tri_count}\n"
        f"solver=VBD iterations=10 substeps={substeps}"
    )

    try:
        while viewer.is_running():
            if viewer.should_step():
                for _ in range(substeps):
                    state_0.clear_forces()
                    collision_pipeline.collide(state_0, contacts)
                    solver.step(
                        state_0,
                        state_1,
                        control,
                        contacts,
                        sim_dt,
                    )
                    state_0, state_1 = state_1, state_0
                sim_time += frame_dt

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()

            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
