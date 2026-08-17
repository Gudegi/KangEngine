"""Run a Newton rigid-body simulation with KangEngine as its viewer.

Newton owns the model, contacts, solver, and state. KangEngine receives the
Newton ViewerBase calls and only handles rendering, input, and the window.
Use Shift + Left drag to apply a Newton-owned picking force.
"""

from __future__ import annotations

import argparse

import numpy as np

from kangengine.adapters.newton import ViewerKE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many rendered frames; zero runs until closed.",
    )
    args = parser.parse_args()

    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    sphere = builder.add_body(
        xform=wp.transform(p=wp.vec3(-2.0, -2.0, 2.0), q=wp.quat_identity()),
        label="sphere",
    )
    builder.add_shape_sphere(sphere, radius=0.5)

    capsule = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, -2.0, 2.0), q=wp.quat_identity()),
        label="capsule",
    )
    builder.add_shape_capsule(capsule, radius=0.3, half_height=0.7)

    box = builder.add_body(
        xform=wp.transform(p=wp.vec3(2.0, -2.0, 2.0), q=wp.quat_identity()),
        label="box",
    )
    builder.add_shape_box(box, hx=0.5, hy=0.35, hz=0.25)

    cylinder = builder.add_body(
        xform=wp.transform(p=wp.vec3(-2.0, 0.0, 2.0), q=wp.quat_identity()),
        label="cylinder",
    )
    builder.add_shape_cylinder(cylinder, radius=0.35, half_height=0.65)

    cone = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity()),
        label="cone",
    )
    builder.add_shape_cone(cone, radius=0.5, half_height=0.65)

    tetra_mesh = newton.Mesh(
        vertices=np.array(
            [
                [-0.6, -0.5, -0.4],
                [0.6, -0.5, -0.4],
                [0.0, 0.6, -0.4],
                [0.0, 0.0, 0.7],
            ],
            dtype=np.float32,
        ),
        indices=np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], dtype=np.int32),
    )
    mesh_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(2.0, 0.0, 2.0), q=wp.quat_identity()),
        label="mesh",
    )
    builder.add_shape_mesh(mesh_body, mesh=tetra_mesh)

    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=10)

    viewer = ViewerKE(
        width=args.width,
        height=args.height,
        headless=args.headless,
    )
    viewer.show_ground = False
    viewer.app.scene.add_ground("/Ground", scale=20.0)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(8.0, -8.0, 4.0), pitch=-10.0, yaw=135.0)

    frame_dt = 1.0 / 60.0
    substeps = 4
    sim_dt = frame_dt / substeps
    sim_time = 0.0
    rendered_frames = 0

    try:
        while viewer.is_running():
            if viewer.should_step():
                for _ in range(substeps):
                    state_0.clear_forces()
                    viewer.apply_forces(state_0)
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
            viewer.log_contacts(contacts, state_0)
            viewer.end_frame()
            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
