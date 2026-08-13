"""Verify Newton CPU particle and cloth-surface visualization."""

from kangengine.adapters.newton import NewtonViewer


def main():
    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.5,
        cell_y=0.5,
        mass=1.0,
        add_springs=True,
    )
    builder.add_soft_mesh(
        pos=wp.vec3(2.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        scale=0.5,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
        ],
        indices=[0, 1, 2, 3],
        density=100.0,
        k_mu=1_000.0,
        k_lambda=1_000.0,
        k_damp=1.0,
    )
    model = builder.finalize()
    state = model.state()

    viewer = NewtonViewer(width=320, height=240, headless=True)
    try:
        viewer.show_particles = True
        viewer.show_triangles = True
        viewer.show_springs = True
        viewer.set_model(model)
        viewer.log_state(state)
        viewer.end_frame()
        viewer.log_state(state)
        viewer.end_frame()
        if "/model/particles" not in viewer._points:
            raise RuntimeError("Newton particles were not registered")
        if "/model/triangles" not in viewer._dynamic_meshes:
            raise RuntimeError("Newton cloth surface was not registered")
        if "/model/particle_constraints" not in viewer._lines:
            raise RuntimeError("Newton particle constraints were not registered")
        print("PASS: Newton particles, cloth, and soft surfaces render")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
