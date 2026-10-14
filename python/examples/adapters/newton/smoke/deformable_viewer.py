"""Verify Newton CPU particle and cloth-surface visualization."""

from kangengine.adapters.newton import ViewerKE
import kangengine as ke
import numpy as np


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
    model = builder.finalize(device="cpu")
    state = model.state()

    viewer = ViewerKE(width=320, height=240, headless=True)
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

        colors = np.zeros((model.tri_count, 3), dtype=np.float32)
        colors[:model.tri_count // 2, 0] = 1.0
        colors[model.tri_count // 2:, 2] = 1.0
        model.tri_color.assign(colors)
        model.tri_opacity.assign(np.full(model.tri_count, 0.4, dtype=np.float32))
        viewer.log_state(state)
        viewer.end_frame()
        visible = [batch for batch in viewer._dynamic_meshes.values()
                   if batch.view.component.visible]
        assert len(visible) == 2
        assert not viewer._dynamic_meshes["/model/triangles"].view.component.visible
        for batch in visible:
            assert batch.view.component.alpha_mode == ke.render.AlphaMode.BLEND
            assert abs(batch.view.get_base_color().w - 0.4) < 1e-6

        # Moving live vertices must not change the registered source mesh.
        source = visible[0].view.prim.get_mesh_component().mesh_data
        original_x = source.vertices[0].x
        cached_colors = [batch.colors for batch in visible]
        points = state.particle_q.numpy().copy()
        points[:, 0] += 0.1
        state.particle_q.assign(points)
        viewer.log_state(state)
        viewer.end_frame()
        assert source.vertices[0].x == original_x
        assert all(batch.colors is cached for batch, cached in zip(visible, cached_colors))
        print("PASS: Newton particles, cloth, and soft surfaces render")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
