"""Verify KangEngine force dragging writes into Newton-owned body forces."""

from types import SimpleNamespace

import numpy as np

import kangengine as ke
from kangengine.adapters.newton import NewtonViewer


def main():
    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity())
    )
    builder.add_shape_sphere(body, radius=0.5)
    model = builder.finalize(device="cpu")
    state = model.state()

    viewer = NewtonViewer(width=320, height=240, headless=True)
    try:
        viewer.set_model(model)
        if not viewer.app.get_external_force_drag_enabled():
            raise RuntimeError("Newton CPU viewer did not opt into force drag")
        viewer.set_camera(wp.vec3(0.0, -5.0, 1.0), pitch=0.0, yaw=90.0)
        viewer.log_state(state)

        pick = SimpleNamespace(hit=True, position=ke.Vec3(0.0, 0.0, 1.0))
        viewer._begin_force_drag(pick, pick.position)
        if not viewer._picking.is_picking():
            raise RuntimeError("Newton Picking did not acquire the rigid body")

        viewer._update_force_drag(pick, ke.Vec3(0.5, 0.0, 1.0))
        state.clear_forces()
        viewer.apply_forces(state)
        forces = np.asarray(state.body_f.numpy(), dtype=np.float32)
        if not np.any(np.abs(forces) > 1.0e-6):
            raise RuntimeError("force dragging did not write Newton body_f")
        viewer.log_state(state)
        if "/model/picking_line" not in viewer._lines:
            raise RuntimeError("force dragging did not create its debug line")

        viewer._end_force_drag()
        if viewer._picking.is_picking():
            raise RuntimeError("Newton Picking did not release the rigid body")
        print("PASS: KangEngine force drag writes Newton body_f")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
