"""Check Newton contact units, frames, optional forces, and lazy solver queries."""

import os
from unittest.mock import patch

import kangengine as ke
import newton
import numpy as np
import pytest
import torch
import warp as wp

from kangengine.adapters.newton import NewtonVisualizer


pytestmark = pytest.mark.skipif(
    os.environ.get("KANGENGINE_TEST_VIEWER") != "1", reason="Requires a display"
)


def _app():
    app = ke.App()
    app.initialize(
        width=128,
        height=128,
        headless=True,
        renderer=os.environ.get("KANGENGINE_TEST_RENDERER", "native"),
    )
    return app


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_contact_geometry_force_and_off(device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.5)
    model = builder.finalize(device=device)
    state = model.state()
    state.body_qd.assign(np.array([[2, 0, 0, 0, 0, 0]], np.float32))
    contacts = newton.Contacts(2, 0, device=device, requested_attributes={"force"})
    contacts.rigid_contact_count.assign([2])
    contacts.rigid_contact_shape0.assign([0, 0])
    contacts.rigid_contact_shape1.assign([1, 1])
    contacts.rigid_contact_point0.assign([[0, 0, 0], [1, 0, 0]])
    contacts.rigid_contact_point1.assign([[0, 0, -0.1], [1, 0, 0.1]])
    contacts.rigid_contact_normal.assign([[0, 0, 1], [0, 0, 1]])
    contacts.force.assign([[0, 0, -12, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    app = _app()
    visualizer = NewtonVisualizer(app)
    try:
        visualizer.set_model(model)
        inspector = app.physics_inspector
        source = inspector.source
        with patch.object(
            source, "contact_data", side_effect=AssertionError("OFF query")
        ):
            visualizer.log_state(state)
            visualizer.log_contacts(contacts, state)
            app.draw_frame()
        assert source._contact_buffers is None
        inspector.set_contacts_visible(True)
        values, _ = source.contact_data(velocities=True)
        rows = values.cpu().numpy() if isinstance(values, torch.Tensor) else values
        # Ground -> sphere: the support force and normal must point upward.
        np.testing.assert_allclose(rows[:, 3:6], [[0, 0, 1]] * 2)
        np.testing.assert_allclose(rows[:, 6:9], [[0, 0, 12], [0, 0, 0]])
        np.testing.assert_allclose(rows[:, 9], [-0.1, 0.1], atol=1e-6)
        np.testing.assert_allclose(rows[:, 10:13], [[2, 0, 0]] * 2)
        assert source.contact_data(velocities=True)[0] is values
        app.draw_frame()
        display = inspector._contact_display
        assert display.length == pytest.approx(0.25)  # Half the sphere radius.
        normal_matrices = display.batches["normal"]["matrices"].cpu().numpy()
        assert np.any(normal_matrices[0, :3, :3])
        assert normal_matrices[0, 2, 2] > 0
        assert not np.any(normal_matrices[1, :3, :3])
        force_matrices = display.batches["force"]["matrices"].cpu().numpy()
        np.testing.assert_allclose(
            force_matrices[0, 2, :3],
            [0, 0, 12 * display.length * display.force_gain],
            atol=1e-6,
        )

        # Legacy flags change visibility only when explicitly toggled.
        visualizer.show_contacts = True
        visualizer.log_contacts(contacts, state)
        assert inspector.contacts_visible
        inspector.set_contacts_visible(False)
        visualizer.log_contacts(contacts, state)
        assert not inspector.contacts_visible
        inspector.set_contacts_visible(True)
        visualizer.show_contacts = False
        visualizer.log_contacts(contacts, state)
        assert not inspector.contacts_visible
        inspector.set_contacts_visible(True)
        app.draw_frame()

        # Missing forces must not turn all candidates into an 'unloaded' display.
        contacts.force = None
        visualizer.log_state(state)
        visualizer.log_contacts(contacts, state)
        app.draw_frame()
        assert not source.supports_contact_forces
        assert not display.batches["force"]["active"]
        assert not display.batches["open"]["active"]
        visualizer.log_state(state)
        app.draw_frame()
        assert display.count == 0  # No contact submission for this state.
        inspector.set_contacts_visible(False)
        with patch.object(
            source, "contact_data", side_effect=AssertionError("OFF query")
        ):
            visualizer.log_state(state)
            visualizer.log_contacts(contacts, state)
            app.draw_frame()
    finally:
        visualizer.release()
        app.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mujoco_contact_force_and_lazy_update(device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.45), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.5)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=device == "cpu")
    state, next_state = model.state(), model.state()
    control = model.control()
    state.clear_forces()
    solver.step(state, next_state, control, None, 1 / 120)
    state, next_state = next_state, state
    app = _app()
    visualizer = NewtonVisualizer(app)
    try:
        visualizer.set_model(model)
        inspector = app.physics_inspector
        source = inspector.source
        with patch.object(
            solver, "update_contacts", side_effect=AssertionError("OFF solver query")
        ):
            visualizer.log_state(state)
            visualizer.log_mujoco_contacts(solver, state)
            app.draw_frame()
        inspector.set_contacts_visible(True)
        values, _ = source.contact_data(velocities=True)
        rows = values.cpu().numpy() if isinstance(values, torch.Tensor) else values
        assert len(rows) > 0
        assert np.isfinite(rows).all()
        assert np.linalg.norm(rows[:, 6:9], axis=1).max() > 0
        np.testing.assert_allclose(rows[:, 3:6], [[0, 0, 1]] * len(rows), atol=1e-6)
        assert np.all(rows[:, 8] > 0)
        assert np.all(np.sum(rows[:, 3:6] * rows[:, 6:9], axis=1) >= 0)
        app.draw_frame()
        inspector.set_contacts_visible(False)
        visualizer.show_contacts = True
        visualizer.log_mujoco_contacts(solver, state)
        assert inspector.contacts_visible
        app.draw_frame()
        buffers = source._contact_buffers
        visualizer.log_state(state)
        visualizer.log_mujoco_contacts(solver, state)
        app.draw_frame()
        assert source._contact_buffers is buffers
    finally:
        visualizer.release()
        app.close()
