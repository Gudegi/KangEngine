"""Validate contact units, point velocities, and diagnostic glyph transforms."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import kangengine as ke

from kangengine.visual._contact_display import ContactDisplay, contact_modes, glyph_matrices


def test_modes_use_tangential_speed_and_loaded_contact():
    normals = torch.tensor([[0., 1., 0.]] * 4)
    forces = torch.tensor([[0., 0., 0.], [0., 5., 0.], [0., 5., 0.], [0., 5., 0.]])
    velocities = torch.tensor([[3., 0., 0.], [0., 0., 0.], [2., 0., 0.], [0., 4., 0.]])
    assert contact_modes(normals, forces, velocities).tolist() == [0, 1, 2, 1]


def test_glyph_transforms_reach_surface_tip_and_collapse_unused_slots():
    matrices = torch.empty((4, 4, 4))
    points = torch.tensor([[1., 2., 3.], [0., 0., 0.]])
    directions = torch.tensor([[0., 1., 0.], [0., 0., 0.]])
    glyph_matrices(matrices, points, directions, torch.tensor([2., 0.]), .1)
    np.testing.assert_allclose(matrices[0].T @ torch.tensor([0., 0., 1., 1.]), [1, 4, 3, 1])
    assert torch.isfinite(matrices).all()
    assert torch.count_nonzero(matrices[1:, :3, :3]) == 0


def test_display_hides_separated_unloaded_points_in_every_glyph_batch():
    data = torch.zeros((4, 13))
    data[:, 4] = 1
    data[:, 9] = torch.tensor([.01, 0, -.01, .01])
    data[3, 7] = 1
    source = SimpleNamespace(contact_revision=1, contact_data=lambda **_: (data, 1.0))
    display = ContactDisplay(None, "/contacts")
    batches = {}

    def batch(name, count, device):
        return batches.setdefault(name, {"matrices": torch.empty((count, 4, 4), device=device)})

    with patch.object(display, "_batch", side_effect=batch), patch.object(display, "_submit"):
        display.sync(source)

    for name, expected in (
        ("normal", [False, True, True, True]),
        ("force", [False, False, False, True]),
        ("open", [False, True, True, False]),
        ("stick", [False, False, False, True]),
        ("slip", [False, False, False, False]),
    ):
        shown = batches[name]["matrices"][:, :3, :3].abs().sum(dim=(1, 2)) > 0
        assert shown.tolist() == expected, name


def test_cpu_contacts_use_last_step_dt_and_cache_until_clear():
    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.y_up())
    world.add_default_ground()
    body = world.create_dynamic_sphere(radius=.5, pos=[0, .6, 0])
    source = ke.physics.PhysXInspectionSource(world)
    assert source.supports_contacts
    length, gain = source.contact_display_settings()
    assert length > 0 and gain > 0
    for _ in range(30):
        world.step()
        if world.num_contacts():
            break
    data, dt = source.contact_data(velocities=True)
    assert len(data) > 0 and dt > 0
    assert np.linalg.norm(data[:, 6:9]) > 0
    assert np.isfinite(data).all()
    np.testing.assert_allclose(np.linalg.norm(data[:, 3:6], axis=1), 1, atol=1e-5)
    world.set_dt(dt * 2)
    assert source.contact_data(velocities=True)[1] == dt
    assert source.contact_data(velocities=True)[0] is data
    world.clear_contacts()
    assert len(source.contact_data(velocities=True)[0]) == 0
    assert body.get_root_position()[1] > 0


def test_contact_reports_disabled_is_explicit():
    config = ke.physics.PhysicsConfig.y_up()
    config.enable_contact_reports = False
    source = ke.physics.PhysXInspectionSource(ke.physics.PhysicsWorld(config))
    assert not source.supports_contacts
    assert "enable_contact_reports" in source.contacts_unavailable_reason
