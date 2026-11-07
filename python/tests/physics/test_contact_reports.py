"""Contact snapshots preserve per-body force direction and sleeping diagnostics."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

import kangengine as ke
from kangengine.utils.math import quat_xyzw_rotate


def vec(value):
    return np.array([value.x, value.y, value.z], dtype=np.float32)


def cube_part(offset=0.0):
    vertices = np.array([
        [x + offset, y, z]
        for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)
    ], dtype=np.float32)
    return ke.physics.ConvexMeshPart(vertices)


def viewer_state(viewer, **kwargs):
    class State(SimpleNamespace):
        pass

    for method in ("on_force_drag_begin", "on_force_drag_update", "on_force_drag_end",
                   "_drag_anchor_world", "_apply_mouse_force"):
        setattr(State, method, getattr(viewer, method))
    return State(**kwargs)


@pytest.fixture
def simulation():
    config = ke.physics.PhysicsConfig.y_up()
    config.enable_contact_reports = True
    world = ke.physics.PhysicsWorld(config)
    world.add_default_ground()
    actors = {
        "physx": world.create_dynamic_convex_compound([cube_part()], pos=(-2, 0.55, 0)),
        "decomposition": world.create_dynamic_convex_compound(
            [cube_part(), cube_part(1.1)], pos=(2, 0.55, 0)
        ),
    }
    return world, actors, config.dt


def test_body_contacts_sum_to_reported_force_and_survive_clear(simulation):
    world, actors, dt = simulation
    for _ in range(120):
        world.step()
        samples = {name: world.get_contacts(rigid=actor) for name, actor in actors.items()}
        if all(any(np.linalg.norm(vec(c.impulse)) > 0 for c in contacts)
               for contacts in samples.values()):
            break
    else:
        pytest.fail("both convex colliders must report ground contacts")

    assert len(world.get_contacts()) == sum(map(len, samples.values()))
    for name, contacts in samples.items():
        impulses = np.array([vec(contact.impulse) for contact in contacts])
        net_force = world.get_rigid_contact_force(actors[name])
        np.testing.assert_allclose(impulses.sum(axis=0) / dt, net_force, rtol=1e-5, atol=1e-5)
        assert net_force[1] > 0
        assert all(contact.normal.y > 0 for contact in contacts)
        assert all((contact.position.x < 0) == (name == "physx") for contact in contacts)

    positions = [[vec(c.position) for c in contacts] for contacts in samples.values()]
    world.clear_contacts()
    assert not world.get_contacts()
    for original, contacts in zip(positions, samples.values(), strict=True):
        np.testing.assert_array_equal([vec(c.position) for c in contacts], original)


def test_body_pair_reports_equal_opposite_impulses():
    config = ke.physics.PhysicsConfig.y_up()
    world = ke.physics.PhysicsWorld(config)
    left = world.create_dynamic_convex_compound([cube_part()], pos=(-0.4, 0, 0))
    right = world.create_dynamic_convex_compound([cube_part()], pos=(0.4, 0, 0))
    left.set_root_velocity([1, 0, 0], [0, 0, 0])
    right.set_root_velocity([-1, 0, 0], [0, 0, 0])
    world.step()
    left_contacts = world.get_contacts(rigid=left)
    right_contacts = world.get_contacts(rigid=right)
    assert left_contacts and len(left_contacts) == len(right_contacts)
    for a, b in zip(left_contacts, right_contacts, strict=True):
        np.testing.assert_array_equal(vec(a.position), vec(b.position))
        np.testing.assert_allclose(vec(a.impulse), -vec(b.impulse))
        np.testing.assert_allclose(vec(a.normal), -vec(b.normal))
    assert sum(c.impulse.x for c in left_contacts) < 0
    assert sum(c.impulse.x for c in right_contacts) > 0


def test_viewer_arrows_keep_sleep_samples_and_clear_on_wake_reset(simulation):
    world, actors, dt = simulation
    example = Path(__file__).resolve().parents[2] / "examples/compare_convex_collisions.py"
    viewer = runpy.run_path(str(example))["ConvexCollisionViewer"]
    lines, points = {}, {}
    state = viewer_state(viewer,
        physics=world, actors=actors, physics_dt=dt,
        contacts={name: [] for name in actors}, show_contacts=True,
        contact_colors={name: np.ones((1, 4), dtype=np.float32) for name in actors},
        contact_force_scale=0.1,
        debug=SimpleNamespace(overlay=ke.DebugOverlay(SimpleNamespace(
            log_debug_points=lambda path, values, colors, size, hidden, overlay:
            points.update({path: values}),
            clear_debug_points=lambda path: points.pop(path, None),
            clear_debug_lines=lambda path: lines.pop(path, None),
            log_debug_lines=lambda path, starts, ends, colors, width, hidden, overlay:
            lines.update({path: (starts, ends, {"overlay": overlay})}),
        ))),
        reset_rng=np.random.default_rng(0),
        initial_positions={"physx": [-2, 3, 0], "decomposition": [2, 3, 0]},
        _sync_visuals=lambda: None,
        _drag_actor=None,
    )
    for _ in range(300):
        viewer.fixed_update(state, dt)
        if all(actor.is_sleeping() for actor in actors.values()):
            break
    assert all(actor.is_sleeping() for actor in actors.values())
    # Advance again: the callback now emits nothing, while the viewer keeps samples.
    viewer.fixed_update(state, dt)
    assert not world.get_contacts()
    assert all(state.contacts.values())
    viewer._update_contact_forces(state)
    assert len(lines) == len(points) == 2
    for name, contacts in state.contacts.items():
        starts, ends, options = lines[f"/debug/contact_forces/{name}"]
        expected_forces = np.array([vec(c.impulse) for c in contacts]) / dt
        active = np.linalg.norm(expected_forces * state.contact_force_scale, axis=1) > 1e-8
        count = int(active.sum())
        assert count > 0
        np.testing.assert_allclose(
            ends[:count] - starts[:count],
            expected_forces[active] * state.contact_force_scale, rtol=1e-5, atol=1e-6,
        )
        assert options["overlay"] is True

    state.show_contacts = False
    viewer._update_contact_forces(state)
    assert not lines and not points
    state.show_contacts = True
    viewer._update_contact_forces(state)
    actors["decomposition"].set_root_state([2, 3, 0], [0, 0, 0, 1])
    viewer.fixed_update(state, dt)
    assert not state.contacts["decomposition"] and state.contacts["physx"]
    viewer._reset(state)
    assert not any(state.contacts.values())
    assert not lines and not points


@pytest.mark.parametrize("name", ["physx", "decomposition"])
@pytest.mark.parametrize("surface", ["source", "collision_0"])
def test_mouse_drag_applies_force_at_picked_point_only_during_steps(simulation, name, surface):
    world, actors, dt = simulation
    example = Path(__file__).resolve().parents[2] / "examples/compare_convex_collisions.py"
    viewer = runpy.run_path(str(example))["ConvexCollisionViewer"]
    cleared = []
    state = viewer_state(viewer,
        _drag_actor=None,
        _pick_actors={f"/{key}/{surface}": actor for key, actor in actors.items()},
        debug=SimpleNamespace(overlay=ke.DebugOverlay(SimpleNamespace(
            clear_debug_lines=cleared.append, clear_debug_points=cleared.append,
        ))),
        get_interaction_mode=lambda: ke.InteractionMode.FORCE,
    )
    actor = actors[name]
    other = actors["decomposition" if name == "physx" else "physx"]
    quat = [0, np.sin(0.3), 0, np.cos(0.3)]
    actor.set_root_state([0, 5, 0], quat)
    other.set_root_state([4, 5, 0], [0, 0, 0, 1])
    local_anchor = np.array([0, 0.5, 0.2], dtype=np.float32)
    hit = actor.get_root_position() + quat_xyzw_rotate(quat, local_anchor)
    pick = SimpleNamespace(
        hit=True, prim=SimpleNamespace(get_path=lambda: f"/{name}/{surface}"),
        position=ke.Vec3(*hit),
    )
    state.on_force_drag_begin(pick, ke.Vec3(*hit))
    np.testing.assert_allclose(state._drag_local_anchor, local_anchor, atol=1e-6)
    target = ke.Vec3(*(hit + [1, 1, 0]))
    for _ in range(5):
        state.on_force_drag_update(pick, target)
    # Mouse/render callbacks must not accumulate forces while physics is paused.
    world.step()
    assert actor.get_root_linear_velocity()[0] == pytest.approx(0)
    state._apply_mouse_force()
    world.step()
    assert actor.get_root_linear_velocity()[0] > 0
    assert np.linalg.norm(actor.get_root_angular_velocity()) > 0
    assert other.get_root_linear_velocity()[0] == pytest.approx(0)

    state.on_force_drag_end()
    assert state._drag_actor is None
    assert cleared[-2:] == ["/debug/mouse_force", "/debug/mouse_target"]
    actor.set_root_velocity([0, 0, 0], [0, 0, 0])
    state._apply_mouse_force()
    world.step()
    assert actor.get_root_linear_velocity()[0] == pytest.approx(0)

    state.on_force_drag_begin(
        SimpleNamespace(hit=True, prim=SimpleNamespace(get_path=lambda: "/ground")), target
    )
    assert state._drag_actor is None
