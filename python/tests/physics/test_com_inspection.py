"""Verify COM ownership, mass weighting, and screen-sized marker projection."""

import numpy as np
import torch
from unittest.mock import patch
import kangengine as ke

from kangengine.visual._com_display import COMDisplay, marker_lines


def test_body_com_rotates_local_offset_and_excludes_static():
    world = ke.physics.PhysicsWorld()
    world.add_static_box(half_extents=[1, 1, 1], pos=[9, 9, 9])
    body = world.create_dynamic_sphere(radius=0.5, pos=[1, 2, 3])
    body.set_local_com_pose([0.2, 0, 0, 0, 0, 0, 1])
    body.set_root_state(
        pos=[1, 2, 3],
        quat_xyzw=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
        linear_velocity=[0, 0, 0],
        angular_velocity=[0, 0, 0],
    )
    source = ke.physics.PhysXInspectionSource(world)
    positions, robots = source.com_positions()
    np.testing.assert_allclose(positions, [[1, 2.2, 3]], atol=1e-6)
    assert robots.shape == (0, 3)
    body.release()
    assert source.com_positions()[0].shape == (0, 3)


def test_robot_com_is_mass_weighted_per_articulation(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text(
        '<mujoco><worldbody><body name="root"><geom type="sphere" size=".1"/>'
        '<body name="child" pos="2 0 0"><joint type="hinge"/>'
        '<geom type="sphere" size=".1"/></body></body></worldbody></mujoco>'
    )
    world = ke.sim.KangSimWorld(num_envs=2, add_ground=False)
    try:
        data = world.load_mjcf(str(path))
        for env_id in range(2):
            robot = world.add_articulation(
                data,
                env_id=env_id,
                obj_id=0,
                config=ke.physics.ArticulationConfig.fixed_base(),
            ).articulation
            robot.reset_root(pos=[0, env_id * 4, 0])
            robot.set_link_masses([0, 1], [1, 3])
        source = ke.physics.PhysXInspectionSource(world.physics)
        bodies, robots = source.com_positions()
        assert len(bodies) == 4
        np.testing.assert_allclose(
            robots[np.argsort(robots[:, 1])], [[1.5, 0, 0], [1.5, 4, 0]], atol=1e-5
        )
        assert source.com_positions()[0] is bodies
        with patch.object(
            torch.Tensor,
            "index_add_",
            side_effect=AssertionError("Unused robot COM reduction"),
        ):
            body_only, hidden = source.com_positions(bodies=True, articulations=False)
        np.testing.assert_allclose(body_only, bodies)
        assert hidden.shape == (0, 3)
        hidden, robot_only = source.com_positions(bodies=False, articulations=True)
        assert hidden.shape == (0, 3)
        np.testing.assert_allclose(robot_only, robots)
        restored_bodies, restored_robots = source.com_positions()
        np.testing.assert_allclose(restored_bodies, bodies)
        np.testing.assert_allclose(restored_robots, robots)
        robot.set_link_masses([0, 1], [3, 1])
        source.clear_com_cache()
        _, robots = source.com_positions()
        np.testing.assert_allclose(
            robots[np.argsort(robots[:, 1])], [[1.5, 0, 0], [0.5, 4, 0]], atol=1e-5
        )
    finally:
        world.release()


def test_marker_pixel_radius_ignores_distance_and_clips_behind_camera():
    projection = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1.02, -0.202], [0, 0, -1, 0]], float
    )
    inverse = np.linalg.inv(projection)
    for depth, width, height in ((-1, 800, 600), (-5, 800, 600), (-5, 1200, 900)):
        starts, ends = marker_lines(
            np.array([[0, 0, depth]]), 6, projection, inverse, width, height
        )
        clip = np.column_stack((starts, np.ones(len(starts)))) @ projection.T
        pixels = clip[:, :2] / clip[:, 3:4] * [width / 2, height / 2]
        np.testing.assert_allclose(np.linalg.norm(pixels[:16], axis=1), 6, atol=1e-5)
    assert (
        len(marker_lines(np.array([[0, 0, 1]]), 6, projection, inverse, 800, 600)[0])
        == 0
    )


def test_com_displays_share_camera_inverse_and_reuse_marker_shape():
    from types import SimpleNamespace
    from test_physics_inspector import RecordingOverlay

    points = np.array([[0.0, 0.0, 0.0]])
    source = SimpleNamespace(com_positions=lambda **_: (points, points))
    app = SimpleNamespace(
        get_render_width=lambda: 800,
        get_render_height=lambda: 600,
        get_projection_matrix=lambda: np.eye(4),
        get_view_matrix=lambda: np.eye(4),
        debug_overlay=RecordingOverlay(),
    )
    display = COMDisplay(app)
    with (
        patch("numpy.linalg.inv", wraps=np.linalg.inv) as inverse,
        patch("numpy.cos", side_effect=AssertionError("Marker shape rebuilt")),
    ):
        display.sync(source, True, True)
        assert inverse.call_count == 1
        display.sync(source, True, True)
        assert inverse.call_count == 1
