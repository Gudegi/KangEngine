"""Check body and joint frame composition, caching, and display lifecycle."""

import numpy as np
from unittest.mock import patch
import kangengine as ke

from test_physics_inspector import RecordingApp


def test_frames_ignore_local_com_and_release_without_polling():
    world = ke.physics.PhysicsWorld()
    world.add_static_box(half_extents=[1, 1, 1], pos=[8, 0, 0])
    body = world.create_dynamic_sphere(radius=0.2, pos=[1, 2, 3])
    body.set_local_com_pose([0.3, 0, 0, 0, 0, 0, 1])
    body.set_root_state(
        pos=[1, 2, 3],
        quat_xyzw=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
        linear_velocity=[0, 0, 0],
        angular_velocity=[0, 0, 0],
    )
    app = RecordingApp()
    inspector = ke.visual.PhysicsInspector(app, world=world)
    source = inspector.source
    assert source._frame_metadata is None and app.sync_callback is None
    inspector.set_frames_visible(bodies=True, joints=True)
    inspector.sync()
    poses, joints = source.frame_poses()
    assert poses.shape == (1, 7) and joints.shape == (0, 7)
    np.testing.assert_allclose(poses[0, :3], [1, 2, 3])
    path = "/__debug__/physics_frames/body"
    endpoints = app.debug_overlay.lines_by_path[path][3:]
    np.testing.assert_allclose(
        endpoints, [[1, 2.12, 3], [0.88, 2, 3], [1, 2, 3.12]], atol=1e-6
    )
    uploads = app.debug_overlay.uploads
    inspector.sync()
    assert app.debug_overlay.uploads == uploads
    assert source.frame_poses()[0] is poses
    body.release()
    inspector.sync()
    assert source.frame_poses()[0].shape == (0, 7)
    inspector.set_source(None)
    assert not app.debug_overlay.lines_by_path and app.sync_callback is None
    assert source._frame_metadata is None


def test_joint_frame_rotates_child_local_offset():
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", inertia_diagonal=[1, 1, 1])
    builder.add_link(
        "child",
        parent="root",
        position=[2, 0, 0],
        joint=builder.RevoluteJoint(joint_offset=(0.25, 0, 0)),
        inertia_diagonal=[1, 1, 1],
    )
    world = ke.sim.KangSimWorld(num_envs=1, add_ground=False)
    try:
        robot = world.add_articulation(
            builder.build(), obj_id=0, config=ke.physics.ArticulationConfig.fixed_base()
        ).articulation
        robot.reset_root(pos=[1, 3, 0], quat_xyzw=[0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        source = ke.physics.PhysXInspectionSource(world.physics)
        bodies, joints = source.frame_poses()
        np.testing.assert_allclose(bodies[:, :3], [[1, 3, 0], [1, 5, 0]], atol=1e-5)
        np.testing.assert_allclose(joints[:, :3], [[1, 5.25, 0]], atol=1e-5)
        with patch(
            "kangengine.utils.batched_rotations.quat_xyzw_rotate",
            side_effect=AssertionError("Unused joint frame calculation"),
        ):
            body_only, hidden = source.frame_poses(bodies=True, joints=False)
        np.testing.assert_allclose(body_only, bodies)
        assert hidden.shape == (0, 7)
        hidden, joint_only = source.frame_poses(bodies=False, joints=True)
        assert hidden.shape == (0, 7)
        np.testing.assert_allclose(joint_only, joints)
        restored_bodies, restored_joints = source.frame_poses()
        np.testing.assert_allclose(restored_bodies, bodies)
        np.testing.assert_allclose(restored_joints, joints)
    finally:
        world.release()


def test_application_display_toggles_only_call_on_changes():
    app = RecordingApp()
    inspector = ke.visual.PhysicsInspector(app, source=object())
    values = []
    inspector.add_display_toggle("Scan", values.append, enabled=True)
    inspector.set_display_toggle("Scan", True)
    inspector.sync()
    assert values == [True] and app.sync_callback is None
    inspector.set_display_toggle("Scan", False)
    assert values == [True, False]
    inspector.remove_display_toggle("Scan")
    assert not inspector._display_toggles


def test_frame_colors_are_reused_when_poses_change():
    from types import SimpleNamespace
    from kangengine.visual._frame_display import FrameDisplay

    poses = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    requests = []

    def frame_poses(**options):
        requests.append(options)
        return poses, np.empty((0, 7))

    display = FrameDisplay(RecordingApp())
    source = SimpleNamespace(frame_poses=frame_poses)
    display.sync(source, True, False)
    colors = display._colors
    poses[0, 0] = 1
    display.sync(source, True, False)
    assert display._colors is colors
    assert requests == [{"bodies": True, "joints": False}] * 2
