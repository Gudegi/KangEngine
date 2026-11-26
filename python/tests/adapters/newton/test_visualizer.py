"""Check embedded and standalone Newton render ownership."""

import os
from unittest.mock import patch

import kangengine as ke
import newton
import numpy as np
import pytest
import torch
import warp as wp

from kangengine.adapters.newton import NewtonVisualizer, NewtonViewer


pytestmark = pytest.mark.skipif(
    os.environ.get("KANGENGINE_TEST_VIEWER") != "1", reason="Requires a display"
)


def test_embedded_release_keeps_app_and_other_geometry_alive():
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    model = builder.finalize(device="cpu")
    app = ke.App()
    app.initialize(
        width=128,
        height=128,
        headless=True,
        renderer=os.environ.get("KANGENGINE_TEST_RENDERER", "native"),
    )
    try:
        other = app.scene.add_mesh(
            path="/other",
            mesh_data=ke.geometry.create_cube_data(0.2),
            material=app.create_standard_materials().pbr,
        )
        visualizer = NewtonVisualizer(app)
        visualizer.set_model(model)
        state = model.state()
        with patch.object(app, "draw_frame") as draw:
            visualizer.log_state(state)
            visualizer.end_frame()
            draw.assert_not_called()
        assert visualizer._instances
        app.draw_frame()
        visualizer.release()
        visualizer.release()
        assert visualizer.model is None
        assert not visualizer._instances
        assert not app.should_close()
        assert other.prim.is_visible()
        app.draw_frame()
    finally:
        app.close()


def test_standalone_viewer_owns_and_closes_app():
    viewer = NewtonViewer(
        width=128,
        height=128,
        headless=True,
        renderer=os.environ.get("KANGENGINE_TEST_RENDERER", "native"),
    )
    try:
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.3)
        model = builder.finalize(device="cpu")
        viewer.set_model(model)
        assert viewer._picking is not None
        viewer.begin_frame(0.0)
        viewer.log_state(model.state())
        viewer.end_frame()
        assert viewer.is_running()
        viewer.show_joints = True
        viewer.log_state(model.state())
        assert viewer.app.physics_inspector.joint_frames_visible
        viewer.show_joints = False
        viewer.log_state(model.state())
        assert not viewer.app.physics_inspector.joint_frames_visible
    finally:
        viewer.close()
    viewer.close()
    assert viewer.app is None


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_inspector_colliders_com_and_state_swap(device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.begin_world()
    inertia = wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
    root = builder.add_link(
        mass=1, com=wp.vec3(0.5, 0, 0), inertia=inertia, lock_inertia=True
    )
    child = builder.add_link(mass=3, inertia=inertia, lock_inertia=True)
    builder.add_shape_box(root, hx=0.2, hy=0.3, hz=0.4)
    builder.add_shape_capsule(child, radius=0.1, half_height=0.5)
    builder.add_articulation(
        [
            builder.add_joint_free(root),
            builder.add_joint_fixed(
                root,
                child,
                child_xform=wp.transform(wp.vec3(0, 0.25, 0), wp.quat_identity()),
            ),
        ]
    )
    builder.end_world()
    model = builder.finalize(device=device)
    app = ke.App()
    app.initialize(
        width=128,
        height=128,
        headless=True,
        up_axis=ke.UpAxis.Z,
        renderer=os.environ.get("KANGENGINE_TEST_RENDERER", "native"),
    )
    visualizer = NewtonVisualizer(app)
    try:
        visualizer.set_model(model)
        visualizer.set_world_offsets((4, 0, 0))
        inspector = app.physics_inspector
        source = inspector.source
        state = model.state()
        state.body_q.assign(
            np.array(
                [
                    [1, 0, 1, 0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                    [3, 0, 1, 0, 0, 0, 1],
                ],
                np.float32,
            )
        )

        # OFF does not query inspection metadata, poses, or COM positions.
        with (
            patch.object(
                source, "_snapshot_shapes", side_effect=AssertionError("OFF metadata")
            ),
            patch.object(source, "_poses", side_effect=AssertionError("OFF poses")),
        ):
            visualizer.log_state(state)
            app.draw_frame()
        assert source._com is None
        assert source._pose_buffers == {}

        inspector.set_static_visible(True)
        inspector.set_dynamic_visible(True)
        assert len(inspector._dynamic_bodies) == 2
        poses, valid = source.body_poses(inspector._dynamic_bodies)
        assert valid.all()
        assert bool(getattr(poses, "is_cuda", False)) == (device == "cuda:0")
        actual = poses.cpu().numpy() if isinstance(poses, torch.Tensor) else poses
        np.testing.assert_allclose(actual[:, :3], [[1, 0, 1], [3, 0, 1]])
        if model.device.is_cuda:
            with patch.object(
                wp.array, "numpy", side_effect=AssertionError("Collider readback")
            ):
                source.body_poses(inspector._dynamic_bodies)
        capsule = next(s for s in source._shapes.values() if s["type"] == "capsule")
        axis = wp.quat_rotate(wp.quat(*capsule["local_pose"][3:]), wp.vec3(1, 0, 0))
        np.testing.assert_allclose(axis, [0, 0, 1], atol=1e-6)

        inspector.set_com_visible(bodies=True, articulations=True)
        centers, robots = source.com_positions()
        np.testing.assert_allclose(centers, [[1, 0.5, 1], [3, 0, 1]], atol=1e-6)
        np.testing.assert_allclose(robots, [[2.5, 0.125, 1]], atol=1e-6)
        assert source.com_positions()[0] is centers
        inspector.set_frames_visible(bodies=True, joints=True)
        body_frames, joint_frames = source.frame_poses()
        np.testing.assert_allclose(
            body_frames[:, :3], [[1, 0, 1], [3, 0, 1]], atol=1e-6
        )
        np.testing.assert_allclose(
            joint_frames[:, :3], [[1, 0, 1], [3, 0.25, 1]], atol=1e-6
        )
        assert source.frame_poses()[0] is body_frames
        assert source.frame_poses(bodies=False, joints=True)[0].shape == (0, 7)
        app.draw_frame()

        body, shape_idx = source.pick([1, 0, 4], [0, 0, -1], 10, False, True)
        assert body.body_idx == root
        assert shape_idx in source._shapes
        assert source.pick([1, 0, 4], [0, 0, -1], 1, False, True) is None
        static, _ = source.pick([7, 0, 4], [0, 0, -1], 10, True, False)
        assert static.body_idx == -1

        # Resolve a rendered ExternalBuffer instance without a second scene raycast.
        from types import SimpleNamespace

        batch = next(b for b in visualizer._shape_instances.values() if not b.static)
        view = visualizer._instances[batch.name].view
        selection = SimpleNamespace(
            hit=True, prim=view.prim, instance_index=0, handle=0
        )
        selected, _ = source.resolve_selection(selection)
        assert selected.body_idx == root

        # Ping-pong state buffers must not leave the inspector on the old pose.
        replacement = model.state()
        replacement.body_q.assign(
            np.array([[2, 0, 1, 0, 0, 0, 1], [4, 0, 1, 0, 0, 0, 1]], np.float32)
        )
        stream = wp.Stream(model.device) if model.device.is_cuda else None
        with wp.ScopedStream(stream):
            visualizer.log_state(replacement)
        app.draw_frame()
        np.testing.assert_allclose(
            source.com_positions()[1], [[3.625, 0, 1]], atol=1e-6
        )
        inspector.set_com_visible(bodies=False, articulations=False)
        inspector.set_frames_visible(bodies=False, joints=False)
        inspector.set_dynamic_visible(False)
        inspector.set_static_visible(False)
        with patch.object(source, "_poses", side_effect=AssertionError("OFF poses")):
            visualizer.log_state(state)
            app.draw_frame()
        visualizer.set_model(model)
        assert not body.valid
        assert inspector.source is not source
        assert not inspector.dynamic_visible
        visualizer.release()
        assert inspector.source is None
    finally:
        visualizer.release()
        app.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_inspector_world_offsets_and_visibility(device):
    world = newton.ModelBuilder()
    body = world.add_body(xform=wp.transform(wp.vec3(1, 2, 3), wp.quat_identity()))
    world.add_shape_sphere(body, radius=0.2)
    world.add_shape_box(-1, hx=0.2, hy=0.3, hz=0.4)
    builder = newton.ModelBuilder()
    builder.replicate(world, world_count=3)
    model = builder.finalize(device=device)
    app = ke.App()
    app.initialize(
        width=128,
        height=128,
        headless=True,
        renderer=os.environ.get("KANGENGINE_TEST_RENDERER", "native"),
    )
    visualizer = NewtonVisualizer(app)
    try:
        visualizer.set_model(model)
        visualizer.set_world_offsets((4, 0, 0))
        visualizer.log_state(model.state())
        inspector = app.physics_inspector
        source = inspector.source
        inspector.set_static_visible(True)
        inspector.set_dynamic_visible(True)
        inspector.set_com_visible(bodies=True, articulations=True)
        offsets = visualizer.world_offsets.numpy()
        expected = model.body_q.numpy()[:, :3] + offsets
        centers, _ = source.com_positions()
        np.testing.assert_allclose(centers, expected)
        statics = source.static_shapes()
        np.testing.assert_allclose([s["actor_pose"][:3] for s in statics], offsets)
        app.draw_frame()

        visualizer.set_visible_worlds([1])
        visualizer.log_state(model.state())
        assert len(inspector._dynamic_bodies) == 1
        assert inspector._dynamic_bodies[0].body_idx == 1
        assert len(source.static_shapes()) == 1
        centers, _ = source.com_positions()
        np.testing.assert_allclose(centers, model.body_q.numpy()[1:2, :3])
        app.draw_frame()
    finally:
        visualizer.release()
        app.close()
