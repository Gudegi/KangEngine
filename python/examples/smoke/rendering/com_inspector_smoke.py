"""Check COM/frame overlays, CPU/CUDA poses, occlusion, camera changes, and OFF cleanup."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import kangengine as ke

from kangengine.utils._example_args import render_example_parser


def check_selected_readback(query, options, expected, columns):
    copied = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        copied.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", record_cpu):
        result = query(**options)
    for actual, reference in zip(result, expected):
        np.testing.assert_allclose(actual, reference, atol=1e-5)
    assert copied == [(sum(len(values) for values in expected), columns)], copied


def run(renderer, gpu):
    app = ke.App()
    world = None
    try:
        app.initialize(width=480, height=360, hide_ui=True, renderer=renderer)
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[0, 0, 6], target=[0, 0, 0])
        config = ke.physics.PhysicsConfig.y_up()
        world = ke.sim.KangSimWorld(
            physics_config=config,
            num_envs=1,
            add_ground=False,
            sim_device="cuda:0" if gpu else "cpu",
        )
        data = ke.asset.RigidDesc(
            collision_geoms=[
                ke.asset.CollisionGeomDesc(
                    type=ke.asset.CollisionGeomDescType.SPHERE,
                    size=(0.2, 0, 0),
                    position=(0, 0, 0),
                    quat_xyzw=(0, 0, 0, 1),
                )
            ]
        )
        rigid = world.add_rigid(data, env_id=0, obj_id=0, pos=[0, 0, 0])
        rigid.rigid.set_local_com_pose([0, 0.2, 0, 0, 0, 0, 1])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "robot.xml"
            path.write_text(
                '<mujoco><worldbody><body name="root"><geom type="sphere" size=".1"/>'
                '<body name="child" pos="1 0 0"><joint type="hinge"/>'
                '<geom type="sphere" size=".1"/></body></body></worldbody></mujoco>'
            )
            robot = world.add_articulation(
                world.load_mjcf(str(path)),
                obj_id=1,
                config=ke.physics.ArticulationConfig.fixed_base(),
            ).articulation
        robot.reset_root(pos=[1.5, 0, 0])
        robot.set_link_masses([0, 1], [1, 3])
        if gpu:
            world.init_gpu_system()
        world.set_root_state(
            None,
            0,
            [0, 0, 0],
            [0, 0, 0, 1],
            linear_velocity=[0, 0, 0],
            angular_velocity=[0, 0, 0],
        )
        world.step(substeps=0, refresh=False, apply_commands=False)
        source = ke.physics.PhysXInspectionSource(
            world.physics, gpu_system=world._gpu_system
        )
        inspector = app.physics_inspector
        inspector.set_source(source)
        # The marker lies inside an opaque mesh; overlay visibility must ignore its depth.
        material = app.create_color_material()
        material.unlit = True
        app.scene.add_mesh("/occluder", ke.geometry.create_cube_data(2), material)

        def capture():
            for _ in range(3):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        empty = capture()
        assert source._com_metadata is None
        inspector.set_com_visible(bodies=True, articulations=True)
        visible = capture()
        bodies, robots = source.com_positions()
        np.testing.assert_allclose(
            bodies[np.argsort(bodies[:, 0])],
            [[0, 0.2, 0], [1.5, 0, 0], [2.5, 0, 0]],
            atol=1e-5,
        )
        np.testing.assert_allclose(robots, [[2.25, 0, 0]], atol=1e-5)
        check_selected_readback(
            source.com_positions,
            {"bodies": True, "articulations": False},
            (bodies, np.empty((0, 3))),
            4,
        )
        check_selected_readback(
            source.com_positions,
            {"bodies": False, "articulations": True},
            (np.empty((0, 3)), robots),
            4,
        )
        capture()
        assert np.count_nonzero(np.abs(visible - empty).max(axis=2) > 10) > 40
        center = np.abs(visible - empty)[100:240, 180:300].max(axis=2)
        assert np.count_nonzero(center > 10) > 20
        if gpu:
            cache = source._com_cache
            system = world.gpu_system
            versions = (
                system.rigid_data().version,
                system.articulation_link_data().version,
            )
            capture()
            assert source._com_cache is cache
            assert versions == (
                system.rigid_data().version,
                system.articulation_link_data().version,
            )

        inspector.set_com_visible(bodies=False, articulations=True)
        robot_only = capture()
        assert not np.array_equal(robot_only, visible)
        inspector.set_com_visible(bodies=True, articulations=False)
        body_only = capture()
        assert not np.array_equal(body_only, robot_only)
        app.set_camera_view(position=[0, 0, 12], target=[0, 0, 0])
        signature = inspector._com_display._signature
        capture()
        assert inspector._com_display._signature != signature
        # Both inspection consumers share native fetches without replacing each other's caches.
        inspector.set_dynamic_visible(True)
        capture()
        assert source._com_metadata is not None
        inspector.set_dynamic_visible(False)
        inspector.set_com_visible(bodies=False, articulations=False)
        assert source._com_metadata is None and source._com_cache is None
        app.set_camera_view(position=[0, 0, 6], target=[0, 0, 0])
        restored = capture()
        # OVRTX silhouette antialiasing varies after camera moves; inspect the flat face.
        np.testing.assert_allclose(
            restored[135:200, 210:270], empty[135:200, 210:270], atol=1
        )
        matrix = np.asarray(app.get_projection_matrix()) @ np.asarray(
            app.get_view_matrix()
        )
        for point in np.concatenate((bodies, robots)):
            clip = matrix @ np.append(point, 1)
            ndc = clip[:2] / clip[3]
            x, y = np.rint((ndc * [1, -1] + 1) * [240, 180]).astype(int)
            np.testing.assert_allclose(
                restored[y - 11 : y + 12, x - 11 : x + 12],
                empty[y - 11 : y + 12, x - 11 : x + 12],
                atol=1,
            )
        inspector.set_com_visible(bodies=True, articulations=True)
        world.set_root_state(
            None,
            0,
            [0, 1, 0],
            [0, 0, 0, 1],
            linear_velocity=[0, 0, 0],
            angular_velocity=[0, 0, 0],
        )
        world.step(substeps=0, refresh=False, apply_commands=False)
        if gpu:
            with torch.cuda.stream(torch.cuda.Stream()):
                inspector.sync()
        capture()
        bodies, _ = source.com_positions()
        np.testing.assert_allclose(
            bodies[np.argmin(bodies[:, 0])], [0, 1.2, 0], atol=1e-5
        )
        inspector.refresh()
        capture()
        inspector.set_com_visible(bodies=False, articulations=False)
        without_axes = capture()
        assert source._frame_metadata is None
        inspector.set_frames_visible(bodies=True, joints=True)
        with_axes = capture()
        frames, joints = source.frame_poses()
        np.testing.assert_allclose(
            frames[np.argsort(frames[:, 0]), :3],
            [[0, 1, 0], [1.5, 0, 0], [2.5, 0, 0]],
            atol=1e-5,
        )
        np.testing.assert_allclose(joints[:, :3], [[2.5, 0, 0]], atol=1e-5)
        check_selected_readback(
            source.frame_poses,
            {"bodies": True, "joints": False},
            (frames, np.empty((0, 7))),
            7,
        )
        check_selected_readback(
            source.frame_poses,
            {"bodies": False, "joints": True},
            (np.empty((0, 7)), joints),
            7,
        )
        capture()
        assert np.count_nonzero(np.abs(with_axes - without_axes).max(axis=2) > 10) > 10
        if gpu:
            cache = source._frame_cache
            capture()
            assert source._frame_cache is cache
        inspector.set_frames_visible(bodies=False, joints=True)
        capture()
        inspector.refresh()
        capture()
        inspector.set_source(None)
        assert not inspector.body_com_visible and not inspector.articulation_com_visible
        assert not inspector.body_frames_visible and not inspector.joint_frames_visible
        assert source._frame_metadata is None
        assert source._com_metadata is None
        print(f"com_inspector_smoke: {renderer} {'CUDA' if gpu else 'CPU'} PASS")
    finally:
        app.close()
        if world is not None:
            world.release()


if __name__ == "__main__":
    parser = render_example_parser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    run(args.renderer, args.gpu)
