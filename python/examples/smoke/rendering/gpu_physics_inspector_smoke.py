"""Verify GPU collider poses, shared geometry, and toggle cleanup on either renderer."""

from pathlib import Path

import numpy as np
import torch
import kangengine as ke

from kangengine.utils._example_args import render_example_parser


def run(renderer):
    app = ke.App()
    world = None
    try:
        app.initialize(width=320, height=240, hide_ui=True, renderer=renderer)
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[3, 2, 6], target=[0, 0, 0])
        world = ke.sim.KangSimWorld(num_envs=2, sim_device="cuda:0", add_ground=False)
        data = ke.asset.RigidDesc(collision_geoms=[ke.asset.CollisionGeomDesc(
            type=ke.asset.CollisionGeomDescType.BOX, size=(.3, .2, .4),
            position=(.1, .2, 0), quat_xyzw=(0, 0, 0, 1),
        )])
        for env_id in range(2):
            world.add_rigid(data, env_id=env_id, obj_id=0, pos=[env_id * 2 - 1, 0, 0])
        asset = Path(ke.__file__).parent / "assets/characters/kw/kw5.xml"
        robot_data = world.load_mjcf(str(asset), order="DFS")
        robot = world.add_articulation(robot_data, env_id=0, obj_id=1,
                                       config=ke.physics.ArticulationConfig.fixed_base())
        world.init_gpu_system()
        world.step(substeps=0, refresh=False, apply_commands=False)
        visualizer = ke.visual.sim.SimWorldVisualizer(app, world)
        inspector = app.physics_inspector
        source = inspector.source
        assert source.supports_dynamic and not source.supports_dynamic_picking
        assert source._pose_cache is None
        shapes, _ = source.dynamic_shapes()
        assert shapes and all(shape["actor_pose"] is None for shape in shapes)

        def capture():
            for _ in range(4):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        empty = capture()
        visual = app.scene.add_mesh("/selected", ke.geometry.create_cube_data(.1),
                                    app.create_color_material())
        visual.set_visible(False)
        inspector.bind_visual(visual, world.rigids[(0, 0)].rigid)
        app.select_prim(visual.prim)
        inspector._sync_selection()
        # Render buffer layout stays float32 regardless of application tensor defaults.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            inspector.set_dynamic_visible(True)
        finally:
            torch.set_default_dtype(previous_dtype)
        visible = capture()
        assert np.count_nonzero(np.abs(visible - empty).max(axis=2) > 10) > 30
        assert inspector._gpu_display is not None
        assert any(len(group[2]) == 2 for group in inspector._gpu_display.groups)
        poses, valid = source.body_poses(inspector._dynamic_bodies)
        assert poses.is_cuda and poses.dtype == torch.float32 and valid.all()
        rows = world.gpu_system._collision_body_rows(inspector._dynamic_bodies)
        rigid_ids = [idx for idx, row in enumerate(rows) if row[0] == 0]
        np.testing.assert_allclose(sorted(poses[rigid_ids, 0].cpu().tolist()), [-1, 1], atol=1e-5)
        assert any(row[0] == 1 for row in rows)
        assert inspector._gpu_display.selected in rigid_ids

        # Pausing skips inspection work, and pre-fetched poses are shared with visuals.
        version = inspector._gpu_display.version
        gpu = world.gpu_system
        rigid_version = gpu.rigid_data().version
        link_version = gpu.articulation_link_data().version
        capture()
        assert inspector._gpu_display.version == version
        assert gpu.rigid_data().version == rigid_version
        assert gpu.articulation_link_data().version == link_version
        gpu.fetch_rigid_data()
        gpu.fetch_articulation_link_pose()
        rigid_version = gpu.rigid_data().version
        link_version = gpu.articulation_link_data().version
        inspector.refresh()
        capture()
        assert gpu.rigid_data().version == rigid_version
        assert gpu.articulation_link_data().version == link_version
        assert app.get_selection().prim == visual.prim
        assert inspector._gpu_display.selected in rigid_ids

        # Ordinary visual selection highlights its GPU collider without a pose readback.
        app.clear_selection()
        capture()
        assert inspector._gpu_display.selected is None
        app.select_prim(visual.prim)
        visual.remove()
        assert not app.get_selection().hit

        world.set_root_state(None, 0, [0, 1, 0], [0, 0, 0, 1],
                             linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        world.step(substeps=0, refresh=False, apply_commands=False)
        with torch.cuda.stream(torch.cuda.Stream()):
            inspector.sync()
        moved = capture()
        poses, _ = source.body_poses(inspector._dynamic_bodies)
        np.testing.assert_allclose(poses[rigid_ids, 1].cpu(), 1, atol=1e-5)
        assert np.count_nonzero(np.abs(moved - visible).max(axis=2) > 10) > 30

        version = inspector._gpu_display.version
        pose_cache = source._pose_cache
        render_api = app.get_renderer()
        if renderer == "native":
            render_api.set_profiler_enabled(True)
            capture()
            idle_uploads = render_api.latest_frame_profile().counters.buffer_upload_bytes
        world.step(substeps=1, refresh=False, apply_commands=False)
        app.draw_frame()
        if renderer == "native":
            # CUDA transforms change without re-uploading unchanged CPU colors.
            assert render_api.latest_frame_profile().counters.buffer_upload_bytes == idle_uploads
            render_api.set_profiler_enabled(False)
        capture()
        assert inspector._gpu_display.version == version + 1
        assert source._pose_cache is pose_cache

        # Joint resets change link poses even without advancing simulation time.
        version = inspector._gpu_display.version
        world.set_dof_state([0], 1, torch.full((1, robot.num_dofs), .15, device="cuda:0"),
                            velocities=torch.zeros((1, robot.num_dofs), device="cuda:0"))
        world.step(substeps=0, refresh=False, apply_commands=False)
        capture()
        assert inspector._gpu_display.version == version + 1

        resources = len(inspector._gpu_resources)
        registered = len(app.resources)
        for _ in range(3):
            inspector.set_dynamic_visible(False)
            assert inspector._gpu_display is None and source._pose_cache is None
            # Temporal denoising can retain the last mesh image after removal.
            for _ in range(20):
                hidden = capture()
                if np.abs(hidden - empty).max() <= 2:
                    break
            np.testing.assert_allclose(hidden, empty, atol=2)
            inspector.set_dynamic_visible(True)
            capture()
            assert len(inspector._gpu_resources) == resources
            assert len(app.resources) == registered
        inspector.set_collider_picking_enabled(True)
        assert not inspector._pick_collider([0, 0, 5], [0, 0, -1], 10)
        visualizer.release()
        assert inspector.source is None and inspector._gpu_display is None
        torch.cuda.synchronize()
        print(f"GPU physics inspector passed: {renderer}")
    finally:
        app.close()
        if world is not None:
            world.release()


if __name__ == "__main__":
    parser = render_example_parser(__doc__)
    run(parser.parse_args().renderer)
