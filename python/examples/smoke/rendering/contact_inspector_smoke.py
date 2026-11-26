"""Check CPU/CUDA contact glyphs, pause caching, and OFF cleanup on either renderer."""

import numpy as np
import kangengine as ke

from kangengine.utils._example_args import render_example_parser


def run(renderer, gpu):
    app = ke.App()
    world = None
    try:
        app.initialize(width=400, height=300, hide_ui=True, renderer=renderer)
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[2, 1.5, 3], target=[0, 0, 0])
        config = ke.physics.PhysicsConfig.y_up()
        world = ke.sim.KangSimWorld(physics_config=config, num_envs=1,
                                    sim_device="cuda:0" if gpu else "cpu", add_ground=True)
        rigid = ke.asset.RigidDesc(collision_geoms=[ke.asset.CollisionGeomDesc(
            type=ke.asset.CollisionGeomDescType.SPHERE, size=(.5, 0, 0),
            position=(0, 0, 0), quat_xyzw=(0, 0, 0, 1),
        )])
        world.add_rigid(rigid, env_id=0, obj_id=0, pos=[0, .6, 0])
        if gpu:
            world.init_gpu_system()
        source = ke.physics.PhysXInspectionSource(world.physics, gpu_system=world._gpu_system)
        inspector = app.physics_inspector
        inspector.set_source(source)

        def capture():
            for _ in range(4):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        empty = capture()
        assert source._contact_cache is None
        inspector.set_contacts_visible(True)
        display = inspector._contact_display
        for _ in range(35):
            world.step(substeps=1, refresh=False, apply_commands=False)
            inspector.sync()
            if display.count:
                break
        assert display.count > 0
        visible = capture()
        assert np.count_nonzero(np.abs(visible - empty).max(axis=2) > 10) > 5
        data, dt = source.contact_data(velocities=True)
        host = data.cpu().numpy() if gpu else data
        assert np.isfinite(host).all() and dt > 0
        assert np.linalg.norm(host[:, 6:9]) > 0
        if gpu:
            assert data.is_cuda
            assert all(batch["matrices"].is_cuda for batch in display.batches.values())
        version = display.version
        cached = source._contact_cache
        capture()
        assert display.version == version and source._contact_cache is cached
        display.force_scale *= 2
        capture()
        assert display.version == version + 1
        assert source._contact_cache is cached
        if gpu:
            import torch

            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                display.thickness = 1.5
                inspector.sync()
            capture()
            assert display.stream == stream
            assert source._contact_cache is cached

        for name in ("normal", "mode", "force"):
            display.normal = display.mode = display.force = False
            setattr(display, name, True)
            image = capture()
            changed = np.count_nonzero(np.abs(image - empty).max(axis=2) > 10)
            assert changed > 3, (name, changed, display.count)
        inspector.set_contacts_visible(False)
        assert source._contact_cache is None and not display.batches
        np.testing.assert_array_equal(capture(), empty)
        world.step(substeps=1, refresh=False, apply_commands=False)
        capture()
        assert source._contact_cache is None
        inspector.set_contacts_visible(True)
        inspector.refresh()
        capture()
        display.normal = display.mode = display.force = False
        capture()
        version = display.version
        capture()
        assert display.version == version
        np.testing.assert_array_equal(capture(), empty)
        inspector.set_contacts_visible(False)
        print(f"contact_inspector_smoke: {renderer} {'CUDA' if gpu else 'CPU'} PASS")
    finally:
        app.close()
        if world is not None:
            world.release()


if __name__ == "__main__":
    parser = render_example_parser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    run(args.renderer, args.gpu)
