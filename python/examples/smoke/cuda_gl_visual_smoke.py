"""Verify CUDA/OpenGL transform mapping and simulation visual synchronization."""

from pathlib import Path


def main():
    import kangengine as ke
    import torch
    from kangengine.visual.sim import SimWorldVisualizer

    app = ke.App()
    app.initialize(width=160, height=120, hide_ui=True, headless=True)
    world = ke.sim.KangSimWorld(num_envs=4, sim_device="cuda:0", add_ground=False)
    visual = None
    try:
        asset = Path(ke.__file__).resolve().parent / "assets/objects/ball.xml"
        data = world.load_mjcf(str(asset))
        balls = ke.sim.GridCloner(world, spacing=(1.0, 1.0)).add_rigid(
            data, obj_id=0, name="ball", density=60, initial_root_pos=(0, 0, 2),
        )
        world.init_gpu_system(cuda_device_id=0)
        world.step(substeps=0, refresh=False, apply_commands=False)
        visual = SimWorldVisualizer(app, world)
        instance = visual.add(balls)

        # Require real CUDA/GL mapping; an empty or CPU fallback path must not
        # turn this initialization regression check into a successful no-op.
        renderer_type = type(app.get_renderer())
        original = renderer_type.map_renderable_cuda_transform_buffers
        mapped_count = 0

        def checked_map(renderer, *args, **kwargs):
            nonlocal mapped_count
            views = original(renderer, *args, **kwargs)
            assert views and all(
                view.ptr != 0 and view.device_id == 0
                and view.memory_type == ke.physics.SimMemoryType.OPENGL_BUFFER
                for view in views
            )
            mapped_count += len(views)
            return views

        renderer_type.map_renderable_cuda_transform_buffers = checked_map
        try:
            for _ in range(3):
                world.step(refresh=False, apply_commands=False)
                visual.sync()
                torch.cuda.synchronize()
                app.draw_frame()
            assert mapped_count > 0
            assert instance.body_handles
        finally:
            renderer_type.map_renderable_cuda_transform_buffers = original
    finally:
        if visual is not None:
            visual.release()
        app.close()
        world.release()
    print("PASS: CUDA/OpenGL visual synchronization")


if __name__ == "__main__":
    main()
