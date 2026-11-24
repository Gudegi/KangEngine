"""Check GPU robot visuals, reset, stream handoff, and recreation on both backends."""

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import kangengine as ke
from kangengine.utils.batched_rotations import quat_xyzw_to_matrix


MODEL = '''<mujoco><worldbody><body name="base" pos="0 0 1">
<geom type="box" size=".2 .2 .2"/>
<body name="fixed" pos=".3 0 0"><geom type="sphere" size=".1"/></body>
<body name="arm" pos="0 0 .4"><joint name="hinge" type="hinge" axis="0 1 0"/>
<geom type="capsule" size=".1" fromto="0 0 0 0 0 .8"/>
</body></body></worldbody></mujoco>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = ke.App()
    world = ke.sim.KangSimWorld(num_envs=3, sim_device="cuda:0", add_ground=False)
    visualizer = None
    try:
        app.initialize(width=320, height=240, hide_ui=True, headless=True,
                       up_axis=ke.UpAxis.Z, render_backend=args.backend)
        app.set_camera_view(position=(5, -8, 5), target=(0, 0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.xml"
            path.write_text(MODEL)
            data = world.load_mjcf(str(path))
        ke.sim.GridCloner(world, spacing=2).add_articulation(
            data, obj_id=0, config=ke.physics.ArticulationConfig.fixed_base())
        world.init_gpu_system(cuda_device_id=0)
        world.step(substeps=0, refresh=False, apply_commands=False)
        robots = world.get_articulation_batch(env_ids=[2, 0], obj_id=0)
        creation_stream = torch.cuda.Stream()
        update_stream = torch.cuda.Stream()

        def capture():
            for _ in range(8):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        empty = capture()
        for _ in range(2):
            visualizer = ke.visual.sim.SimWorldVisualizer(app, world)
            with torch.cuda.stream(creation_stream):
                batch = visualizer.add(robots, path="/robots")
            backend = batch._backend
            assert batch.body_handles
            if args.backend == "ovrtx":
                pointer = backend._transforms.data_ptr()

            before = capture()
            assert np.abs(before - empty).mean() > 0.2
            targets = torch.tensor([[0.7], [-0.6]], device="cuda:0")
            robots.set_dof_state(env_ids=None, positions=targets,
                                 velocities=torch.zeros_like(targets))
            world.apply_resets()
            with torch.cuda.stream(update_stream):
                with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU pose copy")), \
                     patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy pose copy")):
                    visualizer.sync()
                    visualizer.sync()
            torch.cuda.synchronize()

            if args.backend == "ovrtx":
                assert pointer == backend._transforms.data_ptr()
                state = world.gpu_system.articulation_link_data().torch()
                rows = [world.articulation_gpu_row(i, 0) for i in batch.env_ids]
                links = backend._link_indices_tensor.long()
                selected = state[rows][:, links].transpose(0, 1)
                matrices = backend._transforms
                torch.testing.assert_close(matrices[:, :, 3, :3], selected[:, :, :3])
                expected_rotation = quat_xyzw_to_matrix(selected[:, :, 3:7]).transpose(-1, -2)
                torch.testing.assert_close(matrices[:, :, :3, :3], expected_rotation)
            assert np.abs(capture() - before).mean() > 0.1

            for handle in batch.body_handles:
                for instance_idx, env_id in enumerate(batch.env_ids):
                    selection = SimpleNamespace(hit=True, handle=handle, instance_index=instance_idx)
                    hit = batch.pick_body(selection)
                    assert hit.env_id == env_id and hit.obj_id == 0
            batch.set_color(((1, 0.1, 0.1, 1), (0.1, 0.2, 1, 1)))
            colored = capture()
            batch.set_visible(False)
            assert np.abs(capture() - empty).mean() < 1
            batch.set_visible(True)
            assert np.abs(capture() - colored).mean() < 1

            for _ in range(4):
                world.step(refresh=False, apply_commands=False)
                visualizer.sync()
                app.draw_frame()
            # Removal must be safe even with a submitted gather that has not been drawn.
            visualizer.sync()
            visualizer.release()
            visualizer.release()
            visualizer = None
            assert app.get_scene_render_system().registration_count == 0
            assert np.abs(capture() - empty).mean() < 1
            robots.set_dof_state(env_ids=None, positions=torch.zeros_like(targets),
                                 velocities=torch.zeros_like(targets))
            world.apply_resets()
        print(f"PASS: {args.backend} GPU articulation poses, streams, reset, visibility, and recreation")
    finally:
        if visualizer is not None:
            visualizer.release()
        app.close()
        world.release()


if __name__ == "__main__":
    main()
