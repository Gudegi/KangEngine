"""Check grouped GPU rigid visuals, local parts, resets, and buffer lifetime."""

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import kangengine as ke
from kangengine.utils.batched_rotations import quat_xyzw_to_matrix


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
        mesh = ke.geometry.create_cube_data(0.4)
        collision = ke.asset.CollisionGeomDesc(
            type=ke.asset.CollisionGeomDescType.BOX, size=(0.2, 0.2, 0.2),
            position=(0, 0, 0), quat_xyzw=(0, 0, 0, 1))
        half = np.sqrt(0.5)
        first = ke.asset.RigidDesc(collision_geoms=[collision], visual_geoms=[
            ke.asset.VisualGeomDesc(mesh_data=mesh, rgba=(1, 0.2, 0.2, 1)),
            ke.asset.VisualGeomDesc(mesh_data=mesh, position=(0.4, 0, 0.6),
                                   quat_xyzw=(0, half, 0, half), scale=(0.5, 1, 1.5),
                                   rgba=(0.2, 1, 0.2, 1)),
        ])
        second = ke.asset.RigidDesc(collision_geoms=[collision], visual_geoms=[
            ke.asset.VisualGeomDesc(mesh_data=mesh, position=(0, 0, 0.2),
                                   scale=(1, 2, 0.5), rgba=(0.2, 0.2, 1, 1)),
        ])
        sources = [first, second, first]
        cloner = ke.sim.GridCloner(world, spacing=2)
        cloner.add_rigid(second, obj_id=1, initial_root_pos=(0, 0, -10))
        cloner.add_rigid(sources, obj_id=0, initial_root_pos=(0, 0, 1))
        world.init_gpu_system(cuda_device_id=0)
        world.step(substeps=0, refresh=False, apply_commands=False)
        rigids = world.get_rigid_batch(env_ids=[2, 0, 1], obj_id=0)
        creation_stream = torch.cuda.Stream()
        update_stream = torch.cuda.Stream()

        def capture():
            for _ in range(8):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        empty = capture()
        for iteration in range(2):
            visualizer = ke.visual.sim.SimWorldVisualizer(app, world)
            with torch.cuda.stream(creation_stream):
                batch = visualizer.add(rigids, path="/rigids")
            backend = batch._backend
            assert len(backend._groups) == 2
            before = capture()
            assert np.abs(before - empty).mean() > 0.2
            if args.backend == "ovrtx":
                pointers = [buffer.data_ptr() for buffer in backend._transforms]

            positions = torch.tensor([[1.2, 0, 1.5], [-1, 0, 1], [0, 2, 1]], device="cuda:0")
            positions[:, 2] += iteration * 0.4
            rotations = torch.tensor([[0, 0, half, half]] * 3, dtype=torch.float32, device="cuda:0")
            world.set_root_state([2, 0, 1], 0, positions, rotations,
                                 linear_velocity=torch.zeros_like(positions),
                                 angular_velocity=torch.zeros_like(positions))
            world.apply_resets()
            with torch.cuda.stream(update_stream):
                with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU pose copy")), \
                     patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy pose copy")):
                    visualizer.sync()
                    visualizer.sync()
            torch.cuda.synchronize()

            if args.backend == "ovrtx":
                assert pointers == [buffer.data_ptr() for buffer in backend._transforms]
                for (rows, visual), matrices in zip(backend._groups, backend._transforms):
                    env_ids = [batch.env_ids[row] for row in rows]
                    local_source = sources[env_ids[0]]
                    root = torch.eye(4, device="cuda:0").repeat(len(rows), 1, 1)
                    root[:, :3, :3] = quat_xyzw_to_matrix(rotations[list(rows)])
                    root[:, :3, 3] = positions[list(rows)]
                    for part_idx, part in enumerate(local_source.visual_geoms):
                        local = torch.eye(4, device="cuda:0")
                        rotation = torch.tensor(part.quat_xyzw, device="cuda:0")
                        scale = torch.tensor([part.scale.x, part.scale.y, part.scale.z], device="cuda:0")
                        local[:3, :3] = quat_xyzw_to_matrix(rotation) * scale
                        local[:3, 3] = torch.tensor(
                            [part.pos.x, part.pos.y, part.pos.z], device="cuda:0")
                        torch.testing.assert_close(matrices[part_idx], (root @ local).transpose(-1, -2))
                    for handle in visual.body_handles:
                        for instance_idx, env_id in enumerate(env_ids):
                            selection = SimpleNamespace(hit=True, handle=handle, instance_index=instance_idx)
                            hit = batch.pick_body(selection)
                            assert (hit.env_id, hit.obj_id, hit.body_id) == (env_id, 0, 0)
            assert np.abs(capture() - before).mean() > 0.1
            batch.set_color(((1, 0.5, 0.2, 1), (0.2, 1, 0.5, 1), (0.5, 0.2, 1, 1)))
            colored = capture()
            batch.set_visible(False)
            assert np.abs(capture() - empty).mean() < 1
            batch.set_visible(True)
            assert np.abs(capture() - colored).mean() < 1
            for _ in range(4):
                world.step(refresh=False, apply_commands=False)
                visualizer.sync()
                app.draw_frame()
            visualizer.sync()
            visualizer.release()
            visualizer.release()
            visualizer = None
            assert app.get_scene_render_system().registration_count == 0
            assert np.abs(capture() - empty).mean() < 1
        print(f"PASS: {args.backend} GPU rigid groups, local transforms, streams, reset, and recreation")
    finally:
        if visualizer is not None:
            visualizer.release()
        app.close()
        world.release()


if __name__ == "__main__":
    main()
