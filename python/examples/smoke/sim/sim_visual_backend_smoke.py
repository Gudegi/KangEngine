"""Check CPU SceneGraph and ExternalBuffer visuals on both backends."""

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import kangengine as ke


MODEL = '''<mujoco><worldbody><body name="base" pos="0 0 1">
<geom type="box" size=".2 .2 .2"/>
<body name="arm" pos="0 0 .4"><joint name="hinge" type="hinge" axis="0 1 0"/>
<geom type="capsule" size=".1" fromto="0 0 0 0 0 .8"/>
</body></body></worldbody></mujoco>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = ke.App()
    world = ke.sim.KangSimWorld(num_envs=2, add_ground=False)
    viewer = None
    try:
        app.initialize(width=320, height=240, hide_ui=True, headless=True,
                       up_axis=ke.UpAxis.Z, renderer=args.renderer)
        app.set_camera_view(position=(5, -8, 5), target=(0, 0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.xml"
            path.write_text(MODEL)
            data = world.load_mjcf(str(path))
        cloner = ke.sim.GridCloner(world, spacing=3)
        robots = cloner.add_articulation(
            data, obj_id=0, config=ke.physics.ArticulationConfig.fixed_base())
        mesh = ke.geometry.create_cube_data(0.4)
        rigid_data = ke.asset.RigidDesc(
            collision_geoms=[ke.asset.CollisionGeomDesc(
                type=ke.asset.CollisionGeomDescType.BOX, size=(0.2, 0.2, 0.2),
                position=(0, 0, 0), quat_xyzw=(0, 0, 0, 1))],
            visual_geoms=[
                ke.asset.VisualGeomDesc(mesh_data=mesh, rgba=(1, 0, 0, 1)),
                ke.asset.VisualGeomDesc(mesh_data=mesh, position=(0, 0, 0.6), rgba=(0, 1, 0, 1)),
            ])
        rigids = cloner.add_rigid(rigid_data, obj_id=1, kinematic=True,
                                  initial_root_pos=(0, -1.5, 0.5))
        world.step(substeps=0, apply_commands=False)

        def capture():
            for _ in range(4):
                app.draw_frame()
            return app.read_rgb_pixels().copy()

        empty = capture()
        # Recreating visuals must leave physics and shared source meshes usable.
        for _ in range(2):
            viewer = ke.visual.sim.SimWorldVisualizer(app, world)
            scene_robots = [viewer.add_scene_graph(robots, env_id=i, path=f"/scene/robot{i}") for i in range(2)]
            scene_rigids = [viewer.add_scene_graph(rigids, env_id=i, path=f"/scene/rigid{i}") for i in range(2)]
            external_batches = [viewer.add(robots, path="/external/robots"),
                                viewer.add(rigids, path="/external/rigids")]
            assert all(batch.body_handles for batch in external_batches)
            for first, second in zip(scene_robots[0].render_prims, scene_robots[1].render_prims):
                assert first.get_mesh_component().resource_handle == second.get_mesh_component().resource_handle
            for prim in scene_rigids[0].body_prims + scene_rigids[1].body_prims:
                assert prim.get_mesh_component().resource_handle == scene_rigids[0].body_prims[0].get_mesh_component().resource_handle
            assert scene_rigids[0].body_prims[0].get_material() != scene_rigids[0].body_prims[1].get_material()

            for batch in external_batches:
                batch.set_visible(False)
            viewer.sync()
            before = capture()
            assert np.abs(before.astype(float) - empty).mean() > 0.2
            old_pose = np.asarray(scene_robots[0].body_prims[1].compute_world_matrix()).copy()
            robots.set_dof_state(env_ids=None, positions=torch.tensor([[0.8], [-0.6]]),
                                 velocities=torch.zeros((2, 1)), immediate=True)
            viewer.sync()
            assert not np.allclose(old_pose, np.asarray(scene_robots[0].body_prims[1].compute_world_matrix()))
            assert np.abs(capture().astype(float) - before).mean() > 0.1

            # Source-only selections have a prim, while execution batches also have a handle.
            for env_id, record in enumerate(scene_robots):
                if args.renderer == "ovrtx":
                    assert not record.body_handles
                    system = app.get_scene_render_system()
                    component = record.render_prims[0].get_render_component()
                    assert system._handle(component, required=False) is None
                    try:
                        system._handle(component)
                    except RuntimeError as error:
                        assert "Source-only registration" in str(error)
                    else:
                        raise AssertionError("Required batch handle accepted a source-only registration")
                for prim, body_id in zip(record.render_prims, (0, 1)):
                    handle = next((key for key, value in record.handle_body_ids.items() if value == body_id), -1)
                    hit = record.pick_body(SimpleNamespace(hit=True, prim=prim, handle=handle))
                    assert (hit.env_id, hit.obj_id, hit.body_id) == (env_id, 0, body_id)
            for env_id, record in enumerate(scene_rigids):
                handle = record.body_handles[0] if record.body_handles else -1
                hit = record.pick_body(SimpleNamespace(hit=True, prim=record.body_prims[0], handle=handle))
                assert (hit.env_id, hit.obj_id, hit.body_id) == (env_id, 1, 0)
                if args.renderer == "ovrtx":
                    assert record.pick_body(SimpleNamespace(hit=True, prim=scene_robots[0].render_prims[0], handle=-1)) is None

            for record in scene_robots + scene_rigids:
                record.set_visible(False)
            assert np.abs(capture().astype(float) - empty).mean() < 1
            controls = external_batches
            for visual in controls:
                visual.set_visible(True)
            viewer.sync()
            external = capture()
            assert np.abs(external.astype(float) - empty).mean() > 0.2
            for visual in controls:
                visual.set_color(((0.1, 0.3, 1, 1), (1, 0.3, 0.1, 1)))
            assert np.abs(capture().astype(float) - external).mean() > 0.1
            robots.set_dof_state(env_ids=None, positions=torch.zeros((2, 1)),
                                 velocities=torch.zeros((2, 1)), immediate=True)
            viewer.sync()
            capture()
            with tempfile.TemporaryDirectory() as directory:
                video = Path(directory) / "simulation.mp4"
                app.start_video_recording(output_path=video, fps=30)
                capture()
                app.stop_video_recording()
                assert video.stat().st_size > 0
            viewer.release()
            viewer.release()
            assert app.get_scene_render_system().registration_count == 0
            assert np.abs(capture().astype(float) - empty).mean() < 1
            viewer = None
        print(f"PASS: {args.renderer} CPU SceneGraph/ExternalBuffer, picking, reset, video, and recreation")
    finally:
        if viewer is not None:
            viewer.release()
        app.close()
        world.release()


if __name__ == "__main__":
    main()
