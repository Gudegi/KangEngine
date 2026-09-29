"""External D6 suspension/release on two links per free articulation, CPU or GPU."""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import kangengine as ke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    if args.num_envs < 1 or args.cycles < 1:
        parser.error("num-envs and cycles must be positive")
    world = ke.sim.KangSimWorld(num_envs=args.num_envs,
                               sim_device="cuda:0" if args.gpu else "cpu",
                               sim_dt=1 / 120, add_ground=False)
    batch = None
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suspension.xml"
            path.write_text('''<mujoco><worldbody><body name="root" pos="0 0 3">
                <freejoint/><geom type="sphere" size="0.1" mass="2"/>
                <body name="left" pos="0 0.4 0"><joint type="hinge" axis="1 0 0"/>
                <geom type="sphere" size="0.1" mass="1"/></body>
                <body name="right" pos="0 -0.4 0"><joint type="hinge" axis="1 0 0"/>
                <geom type="sphere" size="0.1" mass="1"/></body>
                </body></worldbody></mujoco>''')
            data = world.load_mjcf(str(path), order="DFS")
        records = []
        for env_id in range(args.num_envs):
            config = ke.physics.ArticulationConfig.free_base()
            record = world.add_articulation(data, env_id=env_id, obj_id=0, config=config)
            records.append(record)
            world.set_root_state([env_id], 0, [env_id * 2., 0., 3.], [0., 0., 0., 1.],
                                 linear_velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.])
        if args.gpu:
            world.init_gpu_system(cuda_device_id=0)
        world.step(substeps=0)

        def positions():
            world.state.refresh()
            return world.state.get_body_pos(0).detach().cpu().numpy().copy()

        def frames():
            p = positions()[:, [1, 2]].reshape(-1, 3)
            return np.column_stack((p, np.tile([0., 0., 0., 1.], (len(p), 1))))

        world_frames = frames()
        n = 2 * args.num_envs
        indices = list(range(n))
        batch = ke.physics.D6Batch.create_world_links(
            world.physics, [r.articulation for r in records for _ in range(2)],
            [1, 2] * args.num_envs, world_frames=world_frames,
            link_frames=np.tile([0., 0., 0., 0., 0., 0., 1.], (n, 1)))
        assert batch.size == n and not any(batch.enabled)
        try:
            batch.set_enabled([n], True)
            raise AssertionError("invalid index accepted")
        except ValueError:
            pass
        assert not any(batch.enabled)
        for bad_indices in ([0, 0], [-1]):
            try:
                batch.set_enabled(bad_indices, True)
                raise AssertionError("invalid indices accepted")
            except ValueError:
                pass
        try:
            batch.set_world_frames([0], [[float("nan"), 0., 0., 0., 0., 0., 1.]])
            raise AssertionError("invalid frame accepted")
        except ValueError:
            pass
        batch.set_link_frames(indices, np.tile([0., 0., 0., 0., 0., 0., 1.], (n, 1)))
        max_error = 0.
        min_drop = float("inf")

        def loaded_step():
            if args.gpu:
                forces = world.get_gpu_articulation_link_forces()
                forces.zero_()
                forces[:, 0, 0] = 12.
                world.apply_gpu_articulation_link_wrenches(forces=True, torques=False)
            else:
                world.set_body_force(None, 0, 0, [12., 0., 0.])
            world.step(refresh=False)

        for cycle in range(args.cycles):
            world_frames = frames()
            batch.set_world_frames(indices, world_frames)
            batch.set_enabled(indices, True)
            # Persistent lateral load in addition to gravity.
            for _ in range(120):
                loaded_step()
            held = positions()[:, [1, 2]].reshape(-1, 3)
            error = np.linalg.norm(held - world_frames[:, :3], axis=1).max()
            assert np.isfinite(held).all() and error < 0.04, (cycle, error)
            max_error = max(max_error, float(error))
            # Only alternate environments release: unchanged rows must keep holding.
            released = np.arange(0, args.num_envs, 2)
            subset = np.column_stack((2 * released, 2 * released + 1)).reshape(-1).tolist()
            batch.set_enabled(subset, False)
            before = positions()
            for _ in range(30):
                loaded_step()
            after = positions()
            drop = (before[released, 1, 2] - after[released, 1, 2]).min()
            assert drop > 0.1, (cycle, drop)
            displacement_x = (after[released, 0, 0] - before[released, 0, 0]).min()
            assert displacement_x > 0.02, ("external force did not move released roots", displacement_x)
            min_drop = min(min_drop, float(drop))
            if args.num_envs > 1:
                np.testing.assert_allclose(after[1::2, 1:3], before[1::2, 1:3], atol=0.04)
            # Capture the new location before reactivation, not the original spawn pose.
            batch.set_enabled(indices, False)
        records[0].articulation.release()
        assert not any(j.valid for j in batch.joints[:2])
        assert all(j.valid for j in batch.joints[2:])
        batch.set_enabled(list(range(2, n)), True)
        world.release()
        assert not any(j.valid for j in batch.joints)
        try:
            batch.set_enabled([0], True)
            raise AssertionError("released batch accepted update")
        except RuntimeError:
            pass
        print(f"D6 PASS gpu={args.gpu} envs={args.num_envs} joints={n} cycles={args.cycles} "
              f"max_anchor_error={max_error:.6f} min_release_drop={min_drop:.6f} "
              f"elapsed={time.perf_counter()-start:.2f}s")
    finally:
        if batch is not None:
            batch.release()
        world.release()


if __name__ == "__main__":
    main()
