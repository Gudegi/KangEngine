"""Visualize a KW clip with MuJoCo's native viewer.

Run with::

    PYTHONPATH=python python/.venv/bin/python \
        python/examples/adapters/mujoco/mujoco_articulation_motion.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

import kangengine as ke
from kangengine.adapters.mujoco import MuJoCoMotionAdapter


def _default_mjcf() -> Path:
    return Path(ke.__file__).resolve().parent / "assets/characters/kw/kw.xml"


def _default_motion() -> Path:
    return (
        Path(ke.__file__).resolve().parent / "assets/characters/kw/motions/Walking.npy"
    )


def _load_kw_motion(path: Path, tree) -> ke.animation.SkeletonMotion:
    saved = np.load(path, allow_pickle=True).item()
    rotations = np.asarray(saved["localRotations"], dtype=np.float32)
    roots = np.asarray(saved["rootTranslations"], dtype=np.float32)
    fps = float(np.asarray(saved["frameRate"]).item())
    # Legacy KW motion is y-up/z-forward; the MJCF is z-up/x-forward.
    roots = roots[:, [2, 0, 1]]
    rotations = rotations[..., [0, 3, 1, 2]]
    if rotations.shape[1] != tree.num_joints():
        raise ValueError("legacy KW motion does not match the MJCF DFS body order")
    return ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=tree,
        root_translations=roots,
        local_rotations_wxyz=rotations,
        fps=fps,
        motion_name=path.name,
    )


def _load_model_with_example_floor(mjcf_path: Path, mujoco):
    """Compile the MJCF without its floor and add a Z=0 example plane."""

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    for geom in tuple(spec.worldbody.geoms):
        if geom.name in ("floor", "ground"):
            spec.delete(geom)
    spec.worldbody.add_geom(
        name="example_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 0.125),
        friction=(1.0, 0.1, 0.1),
        rgba=(0.8, 0.9, 0.8, 1.0),
    )
    return spec.compile()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=_default_mjcf())
    parser.add_argument("--motion", type=Path, default=_default_motion())
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()

    mjcf_path = args.mjcf.expanduser().resolve()
    motion_path = args.motion.expanduser().resolve()
    if not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)
    if not motion_path.exists():
        raise FileNotFoundError(motion_path)

    import mujoco
    import mujoco.viewer

    articulation_data = ke.asset.MJCFLoader.load(str(mjcf_path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=articulation_data,
        free_root=True,
    )
    motion = _load_kw_motion(motion_path, articulation_data.skeleton_tree)
    adapter = MuJoCoMotionAdapter(layout)
    library = ke.animation.MotionLibrary([motion], adapter=adapter, device="cpu")

    model = _load_model_with_example_floor(mjcf_path, mujoco)
    data = mujoco.MjData(model)
    adapter.validate_model(model)
    frame = 0
    rendered_frames = 0
    frame_dt = 1.0 / motion.fps()
    print(
        f"MuJoCo articulation motion: {mjcf_path}\n"
        f"source={motion_path}\n"
        f"frames={motion.num_frames()} fps={motion.fps():g} "
        f"nq/nv={model.nq}/{model.nv}\n"
        "viewer=MuJoCo native, source=MotionLibrary -> MuJoCo qpos/qvel"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            sample = library.sample_frames(0, frame)
            if sample.backend_state is None:
                raise RuntimeError("MuJoCo MotionLibrary adapter produced no state")
            data.qpos[:] = sample.backend_state.qpos.numpy()
            data.qvel[:] = sample.backend_state.qvel.numpy()
            mujoco.mj_forward(model, data)
            viewer.sync()

            frame = (frame + 1) % motion.num_frames()
            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
            remaining = frame_dt - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
