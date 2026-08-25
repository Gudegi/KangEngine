"""Visualize a KW clip with Newton's native viewer.

Run with::

    PYTHONPATH=python python/.venv/bin/python \
        python/examples/adapters/newton/newton_articulation_motion.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.adapters.newton import NewtonMotionAdapter


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

    # Legacy KW motion is y-up/z-forward; the loaded MJCF is z-up/x-forward.
    roots = roots[:, [2, 0, 1]]
    rotations = rotations[..., [0, 3, 1, 2]]
    if rotations.shape[1] != tree.num_joints():
        raise ValueError(
            "legacy KW motion must match the MJCF DFS body count because it "
            "does not store body names"
        )
    return ke.animation.SkeletonMotion.from_arrays(
        skeleton_tree=tree,
        root_translations=roots,
        local_rotations_wxyz=rotations,
        fps=fps,
        motion_name=path.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=_default_mjcf())
    parser.add_argument("--motion", type=Path, default=_default_motion())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--show-joints", action="store_true")
    args = parser.parse_args()

    mjcf_path = args.mjcf.expanduser().resolve()
    motion_path = args.motion.expanduser().resolve()
    if not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)
    if not motion_path.exists():
        raise FileNotFoundError(motion_path)

    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    articulation_data = ke.asset.MJCFLoader.load(str(mjcf_path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=articulation_data,
        free_root=True,
    )
    skeleton_motion = _load_kw_motion(motion_path, articulation_data.skeleton_tree)
    adapter = NewtonMotionAdapter(layout)
    library = ke.animation.MotionLibrary(
        [skeleton_motion],
        adapter=adapter,
        device=args.device,
    )

    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(mjcf_path),
        floating=True,
        ignore_names=["floor", "ground"],
        enable_self_collisions=False,
    )
    model = builder.finalize(device=args.device)
    adapter.validate_model(model)
    state = model.state()

    from newton.viewer import ViewerGL

    viewer = ViewerGL(width=args.width, height=args.height, headless=args.headless)
    viewer.set_model(model)
    viewer.show_joints = bool(args.show_joints)
    viewer.set_camera(wp.vec3(4.5, -6.0, 2.7), pitch=-8.0, yaw=125.0)

    frame = 0
    rendered_frames = 0
    print(
        f"Newton articulation motion: {mjcf_path}\n"
        f"source={motion_path}\n"
        f"frames={library.frame_counts[0].item()} fps={skeleton_motion.fps():g} "
        f"canonical nq/nv={layout.nq}/{layout.nv} device={args.device}\n"
        "viewer=Newton ViewerGL, source=MotionLibrary -> Newton native"
    )
    try:
        while viewer.is_running():
            if viewer.should_step():
                frame = (frame + 1) % skeleton_motion.num_frames()
            sample = library.sample_frames(0, frame)
            if sample.backend_state is None:
                raise RuntimeError("Newton MotionLibrary adapter produced no state")
            model.joint_q.assign(
                wp.from_torch(sample.backend_state.joint_q.contiguous())
            )
            model.joint_qd.assign(
                wp.from_torch(sample.backend_state.joint_qd.contiguous())
            )
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)

            viewer.begin_frame(frame / skeleton_motion.fps())
            viewer.log_state(state)
            viewer.end_frame()

            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
