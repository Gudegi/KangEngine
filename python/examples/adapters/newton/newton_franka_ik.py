"""Solve Franka IK and render it with KangEngine.

The robot model, forward kinematics, objectives, and IK solver are owned by
Newton. KangEngine renders the solved articulation and owns an editable
SceneGraph Prim for the TCP target. Click the yellow target sphere and use the
standard KangEngine ImGuizmo to translate or rotate it.

The default URDF is downloaded through Newton's asset cache on first use. Its
DAE visual meshes require ``pycollada`` in the Newton environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine.adapters.newton import NewtonViewer


def quaternion_matrix_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a normalized Newton xyzw quaternion to a 3x3 matrix."""

    x, y, z, w = (float(value) for value in rotation)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def create_target_prim(
    viewer: NewtonViewer,
    position: np.ndarray,
    rotation: np.ndarray,
):
    """Create the SceneGraph-owned target manipulated by KangEngine."""

    target = viewer.app.scene.add_mesh(
        "/Newton/IK/Target",
        ke.geometry.create_sphere_data(0.045, 24, 16),
        viewer.newton_material,
        color=ke.Vec4(1.0, 0.75, 0.05, 1.0),
    )
    target.set_world_translation(ke.Vec3(*map(float, position)))
    target.set_world_rotation(ke.Quat.from_xyzw(rotation))
    return target.prim


def target_prim_pose(target_prim) -> tuple[np.ndarray, np.ndarray]:
    """Read the gizmo-authored target pose in Newton's xyzw convention."""

    position = target_prim.get_world_translation()
    rotation = target_prim.get_world_rotation()
    return (
        np.array([position.x, position.y, position.z], dtype=np.float32),
        np.asarray(rotation.to_xyzw(), dtype=np.float32),
    )


def log_target_axes(
    viewer: NewtonViewer,
    position: np.ndarray,
    rotation: np.ndarray,
):
    """Draw local axes for the SceneGraph target Prim."""

    axes = quaternion_matrix_xyzw(rotation)
    starts = np.repeat(position.reshape(1, 3), 3, axis=0)
    ends = starts + 0.12 * axes.T
    viewer.app.debug_overlay.lines(
        "/ik/target_axes",
        starts,
        ends,
        colors=np.array(
            [
                [1.0, 0.15, 0.15, 1.0],
                [0.15, 1.0, 0.15, 1.0],
                [0.15, 0.35, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        width=3.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=Path,
        help="Use a local Franka-compatible URDF instead of Newton's cached asset.",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many rendered frames; zero runs until closed.",
    )
    parser.add_argument("--ik-iterations", type=int, default=24)
    args = parser.parse_args()

    import newton
    import newton.ik as ik
    import newton.utils
    import warp as wp

    urdf_path = (
        args.urdf.expanduser()
        if args.urdf is not None
        else newton.utils.download_asset("franka_emika_panda")
        / "urdf/fr3_franka_hand.urdf"
    )
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Franka URDF not found: {urdf_path}")

    builder = newton.ModelBuilder()
    builder.add_urdf(urdf_path, floating=False)
    builder.add_ground_plane()
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # fr3_hand_tcp in Newton's bundled FR3 asset. A different URDF may require
    # changing this index until link-name lookup is exposed by this example.
    end_effector_index = 10
    body_transform = state.body_q.numpy()[end_effector_index]
    initial_position = np.asarray(body_transform[:3], dtype=np.float32).copy()
    initial_rotation = np.asarray(body_transform[3:7], dtype=np.float32).copy()

    position_objective = ik.IKObjectivePosition(
        link_index=end_effector_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(*initial_position)], dtype=wp.vec3),
    )
    rotation_objective = ik.IKObjectiveRotation(
        link_index=end_effector_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(*initial_rotation)], dtype=wp.vec4),
    )
    joint_limit_objective = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=10.0,
    )
    joint_q = model.joint_q.reshape((1, model.joint_coord_count))
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[
            position_objective,
            rotation_objective,
            joint_limit_objective,
        ],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )

    viewer = NewtonViewer(
        width=args.width,
        height=args.height,
        headless=args.headless,
    )
    viewer.show_ground = False
    viewer.app.scene.add_ground("/Ground", scale=20.0)
    viewer.set_model(model)
    viewer.app.set_interaction_mode(ke.InteractionMode.EDIT)
    viewer.set_camera(wp.vec3(1.8, -2.2, 1.4), pitch=-10.0, yaw=130.0)
    target_prim = create_target_prim(viewer, initial_position, initial_rotation)
    print("IK target: click the yellow sphere, then translate or rotate its gizmo.")

    frame_dt = 1.0 / 60.0
    sim_time = 0.0
    rendered_frames = 0
    target_position = initial_position
    target_rotation = initial_rotation

    try:
        while viewer.is_running():
            target_position, target_rotation = target_prim_pose(target_prim)
            if viewer.should_step():
                position_objective.set_target_position(0, wp.vec3(*target_position))
                rotation_objective.set_target_rotation(0, wp.vec4(*target_rotation))
                solver.step(
                    joint_q,
                    joint_q,
                    iterations=args.ik_iterations,
                )
                sim_time += frame_dt

            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            viewer.begin_frame(sim_time)
            viewer.log_state(state)
            log_target_axes(viewer, target_position, target_rotation)
            viewer.end_frame()

            rendered_frames += 1
            if args.frames > 0 and rendered_frames >= args.frames:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
