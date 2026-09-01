"""PyRoki inverse-kinematics entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from ...animation.articulation_motion import (
    ArticulationCoordinateLayout,
    ArticulationCoordinateType,
    ArticulationMotion,
)
from ...animation.retarget.ik import (
    IKTargetMotion,
    build_ik_target_motion,
    detect_foot_contacts,
)
from ...animation.retarget.ik_profile import IKRetargetConfig
from ...utils.math import (
    quat_wxyz_multiply_numpy,
    quat_wxyz_slerp_numpy,
    quat_wxyz_to_rotation_vector_numpy,
)
from ._dependency import load_pyroki
from .model import PyrokiRobotModel


@dataclass(frozen=True)
class PyrokiPoseIKConfig:
    """Weights and stopping policy for the one-frame reference solver."""

    position_weight: float = 5.0
    orientation_weight: float = 0.0
    joint_rest_weight: float = 0.05
    root_rest_weight: float = 0.01
    joint_limit_weight: float = 1.0
    max_iterations: int = 100


@dataclass(frozen=True)
class PyrokiPoseIKResult:
    """One solved pose in profile-canonical ordering."""

    root_position: npt.NDArray[np.float32]
    root_rotation_wxyz: npt.NDArray[np.float32]
    joint_configuration: npt.NDArray[np.float32]
    initial_position_rmse: float
    position_rmse: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class PyrokiTrajectoryIKConfig:
    """Initial whole-trajectory objective weights."""

    position_weight: float = 4.0
    orientation_weight: float = 0.0
    local_position_weight: float = 1.0
    local_angle_weight: float = 1.0
    joint_rest_weight: float = 0.02
    joint_smoothness_weight: float = 4.0
    root_smoothness_weight: float = 1.0
    joint_limit_weight: float = 100.0
    joint_velocity_limit_weight: float = 50.0
    foot_skating_weight: float = 30.0
    foot_tilt_weight: float = 1.0
    max_iterations: int = 800
    whole_trajectory_max_frames: int = 1000
    window_size: int = 512
    window_overlap: int = 32
    window_continuity_weight: float = 10.0
    trajectory_joint_reference_weight: float = 0.0
    trajectory_root_reference_weight: float = 0.0


@dataclass(frozen=True)
class PyrokiTrajectoryState:
    """Canonical joint trajectory with optional free-root poses."""

    joint_configurations: npt.ArrayLike
    root_positions: npt.ArrayLike | None = None
    root_rotations_wxyz: npt.ArrayLike | None = None

    def slice(self, start: int, end: int) -> PyrokiTrajectoryState:
        joints = np.asarray(self.joint_configurations)
        return PyrokiTrajectoryState(
            joint_configurations=joints if joints.ndim == 1 else joints[start:end],
            root_positions=(
                None
                if self.root_positions is None
                else np.asarray(self.root_positions)[start:end]
            ),
            root_rotations_wxyz=(
                None
                if self.root_rotations_wxyz is None
                else np.asarray(self.root_rotations_wxyz)[start:end]
            ),
        )


@dataclass(frozen=True)
class PyrokiTrajectoryIKResult:
    """Whole-trajectory solve output in profile-canonical ordering."""

    root_positions: npt.NDArray[np.float32]
    root_rotations_wxyz: npt.NDArray[np.float32]
    joint_configurations: npt.NDArray[np.float32]
    initial_position_rmse: float
    position_rmse: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class PyrokiTrajectoryIKProgress:
    """Progress reported after one trajectory window has been solved."""

    window_index: int
    window_count: int
    frames_completed: int
    frame_count: int
    frame_start: int
    frame_end: int
    position_rmse: float
    iterations: int
    converged: bool


PyrokiProgressCallback = Callable[[PyrokiTrajectoryIKProgress], None]


@dataclass(frozen=True)
class PyrokiOfflineRetargetResult:
    """Canonical motion plus PyRoki trajectory diagnostics."""

    motion: ArticulationMotion
    solve: PyrokiTrajectoryIKResult


def _prepare_trajectory_state(
    state: PyrokiTrajectoryState | None,
    *,
    frame_count: int,
    joint_count: int,
    name: str,
    allow_single_configuration: bool = False,
    allow_prefix: bool = False,
) -> PyrokiTrajectoryState | None:
    if state is None:
        return None
    joints = np.asarray(state.joint_configurations, dtype=np.float32)
    if allow_single_configuration and joints.shape == (joint_count,):
        state_frames = frame_count
    elif joints.ndim == 2 and joints.shape[1] == joint_count:
        state_frames = joints.shape[0]
        if state_frames != frame_count and not (
            allow_prefix and 0 < state_frames <= frame_count
        ):
            raise ValueError(f"{name}.joint_configurations has an invalid shape")
    else:
        raise ValueError(f"{name}.joint_configurations has an invalid shape")

    roots = (state.root_positions, state.root_rotations_wxyz)
    if any(value is not None for value in roots) and not all(
        value is not None for value in roots
    ):
        raise ValueError(
            f"{name} root_positions and root_rotations_wxyz must be provided together"
        )
    root_positions = None
    root_rotations = None
    if state.root_positions is not None:
        root_positions = np.asarray(state.root_positions, dtype=np.float32)
        root_rotations = np.asarray(state.root_rotations_wxyz, dtype=np.float32)
        expected_frames = state_frames if joints.ndim == 2 else frame_count
        if root_positions.shape != (expected_frames, 3):
            raise ValueError(f"{name}.root_positions has an invalid shape")
        if root_rotations.shape != (expected_frames, 4):
            raise ValueError(f"{name}.root_rotations_wxyz has an invalid shape")
    return PyrokiTrajectoryState(joints, root_positions, root_rotations)


def _backend_from_canonical(
    model: PyrokiRobotModel, profile: IKRetargetConfig
) -> npt.NDArray[np.int32]:
    canonical_index = {name: index for index, name in enumerate(profile.joint_order)}
    return np.asarray(
        [canonical_index[name] for name in model.actuated_joint_names],
        dtype=np.int32,
    )


def solve_pose_ik(
    model: PyrokiRobotModel,
    profile: IKRetargetConfig,
    targets: IKTargetMotion,
    *,
    frame: int = 0,
    initial_joint_configuration: npt.ArrayLike | None = None,
    config: PyrokiPoseIKConfig = PyrokiPoseIKConfig(),
) -> PyrokiPoseIKResult:
    """Solve one target frame with a free root and URDF joint limits."""

    if not 0 <= frame < targets.num_frames:
        raise IndexError(f"frame {frame} is outside [0, {targets.num_frames})")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    pyroki, _ = load_pyroki()
    import jax.numpy as jnp
    import jaxlie
    import jaxls

    robot = model.robot
    backend_order = model.actuated_joint_names
    backend_from_canonical = _backend_from_canonical(model, profile)

    canonical_rest = (
        np.asarray(profile.rest_configuration, dtype=np.float32)
        if profile.rest_configuration is not None
        else np.zeros(len(profile.joint_order), dtype=np.float32)
    )
    if initial_joint_configuration is None:
        canonical_initial = canonical_rest
    else:
        canonical_initial = np.asarray(initial_joint_configuration, dtype=np.float32)
    expected_shape = (len(profile.joint_order),)
    if canonical_initial.shape != expected_shape:
        raise ValueError(
            f"initial_joint_configuration shape is {canonical_initial.shape}, "
            f"expected {expected_shape}"
        )
    backend_initial = canonical_initial[backend_from_canonical]
    backend_rest = canonical_rest[backend_from_canonical]

    try:
        target_link_indices = np.asarray(
            [model.link_names.index(name) for name in targets.target_link_names],
            dtype=np.int32,
        )
        root_target_index = targets.target_link_names.index(targets.target_root_link)
    except ValueError as error:
        raise ValueError(
            "IK targets contain a link absent from the loaded model"
        ) from error

    target_positions = np.asarray(targets.positions[frame].cpu(), dtype=np.float32)
    target_rotations = np.asarray(targets.rotations_wxyz[frame].cpu(), dtype=np.float32)
    target_poses = jaxlie.SE3(
        jnp.asarray(np.concatenate((target_rotations, target_positions), axis=1))
    )

    initial_root = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.asarray(target_positions[root_target_index])
    )

    class RootVar(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: initial_root,
        retract_fn=jaxls.SE3Var.retract_fn,
        tangent_dim=jaxlie.SE3.tangent_dim,
    ):
        pass

    joint_var = robot.joint_var_cls(0)
    root_var = RootVar(0)

    position_weights = (
        jnp.asarray(np.asarray(targets.position_weights.cpu(), dtype=np.float32))
        * config.position_weight
    )[:, None]
    orientation_weights = (
        jnp.asarray(np.asarray(targets.rotation_weights.cpu(), dtype=np.float32))
        * config.orientation_weight
    )[:, None]
    rest_weights = jnp.concatenate(
        (
            jnp.full((len(backend_order),), config.joint_rest_weight),
            jnp.full((6,), config.root_rest_weight),
        )
    )
    costs = [
        pyroki.costs.pose_cost_with_base(
            robot,
            joint_var,
            root_var,
            target_poses,
            jnp.asarray(target_link_indices),
            pos_weight=position_weights,
            ori_weight=orientation_weights,
        ),
        pyroki.costs.rest_with_base_cost(
            joint_var,
            root_var,
            jnp.asarray(backend_rest),
            rest_weights,
        ),
        pyroki.costs.limit_constraint(
            robot,
            joint_var,
            weight=jnp.asarray(config.joint_limit_weight),
        ),
    ]
    initial_values = jaxls.VarValues.make(
        [
            joint_var.with_value(jnp.asarray(backend_initial)),
            root_var.with_value(initial_root),
        ]
    )
    solution, summary = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var, root_var])
        .analyze()
        .solve(
            initial_vals=initial_values,
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=config.max_iterations),
            verbose=False,
            return_summary=True,
        )
    )

    root_pose = solution[root_var]
    backend_solution = np.asarray(solution[joint_var], dtype=np.float32)
    canonical_solution = np.empty_like(backend_solution)
    canonical_solution[backend_from_canonical] = backend_solution

    initial_positions = _world_link_positions(
        model, backend_initial, np.asarray(initial_root.wxyz_xyz, dtype=np.float32)
    )[target_link_indices]
    solved_root = np.asarray(root_pose.wxyz_xyz, dtype=np.float32)
    solved_positions = _world_link_positions(model, backend_solution, solved_root)[
        target_link_indices
    ]
    initial_rmse = float(np.sqrt(np.mean((initial_positions - target_positions) ** 2)))
    solved_rmse = float(np.sqrt(np.mean((solved_positions - target_positions) ** 2)))

    return PyrokiPoseIKResult(
        root_position=solved_root[4:].copy(),
        root_rotation_wxyz=solved_root[:4].copy(),
        joint_configuration=canonical_solution,
        initial_position_rmse=initial_rmse,
        position_rmse=solved_rmse,
        iterations=int(np.asarray(summary.iterations)),
        converged=bool(np.any(np.asarray(summary.termination_criteria))),
    )


def solve_trajectory_ik(
    model: PyrokiRobotModel,
    profile: IKRetargetConfig,
    targets: IKTargetMotion,
    *,
    initial_state: PyrokiTrajectoryState | None = None,
    continuity_reference: PyrokiTrajectoryState | None = None,
    trajectory_reference: PyrokiTrajectoryState | None = None,
    config: PyrokiTrajectoryIKConfig = PyrokiTrajectoryIKConfig(),
) -> PyrokiTrajectoryIKResult:
    """Jointly solve every frame with temporal joint/root coupling."""

    if targets.num_frames < 1:
        raise ValueError("trajectory IK requires at least one frame")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.trajectory_joint_reference_weight < 0.0:
        raise ValueError("trajectory_joint_reference_weight must be non-negative")
    if config.trajectory_root_reference_weight < 0.0:
        raise ValueError("trajectory_root_reference_weight must be non-negative")

    pyroki, _ = load_pyroki()
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls

    frames = targets.num_frames
    robot = model.robot
    backend_order = model.actuated_joint_names
    backend_from_canonical = _backend_from_canonical(model, profile)

    canonical_rest = (
        np.asarray(profile.rest_configuration, dtype=np.float32)
        if profile.rest_configuration is not None
        else np.zeros(len(profile.joint_order), dtype=np.float32)
    )
    initial_state = _prepare_trajectory_state(
        initial_state,
        frame_count=frames,
        joint_count=len(profile.joint_order),
        name="initial_state",
        allow_single_configuration=True,
    )
    if initial_state is not None:
        supplied_initial = np.asarray(
            initial_state.joint_configurations, dtype=np.float32
        )
        if supplied_initial.ndim == 1:
            canonical_initial = np.broadcast_to(
                supplied_initial, (frames, supplied_initial.shape[0])
            ).copy()
        else:
            canonical_initial = supplied_initial.copy()
    else:
        canonical_initial = np.broadcast_to(
            canonical_rest, (frames, canonical_rest.shape[0])
        ).copy()
    backend_initial = canonical_initial[:, backend_from_canonical]
    backend_rest = canonical_rest[backend_from_canonical]
    backend_rest_weights = np.asarray(
        [
            profile.joint_rest_weights.get(name, config.joint_rest_weight)
            for name in backend_order
        ],
        dtype=np.float32,
    )

    try:
        target_link_indices = np.asarray(
            [model.link_names.index(name) for name in targets.target_link_names],
            dtype=np.int32,
        )
        root_target_index = targets.target_link_names.index(targets.target_root_link)
    except ValueError as error:
        raise ValueError(
            "IK targets contain a link absent from the loaded model"
        ) from error

    target_positions = np.asarray(targets.positions.cpu(), dtype=np.float32)
    target_rotations = np.asarray(targets.rotations_wxyz.cpu(), dtype=np.float32)
    foot_shape_link_pairs = tuple(
        (ankle_name, foot_name)
        for ankle_name, foot_name in profile.foot_shape_pairs
        if ankle_name in targets.target_link_names
        and foot_name in targets.target_link_names
    )
    primary_count = targets.primary_effector_count
    primary_link_names = targets.target_link_names[:primary_count]
    if profile.local_alignment_pairs:
        local_mask = np.zeros((primary_count, primary_count), dtype=np.float32)
        primary_slots = {name: index for index, name in enumerate(primary_link_names)}
        for first, second in profile.local_alignment_pairs:
            first_slot = primary_slots[first]
            second_slot = primary_slots[second]
            local_mask[first_slot, second_slot] = 1.0
            local_mask[second_slot, first_slot] = 1.0
        connection_mask = jnp.asarray(local_mask)
    else:
        connection_mask = jnp.asarray(model.connection_mask(primary_link_names))
    target_offsets = jnp.asarray(
        np.asarray(targets.target_offsets.cpu(), dtype=np.float32)
    )
    initial_root_values = np.zeros((frames, 7), dtype=np.float32)
    if targets.initial_root_rotations_wxyz is None:
        initial_root_values[:, 0] = 1.0
    else:
        initial_root_values[:, :4] = np.asarray(
            targets.initial_root_rotations_wxyz.cpu(), dtype=np.float32
        )
    initial_root_values[:, 4:] = target_positions[:, root_target_index]
    if initial_state is not None and initial_state.root_positions is not None:
        initial_root_values[:, :4] = initial_state.root_rotations_wxyz
        initial_root_values[:, 4:] = initial_state.root_positions
    initial_roots = jaxlie.SE3(jnp.asarray(initial_root_values))

    continuity_reference = _prepare_trajectory_state(
        continuity_reference,
        frame_count=frames,
        joint_count=len(profile.joint_order),
        name="continuity_reference",
        allow_prefix=True,
    )
    reference_frames = 0
    backend_reference_joints = None
    reference_roots = None
    if continuity_reference is not None:
        if continuity_reference.root_positions is None:
            raise ValueError("continuity_reference requires free-root poses")
        canonical_reference_joints = np.asarray(
            continuity_reference.joint_configurations, dtype=np.float32
        )
        root_reference_positions = continuity_reference.root_positions
        root_reference_rotations = continuity_reference.root_rotations_wxyz
        reference_frames = canonical_reference_joints.shape[0]
        backend_reference_joints = canonical_reference_joints[:, backend_from_canonical]
        reference_roots = np.concatenate(
            (root_reference_rotations, root_reference_positions), axis=1
        )

    trajectory_reference = _prepare_trajectory_state(
        trajectory_reference,
        frame_count=frames,
        joint_count=len(profile.joint_order),
        name="trajectory_reference",
    )
    trajectory_reference_joints = None
    trajectory_reference_roots = None
    if trajectory_reference is not None:
        trajectory_reference_joints = np.asarray(
            trajectory_reference.joint_configurations, dtype=np.float32
        )
        trajectory_reference_joints = trajectory_reference_joints[
            :, backend_from_canonical
        ]
        if trajectory_reference.root_positions is not None:
            trajectory_reference_roots = np.concatenate(
                (
                    trajectory_reference.root_rotations_wxyz,
                    trajectory_reference.root_positions,
                ),
                axis=1,
            )

    class RootTrajectoryVar(
        jaxls.Var[jaxlie.SE3],
        default_factory=jaxlie.SE3.identity,
        retract_fn=jaxls.SE3Var.retract_fn,
        tangent_dim=jaxlie.SE3.tangent_dim,
    ):
        pass

    class BodyScaleVar(
        jaxls.Var[jax.Array],
        # Store flattened because current jaxls infers a matrix variable's
        # tangent width from only its last dimension. Body proportions are
        # constant for a clip, so all frames share this one scale matrix.
        default_factory=lambda: jnp.ones(primary_count * primary_count),
    ):
        pass

    @jaxls.Cost.factory
    def global_alignment_cost(
        values: jaxls.VarValues,
        roots: RootTrajectoryVar,
        joints: jaxls.Var[jax.Array],
        positions: jax.Array,
    ) -> jax.Array:
        local_links = jaxlie.SE3(robot.forward_kinematics(values[joints]))
        world_links = values[roots] @ local_links
        selected = jaxlie.SE3(world_links.wxyz_xyz[jnp.asarray(target_link_indices)])
        actual = selected.translation() + selected.rotation().apply(target_offsets)
        return ((actual - positions) * position_weights).flatten()

    @jaxls.Cost.factory
    def local_alignment_cost(
        values: jaxls.VarValues,
        roots: RootTrajectoryVar,
        joints: jaxls.Var[jax.Array],
        body_scale: BodyScaleVar,
        positions: jax.Array,
    ) -> jax.Array:
        local_links = jaxlie.SE3(robot.forward_kinematics(values[joints]))
        world_links = values[roots] @ local_links
        selected = jaxlie.SE3(
            world_links.wxyz_xyz[jnp.asarray(target_link_indices[:primary_count])]
        )
        actual = selected.translation() + selected.rotation().apply(
            target_offsets[:primary_count]
        )
        positions = positions[:primary_count]
        target_delta = positions[:, None, :] - positions[None, :, :]
        actual_delta = actual[:, None, :] - actual[None, :, :]
        off_diagonal = 1.0 - jnp.eye(target_delta.shape[0])
        active = connection_mask * off_diagonal
        scale = values[body_scale].reshape((primary_count, primary_count))
        vector_residual = (
            (actual_delta * scale[..., None] - target_delta)
            * active[..., None]
            * config.local_position_weight
        )
        target_direction = target_delta / jnp.linalg.norm(
            target_delta + 1.0e-6, axis=-1, keepdims=True
        )
        actual_direction = actual_delta / jnp.linalg.norm(
            actual_delta + 1.0e-6, axis=-1, keepdims=True
        )
        angle_residual = (
            (1.0 - jnp.sum(target_direction * actual_direction, axis=-1))
            * active
            * config.local_angle_weight
        )
        return jnp.concatenate((vector_residual.flatten(), angle_residual.flatten()))

    @jaxls.Cost.factory
    def scale_regularization_cost(
        values: jaxls.VarValues,
        body_scale: BodyScaleVar,
    ) -> jax.Array:
        scale = values[body_scale].reshape((primary_count, primary_count))
        close_to_one = (scale - 1.0).flatten()
        symmetric = ((scale - scale.T) * 100.0).flatten()
        non_negative = (jnp.clip(-scale, min=0.0) * 100.0).flatten()
        return jnp.concatenate((close_to_one, symmetric, non_negative))

    @jaxls.Cost.factory
    def orientation_alignment_cost(
        values: jaxls.VarValues,
        roots: RootTrajectoryVar,
        joints: jaxls.Var[jax.Array],
        rotations_wxyz: jax.Array,
    ) -> jax.Array:
        local_links = jaxlie.SE3(robot.forward_kinematics(values[joints]))
        actual = (values[roots] @ local_links).rotation()[
            jnp.asarray(target_link_indices)
        ]
        target = jaxlie.SO3(rotations_wxyz)
        return ((actual.inverse() @ target).log() * orientation_weights).flatten()

    @jaxls.Cost.factory
    def root_smoothness_cost(
        values: jaxls.VarValues,
        current: RootTrajectoryVar,
        previous: RootTrajectoryVar,
        weight: jax.Array,
    ) -> jax.Array:
        delta = (values[current].inverse() @ values[previous]).log()
        return (delta * weight).flatten()

    @jaxls.Cost.factory
    def joint_reference_cost(
        values: jaxls.VarValues,
        joints: jaxls.Var[jax.Array],
        reference: jax.Array,
        weight: jax.Array,
    ) -> jax.Array:
        return ((values[joints] - reference) * weight).flatten()

    @jaxls.Cost.factory
    def root_reference_cost(
        values: jaxls.VarValues,
        roots: RootTrajectoryVar,
        reference: jax.Array,
        weight: jax.Array,
    ) -> jax.Array:
        delta = (jaxlie.SE3(reference).inverse() @ values[roots]).log()
        return (delta * weight).flatten()

    @jaxls.Cost.factory
    def joint_velocity_limit_cost(
        values: jaxls.VarValues,
        current: jaxls.Var[jax.Array],
        previous: jaxls.Var[jax.Array],
        dt: jax.Array,
        weight: jax.Array,
    ) -> jax.Array:
        """ProtoMotions' soft, uniform 20 rad/s velocity penalty."""

        velocity = (values[current] - values[previous]) / dt
        excess = jnp.maximum(jnp.abs(velocity) - 20.0, 0.0)
        return (excess * weight).flatten()

    @jaxls.Cost.factory
    def contact_skating_cost(
        values: jaxls.VarValues,
        current_root: RootTrajectoryVar,
        current_joints: jaxls.Var[jax.Array],
        previous_root: RootTrajectoryVar,
        previous_joints: jaxls.Var[jax.Array],
        active_contacts: jax.Array,
    ) -> jax.Array:
        current_links = values[current_root] @ jaxlie.SE3(
            robot.forward_kinematics(values[current_joints])
        )
        previous_links = values[previous_root] @ jaxlie.SE3(
            robot.forward_kinematics(values[previous_joints])
        )
        current_selected = jaxlie.SE3(
            current_links.wxyz_xyz[jnp.asarray(target_link_indices)]
        )
        previous_selected = jaxlie.SE3(
            previous_links.wxyz_xyz[jnp.asarray(target_link_indices)]
        )
        current_positions = (
            current_selected.translation()
            + current_selected.rotation().apply(target_offsets)
        )
        previous_positions = (
            previous_selected.translation()
            + previous_selected.rotation().apply(target_offsets)
        )
        return (
            (current_positions - previous_positions)
            * active_contacts[:, None]
            * config.foot_skating_weight
        ).flatten()

    @jaxls.Cost.factory
    def foot_shape_cost(
        values: jaxls.VarValues,
        roots: RootTrajectoryVar,
        joints: jaxls.Var[jax.Array],
        contacts: jax.Array,
    ) -> jax.Array:
        world_links = values[roots] @ jaxlie.SE3(
            robot.forward_kinematics(values[joints])
        )
        selected = jaxlie.SE3(world_links.wxyz_xyz[jnp.asarray(target_link_indices)])
        positions = selected.translation() + selected.rotation().apply(target_offsets)
        residuals = []
        for ankle_name, foot_name in foot_shape_link_pairs:
            ankle = targets.target_link_names.index(ankle_name)
            foot = targets.target_link_names.index(foot_name)
            active = jnp.maximum(contacts[..., ankle], contacts[..., foot])
            height = (positions[..., ankle, 2] - positions[..., foot, 2]) * active
            foot_link_index = target_link_indices[foot]
            up = world_links.rotation().as_matrix()[..., foot_link_index, 2, 2]
            tilt = (up - 1.0) * active
            residuals.extend(
                (height * config.foot_skating_weight, tilt * config.foot_tilt_weight)
            )
        return jnp.stack(residuals, axis=-1)

    frame_ids = jnp.arange(frames)
    joint_vars = robot.joint_var_cls(frame_ids)
    root_vars = RootTrajectoryVar(frame_ids)
    body_scale_var = BodyScaleVar(0)
    position_weights = (
        jnp.asarray(np.asarray(targets.position_weights.cpu(), dtype=np.float32))
        * config.position_weight
    )[:, None]
    orientation_weights = (
        jnp.asarray(np.asarray(targets.rotation_weights.cpu(), dtype=np.float32))
        * config.orientation_weight
    )[:, None]
    costs = [
        global_alignment_cost(
            root_vars,
            joint_vars,
            jnp.asarray(target_positions),
        ),
        local_alignment_cost(
            root_vars,
            joint_vars,
            body_scale_var,
            jnp.asarray(target_positions),
        ),
        scale_regularization_cost(body_scale_var),
        pyroki.costs.rest_cost(
            joint_vars,
            jnp.asarray(backend_rest)[None],
            weight=jnp.asarray(backend_rest_weights)[None],
        ),
        pyroki.costs.limit_cost(
            jax.tree.map(lambda value: value[None], robot),
            joint_vars,
            jnp.asarray(config.joint_limit_weight),
        ),
    ]
    if np.any(np.asarray(targets.rotation_weights.cpu()) > 0.0):
        costs.append(
            orientation_alignment_cost(
                root_vars,
                joint_vars,
                jnp.asarray(target_rotations),
            )
        )
    if reference_frames > 0 and config.window_continuity_weight > 0.0:
        reference_weights = np.linspace(
            config.window_continuity_weight,
            config.window_continuity_weight * 0.1,
            reference_frames,
            dtype=np.float32,
        )
        reference_ids = jnp.arange(reference_frames)
        costs.extend(
            [
                joint_reference_cost(
                    robot.joint_var_cls(reference_ids),
                    jnp.asarray(backend_reference_joints),
                    jnp.asarray(reference_weights)[:, None],
                ),
                root_reference_cost(
                    RootTrajectoryVar(reference_ids),
                    jnp.asarray(reference_roots),
                    jnp.asarray(reference_weights)[:, None],
                ),
            ]
        )
    if (
        trajectory_reference_joints is not None
        and config.trajectory_joint_reference_weight > 0.0
    ):
        costs.append(
            joint_reference_cost(
                joint_vars,
                jnp.asarray(trajectory_reference_joints),
                jnp.asarray(config.trajectory_joint_reference_weight),
            )
        )
    if (
        trajectory_reference_roots is not None
        and config.trajectory_root_reference_weight > 0.0
    ):
        costs.append(
            root_reference_cost(
                root_vars,
                jnp.asarray(trajectory_reference_roots),
                jnp.asarray(config.trajectory_root_reference_weight),
            )
        )
    if frames > 1:
        costs.extend(
            [
                pyroki.costs.smoothness_cost(
                    robot.joint_var_cls(jnp.arange(1, frames)),
                    robot.joint_var_cls(jnp.arange(0, frames - 1)),
                    weight=jnp.asarray(config.joint_smoothness_weight),
                ),
                root_smoothness_cost(
                    RootTrajectoryVar(jnp.arange(1, frames)),
                    RootTrajectoryVar(jnp.arange(0, frames - 1)),
                    jnp.asarray(config.root_smoothness_weight),
                ),
            ]
        )
        if config.joint_velocity_limit_weight > 0.0:
            costs.append(
                joint_velocity_limit_cost(
                    robot.joint_var_cls(jnp.arange(1, frames)),
                    robot.joint_var_cls(jnp.arange(0, frames - 1)),
                    jnp.asarray(1.0 / targets.fps),
                    jnp.asarray(config.joint_velocity_limit_weight),
                )
            )
        if targets.contact_mask is not None:
            contacts = np.asarray(targets.contact_mask.cpu(), dtype=np.float32)
            contact_pairs = np.minimum(contacts[1:], contacts[:-1])
            costs.append(
                contact_skating_cost(
                    RootTrajectoryVar(jnp.arange(1, frames)),
                    robot.joint_var_cls(jnp.arange(1, frames)),
                    RootTrajectoryVar(jnp.arange(0, frames - 1)),
                    robot.joint_var_cls(jnp.arange(0, frames - 1)),
                    jnp.asarray(contact_pairs, dtype=jnp.float32),
                )
            )
            if foot_shape_link_pairs:
                costs.append(
                    foot_shape_cost(
                        RootTrajectoryVar(jnp.arange(frames)),
                        robot.joint_var_cls(jnp.arange(frames)),
                        jnp.asarray(contacts, dtype=jnp.float32),
                    )
                )

    initial_values = jaxls.VarValues.make(
        [
            joint_vars.with_value(jnp.asarray(backend_initial)),
            root_vars.with_value(initial_roots),
            body_scale_var.with_value(jnp.ones(primary_count * primary_count)),
        ]
    )
    solution, summary = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[joint_vars, root_vars, body_scale_var],
        )
        .analyze()
        .solve(
            initial_vals=initial_values,
            termination=jaxls.TerminationConfig(max_iterations=config.max_iterations),
            verbose=False,
            return_summary=True,
        )
    )

    solved_roots = np.asarray(solution[root_vars].wxyz_xyz, dtype=np.float32)
    backend_solution = np.asarray(solution[joint_vars], dtype=np.float32)
    canonical_solution = np.empty_like(backend_solution)
    canonical_solution[:, backend_from_canonical] = backend_solution
    initial_positions = _world_link_positions(
        model, backend_initial, initial_root_values
    )[:, target_link_indices]
    solved_positions = _world_link_positions(model, backend_solution, solved_roots)[
        :, target_link_indices
    ]
    initial_rmse = float(np.sqrt(np.mean((initial_positions - target_positions) ** 2)))
    solved_rmse = float(np.sqrt(np.mean((solved_positions - target_positions) ** 2)))

    return PyrokiTrajectoryIKResult(
        root_positions=solved_roots[:, 4:].copy(),
        root_rotations_wxyz=solved_roots[:, :4].copy(),
        joint_configurations=canonical_solution,
        initial_position_rmse=initial_rmse,
        position_rmse=solved_rmse,
        iterations=int(np.asarray(summary.iterations)),
        converged=bool(np.any(np.asarray(summary.termination_criteria))),
    )


def solve_trajectory_ik_windowed(
    model: PyrokiRobotModel,
    profile: IKRetargetConfig,
    targets: IKTargetMotion,
    *,
    initial_state: PyrokiTrajectoryState | None = None,
    trajectory_reference: PyrokiTrajectoryState | None = None,
    config: PyrokiTrajectoryIKConfig = PyrokiTrajectoryIKConfig(),
    progress_callback: PyrokiProgressCallback | None = None,
) -> PyrokiTrajectoryIKResult:
    """Solve a long clip in overlapping whole-trajectory windows."""

    window = config.window_size
    overlap = config.window_overlap
    if window < 2:
        raise ValueError("window_size must be at least 2")
    if not 0 <= overlap < window:
        raise ValueError("window_overlap must be in [0, window_size)")
    if config.window_continuity_weight < 0.0:
        raise ValueError("window_continuity_weight must be non-negative")
    step = window - overlap
    window_count = max(1, int(np.ceil((targets.num_frames - overlap) / step)))
    if config.whole_trajectory_max_frames < 1:
        raise ValueError("whole_trajectory_max_frames must be positive")
    initial_state = _prepare_trajectory_state(
        initial_state,
        frame_count=targets.num_frames,
        joint_count=len(profile.joint_order),
        name="initial_state",
        allow_single_configuration=True,
    )
    trajectory_reference = _prepare_trajectory_state(
        trajectory_reference,
        frame_count=targets.num_frames,
        joint_count=len(profile.joint_order),
        name="trajectory_reference",
    )
    if targets.num_frames <= config.whole_trajectory_max_frames:
        solved = solve_trajectory_ik(
            model,
            profile,
            targets,
            initial_state=initial_state,
            trajectory_reference=trajectory_reference,
            config=config,
        )
        if progress_callback is not None:
            progress_callback(
                PyrokiTrajectoryIKProgress(
                    window_index=1,
                    window_count=1,
                    frames_completed=targets.num_frames,
                    frame_count=targets.num_frames,
                    frame_start=0,
                    frame_end=targets.num_frames,
                    position_rmse=solved.position_rmse,
                    iterations=solved.iterations,
                    converged=solved.converged,
                )
            )
        return solved

    frame_count = targets.num_frames
    joint_output = np.empty((frame_count, len(profile.joint_order)), dtype=np.float32)
    root_position_output = np.empty((frame_count, 3), dtype=np.float32)
    root_rotation_output = np.empty((frame_count, 4), dtype=np.float32)
    weighted_initial_error = 0.0
    weighted_final_error = 0.0
    solved_frames = 0
    iterations = 0
    converged = True
    start = 0
    window_index = 0
    while start < targets.num_frames:
        window_index += 1
        end = min(start + window, targets.num_frames)
        window_targets = _slice_targets(targets, start, end)
        retained = min(overlap, end - start, solved_frames) if start > 0 else 0
        continuity_reference = None
        if retained > 0:
            continuity_reference = PyrokiTrajectoryState(
                joint_configurations=joint_output[start : start + retained].copy(),
                root_positions=root_position_output[start : start + retained].copy(),
                root_rotations_wxyz=root_rotation_output[
                    start : start + retained
                ].copy(),
            )

        window_initial = (
            None if initial_state is None else initial_state.slice(start, end)
        )
        if retained > 0:
            if window_initial is None:
                base = profile.rest_configuration
                if base is None:
                    base = np.zeros(len(profile.joint_order), dtype=np.float32)
                initial_joints = np.broadcast_to(
                    np.asarray(base, dtype=np.float32),
                    (end - start, len(profile.joint_order)),
                ).copy()
            else:
                initial_joints = np.asarray(
                    window_initial.joint_configurations, dtype=np.float32
                )
                if initial_joints.ndim == 1:
                    initial_joints = np.broadcast_to(
                        initial_joints, (end - start, initial_joints.shape[0])
                    ).copy()
                else:
                    initial_joints = initial_joints.copy()
            initial_joints[:retained] = continuity_reference.joint_configurations
            window_initial = PyrokiTrajectoryState(
                initial_joints,
                None if window_initial is None else window_initial.root_positions,
                None if window_initial is None else window_initial.root_rotations_wxyz,
            )

        solved = solve_trajectory_ik(
            model,
            profile,
            window_targets,
            initial_state=window_initial,
            continuity_reference=continuity_reference,
            trajectory_reference=(
                None
                if trajectory_reference is None
                else trajectory_reference.slice(start, end)
            ),
            config=config,
        )
        if retained > 0:
            alpha = np.linspace(0.0, 1.0, retained, dtype=np.float32)
            joint_output[start : start + retained] = (
                joint_output[start : start + retained] * (1.0 - alpha[:, None])
                + solved.joint_configurations[:retained] * alpha[:, None]
            )
            root_position_output[start : start + retained] = (
                root_position_output[start : start + retained] * (1.0 - alpha[:, None])
                + solved.root_positions[:retained] * alpha[:, None]
            )
            root_rotation_output[start : start + retained] = quat_wxyz_slerp_numpy(
                root_rotation_output[start : start + retained],
                solved.root_rotations_wxyz[:retained],
                alpha,
            ).astype(np.float32, copy=False)
        write_start = start + retained
        joint_output[write_start:end] = solved.joint_configurations[retained:]
        root_position_output[write_start:end] = solved.root_positions[retained:]
        root_rotation_output[write_start:end] = solved.root_rotations_wxyz[retained:]
        count = end - write_start
        weighted_initial_error += solved.initial_position_rmse**2 * count
        weighted_final_error += solved.position_rmse**2 * count
        solved_frames += count
        iterations += solved.iterations
        converged = converged and solved.converged
        if progress_callback is not None:
            progress_callback(
                PyrokiTrajectoryIKProgress(
                    window_index=window_index,
                    window_count=window_count,
                    frames_completed=solved_frames,
                    frame_count=targets.num_frames,
                    frame_start=start,
                    frame_end=end,
                    position_rmse=solved.position_rmse,
                    iterations=solved.iterations,
                    converged=solved.converged,
                )
            )
        if end == targets.num_frames:
            break
        start = end - overlap

    return PyrokiTrajectoryIKResult(
        root_positions=root_position_output,
        root_rotations_wxyz=root_rotation_output,
        joint_configurations=joint_output,
        initial_position_rmse=float(np.sqrt(weighted_initial_error / solved_frames)),
        position_rmse=float(np.sqrt(weighted_final_error / solved_frames)),
        iterations=iterations,
        converged=converged,
    )


def trajectory_result_to_articulation_motion(
    result: PyrokiTrajectoryIKResult,
    profile: IKRetargetConfig,
    layout: ArticulationCoordinateLayout,
    *,
    fps: float,
    motion_name: str = "PyRoki Retarget",
) -> ArticulationMotion:
    """Pack a solved PyRoki trajectory into canonical ``q``/``qd`` layout."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    frames = result.joint_configurations.shape[0]
    if result.root_positions.shape != (frames, 3):
        raise ValueError("root_positions must have shape (frames, 3)")
    if result.root_rotations_wxyz.shape != (frames, 4):
        raise ValueError("root_rotations_wxyz must have shape (frames, 4)")
    expected_joint_shape = (frames, len(profile.joint_order))
    if result.joint_configurations.shape != expected_joint_shape:
        raise ValueError(
            f"joint_configurations shape is {result.joint_configurations.shape}, "
            f"expected {expected_joint_shape}"
        )

    free_blocks = [
        block
        for block in layout.blocks
        if block.type == ArticulationCoordinateType.FREE
    ]
    if len(free_blocks) != 1:
        raise ValueError("offline humanoid retargeting requires one free-root block")
    free = free_blocks[0]
    blocks_by_joint = {
        block.joint_name: block
        for block in layout.blocks
        if block.type
        in (
            ArticulationCoordinateType.REVOLUTE,
            ArticulationCoordinateType.PRISMATIC,
        )
    }
    missing = sorted(set(profile.joint_order) - set(blocks_by_joint))
    extra = sorted(set(blocks_by_joint) - set(profile.joint_order))
    if missing or extra:
        details = []
        if missing:
            details.append("missing from layout: " + ", ".join(missing))
        if extra:
            details.append("missing from profile: " + ", ".join(extra))
        raise ValueError("canonical joint mismatch; " + "; ".join(details))

    q = np.zeros((frames, layout.nq), dtype=np.float32)
    qd = np.zeros((frames, layout.nv), dtype=np.float32)
    q[:, free.q_offset : free.q_offset + 3] = result.root_positions
    q[:, free.q_offset + 3 : free.q_offset + 7] = result.root_rotations_wxyz
    for source_index, joint_name in enumerate(profile.joint_order):
        block = blocks_by_joint[joint_name]
        q[:, block.q_offset] = result.joint_configurations[:, source_index]

    if frames > 1:
        qd[:-1, free.qd_offset : free.qd_offset + 3] = (
            np.diff(result.root_positions, axis=0) * fps
        )
        previous_inverse = result.root_rotations_wxyz[:-1].copy()
        previous_inverse[:, 1:] *= -1.0
        relative_root = quat_wxyz_multiply_numpy(
            result.root_rotations_wxyz[1:], previous_inverse
        )
        qd[:-1, free.qd_offset + 3 : free.qd_offset + 6] = (
            quat_wxyz_to_rotation_vector_numpy(relative_root) * fps
        )
        for source_index, joint_name in enumerate(profile.joint_order):
            block = blocks_by_joint[joint_name]
            delta = np.diff(result.joint_configurations[:, source_index])
            if block.type == ArticulationCoordinateType.REVOLUTE:
                delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            qd[:-1, block.qd_offset] = delta * fps
        qd[-1] = qd[-2]

    return ArticulationMotion.from_arrays(
        layout=layout,
        q=q,
        qd=qd,
        fps=fps,
        motion_name=motion_name,
    )


def retarget_motion_offline(
    source_motion,
    model: PyrokiRobotModel,
    profile: IKRetargetConfig,
    layout: ArticulationCoordinateLayout,
    *,
    initial_state: PyrokiTrajectoryState | None = None,
    contact_mask=None,
    detect_contacts: bool = True,
    contact_up_axis: int = 2,
    config: PyrokiTrajectoryIKConfig = PyrokiTrajectoryIKConfig(),
    progress_callback: PyrokiProgressCallback | None = None,
) -> PyrokiOfflineRetargetResult:
    """Retarget a coordinate-normalized skeleton clip into canonical motion."""

    if contact_mask is None and detect_contacts:
        contact_mask = detect_foot_contacts(
            source_motion, profile, up_axis=contact_up_axis
        )
    targets = build_ik_target_motion(source_motion, profile, contact_mask=contact_mask)
    solve = solve_trajectory_ik_windowed(
        model,
        profile,
        targets,
        initial_state=initial_state,
        config=config,
        progress_callback=progress_callback,
    )
    solve = _align_contact_links_to_ground(
        solve,
        model,
        profile,
        targets=targets,
        up_axis=contact_up_axis,
    )
    motion = trajectory_result_to_articulation_motion(
        solve,
        profile,
        layout,
        fps=targets.fps,
        motion_name=f"{source_motion.motion_name()} (PyRoki)",
    )
    return PyrokiOfflineRetargetResult(motion=motion, solve=solve)


def _align_contact_links_to_ground(
    result: PyrokiTrajectoryIKResult,
    model: PyrokiRobotModel,
    profile: IKRetargetConfig,
    *,
    targets: IKTargetMotion,
    up_axis: int,
) -> PyrokiTrajectoryIKResult:
    """Align detected support frames to ground with a smooth root correction."""

    if not profile.contact_links:
        return result
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")

    backend_configuration = result.joint_configurations[
        :, _backend_from_canonical(model, profile)
    ]
    roots = np.concatenate((result.root_rotations_wxyz, result.root_positions), axis=1)
    world_positions = _world_link_positions(model, backend_configuration, roots)
    try:
        contact_indices = [
            model.link_names.index(name) for name in profile.contact_links
        ]
    except ValueError as error:
        raise ValueError(
            f"contact link is absent from the loaded model: {error.args[0]}"
        ) from error
    root_positions = result.root_positions.copy()
    correction: np.ndarray
    if targets.contact_mask is not None:
        mask = np.asarray(targets.contact_mask.cpu(), dtype=np.float32)
        contact_slots = [
            targets.target_link_names.index(name) for name in profile.contact_links
        ]
        active = mask[:, contact_slots] >= 0.5
        per_frame = np.full(len(root_positions), np.nan, dtype=np.float32)
        for frame in range(len(root_positions)):
            if np.any(active[frame]):
                support_height = np.min(
                    world_positions[frame, contact_indices, up_axis][active[frame]]
                )
                per_frame[frame] = profile.root_height_offset - support_height
        valid = np.flatnonzero(np.isfinite(per_frame))
        if len(valid):
            correction = np.interp(
                np.arange(len(per_frame)), valid, per_frame[valid]
            ).astype(np.float32)
            from ...animation.filter import gaussian_filter_time

            correction = gaussian_filter_time(
                correction[:, None], sigma=max(targets.fps * 0.03, 0.5)
            )[:, 0]
        else:
            correction = np.full(
                len(root_positions),
                profile.root_height_offset
                - float(np.min(world_positions[:, contact_indices, up_axis])),
                dtype=np.float32,
            )
    else:
        correction = np.full(
            len(root_positions),
            profile.root_height_offset
            - float(np.min(world_positions[:, contact_indices, up_axis])),
            dtype=np.float32,
        )
    root_positions[:, up_axis] += correction
    return PyrokiTrajectoryIKResult(
        root_positions=root_positions,
        root_rotations_wxyz=result.root_rotations_wxyz,
        joint_configurations=result.joint_configurations,
        initial_position_rmse=result.initial_position_rmse,
        position_rmse=result.position_rmse,
        iterations=result.iterations,
        converged=result.converged,
    )


def _slice_targets(targets: IKTargetMotion, start: int, end: int) -> IKTargetMotion:
    return IKTargetMotion(
        fps=targets.fps,
        source_root_joint=targets.source_root_joint,
        target_root_link=targets.target_root_link,
        target_link_names=targets.target_link_names,
        target_offsets=targets.target_offsets,
        primary_effector_count=targets.primary_effector_count,
        positions=targets.positions[start:end],
        rotations_wxyz=targets.rotations_wxyz[start:end],
        position_weights=targets.position_weights,
        rotation_weights=targets.rotation_weights,
        contact_mask=(
            None if targets.contact_mask is None else targets.contact_mask[start:end]
        ),
        initial_root_rotations_wxyz=(
            None
            if targets.initial_root_rotations_wxyz is None
            else targets.initial_root_rotations_wxyz[start:end]
        ),
    )


def _world_link_positions(
    model: PyrokiRobotModel,
    backend_configuration: npt.ArrayLike,
    root_wxyz_xyz: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    import jax.numpy as jnp
    import jaxlie

    root_values = jnp.asarray(root_wxyz_xyz)
    if root_values.ndim == 2:
        root_values = root_values[:, None, :]
    root = jaxlie.SE3(root_values)
    local = jaxlie.SE3(jnp.asarray(model.forward_kinematics(backend_configuration)))
    return np.asarray((root @ local).translation(), dtype=np.float32)


__all__ = [
    "PyrokiPoseIKConfig",
    "PyrokiPoseIKResult",
    "PyrokiOfflineRetargetResult",
    "PyrokiTrajectoryIKConfig",
    "PyrokiTrajectoryIKProgress",
    "PyrokiTrajectoryIKResult",
    "PyrokiTrajectoryState",
    "solve_pose_ik",
    "solve_trajectory_ik",
    "solve_trajectory_ik_windowed",
    "trajectory_result_to_articulation_motion",
    "retarget_motion_offline",
]
