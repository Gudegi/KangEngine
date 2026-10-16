"""Animation typing. Shapes use F=frames, S=skeleton nodes, B=skin bones, V=vertices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, overload

import numpy as np
import numpy.typing as npt
import torch

from .. import Quat, Vec3
from . import IK as IK, datasets as datasets, filter as filter
from .coordinates import (
    CoordinateSystem as CoordinateSystem,
    convert_motion_coordinates as convert_motion_coordinates,
    convert_skeleton_coordinates as convert_skeleton_coordinates,
    convert_state_coordinates as convert_state_coordinates,
)
from .transform import transform_motion as transform_motion
from .contact_detector import FootContactDetector as FootContactDetector
from .modules import FootContactModule as FootContactModule
from .articulation_motion import (
    ArticulationCoordinateBlock as ArticulationCoordinateBlock,
    ArticulationCoordinateLayout as ArticulationCoordinateLayout,
    ArticulationCoordinateType as ArticulationCoordinateType,
    ArticulationMappingResult as ArticulationMappingResult,
    ArticulationMotion as ArticulationMotion,
    ArticulationMotionMapper as ArticulationMotionMapper,
    ArticulationMotionMappingResult as ArticulationMotionMappingResult,
)
from .articulation_io import (
    load_articulation_motion_npz as load_articulation_motion_npz,
    save_articulation_motion_npz as save_articulation_motion_npz,
)
from .retarget import (
    AngleRetargetConfig as AngleRetargetConfig,
    AngleRetargetProcessor as AngleRetargetProcessor,
    AngleRetargetResult as AngleRetargetResult,
    AngleRetargeter as AngleRetargeter,
    AngleTargetProfile as AngleTargetProfile,
    AuxiliaryRetargetPoint as AuxiliaryRetargetPoint,
    IKEffectorMapping as IKEffectorMapping,
    IKRetargetConfig as IKRetargetConfig,
    IKRetargetProcessor as IKRetargetProcessor,
    IKRetargetResult as IKRetargetResult,
    IKTargetProfile as IKTargetProfile,
    MotionSourceProfile as MotionSourceProfile,
    retarget_angle_motion as retarget_angle_motion,
    scale_skeleton_motion as scale_skeleton_motion,
)
from .motion_library import (
    MotionKinematics as MotionKinematics,
    MotionLibrary as MotionLibrary,
    MotionLibraryBuffers as MotionLibraryBuffers,
    MotionSample as MotionSample,
)
from .matching import (
    MotionMatchingConfig as MotionMatchingConfig,
    MotionMatchingDatabase as MotionMatchingDatabase,
    MotionMatchingController as MotionMatchingController,
    WalkingFeatures as WalkingFeatures,
)

_Float32Array: TypeAlias = npt.NDArray[np.float32]

def trajectory_spring_position(
    position: npt.ArrayLike,
    velocity: npt.ArrayLike,
    acceleration: npt.ArrayLike,
    *,
    desired_velocity: npt.ArrayLike,
    half_life: float,
    dt: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Predict [3] vectors, or [H,3] arrays for independent time horizons."""
    ...

def trajectory_spring_rotation(
    rotation_wxyz: npt.ArrayLike,
    angular_velocity: npt.ArrayLike,
    *,
    desired_rotation_wxyz: npt.ArrayLike,
    half_life: float,
    dt: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Predict WXYZ rotation and spatial angular velocity at one or H times."""
    ...

class SkeletonTree:
    """Read-only skeleton hierarchy and local bind transforms."""

    def __init__(
        self,
        names: list[str],
        parents: list[int],
        local_translations: npt.ArrayLike | torch.Tensor,
        local_rotations_wxyz: npt.ArrayLike | torch.Tensor,
    ) -> None:
        """Create a skeleton from translations and WXYZ bind rotations."""
        ...

    def num_joints(self) -> int:
        """Return S, the number of skeleton nodes."""
        ...
    def index(self, name: str) -> int:
        """Return the node index for a name.

        Raises:
            RuntimeError: If the name is not in the skeleton tree.
        """
        ...
    def node_name(self, index: int) -> str:
        """Return a skeleton node name."""
        ...
    def parent_index(self, index: int) -> int:
        """Return the parent node index, or -1 for the root."""
        ...
    def local_translation(self, index: int) -> Vec3:
        """Return a node's local bind translation."""
        ...
    def local_rotation(self, index: int) -> Quat:
        """Return a node's local bind rotation."""
        ...
    def node_names(self) -> list[str]:
        """Return S skeleton node names."""
        ...
    def parent_indices(self) -> list[int]:
        """Return S parent indices."""
        ...
    def print(self) -> None:
        """Print the hierarchy for debugging."""
        ...

class SkeletonMotion:
    """Sampled skeleton animation with root motion and local rotations."""

    @staticmethod
    def from_arrays(
        skeleton_tree: SkeletonTree,
        root_translations: npt.ArrayLike | torch.Tensor,
        local_rotations_wxyz: npt.ArrayLike | torch.Tensor,
        fps: float,
        motion_name: str = "Motion",
    ) -> SkeletonMotion:
        """Create a motion from root ``(F, 3)`` and WXYZ rotations ``(F, J, 4)``."""
        ...
    def num_frames(self) -> int:
        """Return F, the number of frames."""
        ...
    def num_joints(self) -> int:
        """Return S, the number of skeleton nodes."""
        ...
    def fps(self) -> float:
        """Return frames per second."""
        ...
    def duration(self) -> float:
        """Return duration in seconds."""
        ...
    def motion_name(self) -> str:
        """Return the clip name."""
        ...
    def node_names(self) -> list[str]:
        """Return S skeleton node names."""
        ...
    def parent_indices(self) -> list[int]:
        """Return S parent indices."""
        ...
    @property
    def skeleton_tree(self) -> SkeletonTree:
        """Return the motion's read-only skeleton hierarchy."""
        ...
    def frame(self, frame_index: int) -> SkeletonState:
        """Return the pose at one frame."""
        ...
    def sample(self, time: float, loop: bool = True) -> SkeletonState:
        """Sample a pose at time in seconds."""
        ...
    def root_translation(self, frame: int) -> Vec3:
        """Return the root translation at one frame."""
        ...
    def local_rotation(self, frame: int, joint: int) -> Quat:
        """Return one node's local rotation at one frame."""
        ...
    def root_translations_flat(self) -> list[float]:
        """Return flattened root translations with logical shape ``(F, 3)``."""
        ...
    def local_rotations_wxyz_flat(self) -> list[float]:
        """Return flattened WXYZ rotations with logical shape ``(F, S, 4)``."""
        ...
    def root_translations(self) -> _Float32Array:
        """Return root translations with shape ``(F, 3)``."""
        ...
    def local_rotations_wxyz(self) -> _Float32Array:
        """Return WXYZ local rotations with shape ``(F, J, 4)``."""
        ...
    def global_matrices(self) -> _Float32Array:
        """Compute global transforms with shape ``(F, J, 4, 4)``."""
        ...
    def global_positions(self) -> _Float32Array:
        """Compute global joint positions with shape ``(F, J, 3)``."""
        ...
    def global_rotations_wxyz(self) -> _Float32Array:
        """Compute WXYZ global joint rotations with shape ``(F, J, 4)``."""
        ...
    def root_linear_velocities(self) -> _Float32Array:
        """Compute root linear velocities with shape ``(F, 3)``."""
        ...
    def root_linear_accelerations(self) -> _Float32Array:
        """Compute root linear accelerations with shape ``(F, 3)``."""
        ...
    def global_linear_velocities(self) -> _Float32Array:
        """Compute global joint linear velocities with shape ``(F, J, 3)``."""
        ...
    def global_angular_velocities(self) -> _Float32Array:
        """Compute global joint angular velocities with shape ``(F, J, 3)``."""
        ...
    def global_linear_accelerations(self) -> _Float32Array:
        """Compute global joint linear accelerations with shape ``(F, J, 3)``."""
        ...
    def global_angular_accelerations(self) -> _Float32Array:
        """Compute global joint angular accelerations with shape ``(F, J, 3)``."""
        ...

class Transform:
    """Forward-kinematics transform result."""

    @property
    def rotation(self) -> Quat:
        """Return the transform rotation."""
        ...
    @property
    def translation(self) -> Vec3:
        """Return the transform translation."""
        ...

class SkeletonState:
    """Skeleton pose stored as root translation and S node rotations."""

    @staticmethod
    def from_rotation_and_root_translation(
        tree: SkeletonTree,
        rotations_wxyz: npt.ArrayLike | torch.Tensor,
        root_translation: npt.ArrayLike | torch.Tensor,
        is_local: bool = True,
    ) -> SkeletonState:
        """Create a pose from WXYZ rotations ``(S, 4)`` and root ``(3,)``."""
        ...
    def num_joints(self) -> int:
        """Return S, the number of skeleton nodes."""
        ...
    @property
    def skeleton_tree(self) -> SkeletonTree:
        """Return the state's read-only skeleton hierarchy."""
        ...
    def is_local(self) -> bool:
        """Return whether rotations are local to parent nodes."""
        ...
    def compute_global_transforms(self) -> list[Transform]:
        """Return S global forward-kinematics transforms."""
        ...
    def compute_global_matrices(self) -> _Float32Array:
        """Return global float32 matrices with shape ``(S, 4, 4)``."""
        ...
    def compute_global_positions(self) -> list[Vec3]:
        """Return S global node positions."""
        ...
    def get_joint_quat(self, index: int) -> Quat:
        """Return one stored node rotation."""
        ...
    @overload
    def set_joint_quat_wxyz(
        self, index: int, quat_wxyz: npt.ArrayLike | torch.Tensor
    ) -> None:
        """Set a skeleton node's quaternion without changing coordinate frames.

        Args:
            index: Skeleton node index, including the root.
            quat_wxyz: WXYZ quaternion of shape ``(4,)`` in the state's
                local or global convention, as indicated by ``is_local()``.
        """
        ...
    @overload
    def set_joint_quat_wxyz(self, index: int, quat_wxyz: Quat) -> None: ...
    def root_translation(self) -> Vec3:
        """Return the root translation."""
        ...
    @overload
    def set_root_translation(self, translation: npt.ArrayLike | torch.Tensor) -> None:
        """Set the root translation from shape ``(3,)``."""
        ...
    @overload
    def set_root_translation(self, translation: Vec3) -> None: ...
    def print_global_positions(self) -> None:
        """Print global node positions for debugging."""
        ...

class MotionMatch:
    """Immutable search result; cost includes the continuation bias."""

    def __init__(
        self, *, row: int, motion_index: int, frame_index: int,
        time: float, cost: float,
    ) -> None: ...
    @property
    def row(self) -> int:
        """Return the database candidate row."""
        ...
    @property
    def motion_index(self) -> int:
        """Return the source clip index."""
        ...
    @property
    def frame_index(self) -> int:
        """Return the frame index within the source clip."""
        ...
    @property
    def time(self) -> float:
        """Return clip-local time in seconds."""
        ...
    @property
    def cost(self) -> float:
        """Return squared feature distance minus bias; it may be negative."""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class MotionMatchingIndex:
    """Owned float64 feature storage with exact nearest-candidate search."""

    def __init__(
        self, features: npt.ArrayLike, *, clip_frame_counts: Sequence[int],
        clip_candidate_counts: Sequence[int], clip_fps: Sequence[float],
    ) -> None:
        """Copy normalized, weighted features and per-clip metadata.

        Args:
            features: Nonempty finite array of shape (candidate rows, dimensions).
            clip_frame_counts: Total sampled frames per clip, at least two.
            clip_candidate_counts: Candidate prefix lengths, positive and less
                than frame counts; their sum must equal the feature row count.
            clip_fps: Positive finite sample rates, one per clip.
        """
        ...
    @property
    def num_frames(self) -> int:
        """Return the total number of candidate rows."""
        ...
    @property
    def feature_dimension(self) -> int:
        """Return the number of values in each feature vector."""
        ...
    @property
    def num_motions(self) -> int:
        """Return the number of source clips."""
        ...
    def search(
        self, query: npt.ArrayLike, *, current_motion_index: int | None = None,
        current_time: float | None = None, continuation_bias: float = 0.0,
        allowed_motion_indices: Sequence[int] | None = None,
    ) -> MotionMatch:
        """Find the lowest-cost candidate; ties select the lowest database row.

        Args:
            query: Finite normalized, weighted vector of shape (feature_dimension,).
            current_motion_index: Current clip; supply together with current_time.
            current_time: Clip-local seconds in the sampled clip interval.
            continuation_bias: Nonnegative cost reduction for the current row;
                a nonzero value requires the current clip and time.
            allowed_motion_indices: None searches all clips; an empty list is invalid.
        """
        ...

class SkeletonPoseSample:
    """Local joint pose with world root placement and joint velocities."""

    def __init__(
        self, state: SkeletonState, *, root_linear_velocity: npt.ArrayLike,
        local_angular_velocities: npt.ArrayLike,
    ) -> None:
        """Copy a pose and its velocities.

        Args:
            state: Local joint rotations with the root pose in world space.
            root_linear_velocity: World velocity (3,) in scene units per second.
            local_angular_velocities: Angular velocities (S, 3) in radians per
                second, expressed in parent frames (world space for the root).
        """
        ...
    @property
    def state(self) -> SkeletonState:
        """Return a copy of the pose."""
        ...
    @property
    def root_linear_velocity(self) -> _Float32Array:
        """Return a world velocity copy (3,) in scene units per second."""
        ...
    @property
    def local_angular_velocities(self) -> _Float32Array:
        """Return velocities (S, 3) in rad/s in parent frames; root uses world."""
        ...

class SkeletonPoseSmoother:
    """Decay transition offsets while following supplied poses and velocities."""

    def __init__(
        self, initial: SkeletonPoseSample, *, half_life: float = 0.1,
    ) -> None: ...
    def reset(self, target: SkeletonPoseSample) -> None:
        """Snap to the target and clear transition offsets."""
        ...
    def transition(self, target: SkeletonPoseSample) -> None:
        """Switch targets while preserving output pose and velocity unless disabled."""
        ...
    def advance(self, target: SkeletonPoseSample, dt: float) -> None:
        """Follow a target already advanced by dt seconds; dt=0 only validates."""
        ...
    @property
    def half_life(self) -> float:
        """Return the nonnegative spring decay setting in seconds; zero disables it."""
        ...
    @half_life.setter
    def half_life(self, value: float) -> None: ...
    @property
    def sample(self) -> SkeletonPoseSample:
        """Return an independent snapshot of the output pose and velocities."""
        ...

class SkeletonMotionPlayer:
    """Play immutable clips with root alignment and inertial transitions."""

    def __init__(
        self, motions: Sequence[SkeletonMotion], *, up_axis: str = "Z",
        forward_axis: str = "X", transition_half_life: float = 0.1,
    ) -> None:
        """Start clip 0 with walking alignment, or source placement if heading is undefined."""
        ...
    def reset(self, motion_index: int = 0, time: float = 0.0) -> None:
        """Reset at the planar origin and clip-local seconds, clearing smoothing."""
        ...
    def transition(self, motion_index: int, time: float) -> None:
        """Switch at clip-local seconds, aligning to the current position and heading."""
        ...
    def reset_placed(
        self, motion_index: int, time: float, *, rotation_wxyz: npt.ArrayLike,
        translation: npt.ArrayLike,
    ) -> None:
        """Reset at clip-local seconds using a source-to-world rigid transform.

        Args:
            motion_index: Source clip index.
            time: Clip-local seconds.
            rotation_wxyz: Source-to-world unit quaternion (4,) in WXYZ order.
            translation: Source-to-world translation (3,) in scene units.
        """
        ...
    def transition_placed(
        self, motion_index: int, time: float, *, rotation_wxyz: npt.ArrayLike,
        translation: npt.ArrayLike,
    ) -> None:
        """Switch clips with source-to-world placement before smoothing.

        Args:
            motion_index: Source clip index.
            time: Clip-local seconds.
            rotation_wxyz: Source-to-world unit quaternion (4,) in WXYZ order.
            translation: Source-to-world translation (3,), not a target root position.
        """
        ...
    def advance(self, dt: float) -> None:
        """Advance by finite nonnegative seconds; clip time clamps at its end."""
        ...
    @property
    def state(self) -> SkeletonState:
        """Return an independent snapshot of the output pose."""
        ...
    @property
    def motion_index(self) -> int:
        """Return the current clip index."""
        ...
    @property
    def time(self) -> float:
        """Return the current clip-local time in seconds."""
        ...
    @property
    def duration(self) -> float:
        """Return the current clip duration in seconds."""
        ...
    @property
    def transition_half_life(self) -> float:
        """Return the spring decay setting in seconds; zero disables smoothing."""
        ...
    @transition_half_life.setter
    def transition_half_life(self, value: float) -> None: ...
    @property
    def root_position(self) -> _Float32Array:
        """Return world root position (3,) in scene units."""
        ...
    @property
    def heading_rotation_wxyz(self) -> _Float32Array:
        """Return the world heading quaternion (4,) in WXYZ order."""
        ...
    @property
    def root_linear_velocity(self) -> _Float32Array:
        """Return world root velocity (3,) in scene units per second."""
        ...
    @property
    def local_angular_velocities(self) -> _Float32Array:
        """Return velocities (S, 3) in rad/s in parent frames; root uses world."""
        ...

def solve_full_body_ik(
    state: SkeletonState,
    targets: npt.ArrayLike | torch.Tensor,
    effector_joints: npt.ArrayLike | torch.Tensor,
    effector_offsets: npt.ArrayLike | torch.Tensor,
    control_joints: npt.ArrayLike | torch.Tensor,
    control_axes: npt.ArrayLike | torch.Tensor,
    max_iterations: int = 0,
) -> tuple[SkeletonState, _Float32Array, _Float32Array, int]: ...
def solve_full_body_ik_batch(
    motion: SkeletonMotion,
    targets: npt.ArrayLike | torch.Tensor,
    effector_joints: npt.ArrayLike | torch.Tensor,
    effector_offsets: npt.ArrayLike | torch.Tensor,
    control_joints: npt.ArrayLike | torch.Tensor,
    control_axes: npt.ArrayLike | torch.Tensor,
    max_iterations: int = 0,
) -> tuple[SkeletonMotion, _Float32Array, _Float32Array, npt.NDArray[np.int32]]: ...
def cpu_skin(
    bind_positions: npt.ArrayLike | torch.Tensor,
    bind_normals: npt.ArrayLike | torch.Tensor,
    skin_bone_indices: npt.ArrayLike | torch.Tensor,
    skin_bone_weights: npt.ArrayLike | torch.Tensor,
    skin_bone_node_indices: npt.ArrayLike | torch.Tensor,
    inverse_bind_matrices: npt.ArrayLike | torch.Tensor,
    skeleton_global_matrices: npt.ArrayLike | torch.Tensor,
) -> dict[str, _Float32Array]:
    """CPU-skin bind-space positions and normals.

    Shapes use V=vertices, B=skin bones, and S=skeleton nodes.

    Args:
        bind_positions: Float32 vertex positions with shape ``(V, 3)``.
        bind_normals: Float32 vertex normals with shape ``(V, 3)``.
        skin_bone_indices: Int32 skin-bone indices with shape ``(V, 4)``.
        skin_bone_weights: Float32 skin-bone weights with shape ``(V, 4)``.
        skin_bone_node_indices: Bone-to-node map with shape ``(B,)``.
        inverse_bind_matrices: Float32 transforms with shape ``(B, 4, 4)``.
        skeleton_global_matrices: Skeleton globals with shape ``(S, 4, 4)``.

    Returns:
        Float32 ``positions`` and ``normals`` arrays with shape ``(V, 3)``.
    """
    ...

def compute_skinning_matrices(
    skin_bone_node_indices: npt.ArrayLike | torch.Tensor,
    inverse_bind_matrices: npt.ArrayLike | torch.Tensor,
    skeleton_global_matrices: npt.ArrayLike | torch.Tensor,
) -> _Float32Array:
    """Return skinning matrices ``(B, 4, 4)`` from B skin bones and S nodes."""
    ...

def compute_skinning_matrices_into(
    skin_bone_node_indices: npt.ArrayLike | torch.Tensor,
    inverse_bind_matrices: npt.ArrayLike | torch.Tensor,
    skeleton_global_matrices: npt.ArrayLike | torch.Tensor,
    output: npt.ArrayLike | torch.Tensor,
) -> _Float32Array:
    """Write and return skinning matrices using output shape ``(B, 4, 4)``."""
    ...
