"""Animation typing. Shapes use F=frames, S=skeleton nodes, B=skin bones, V=vertices."""

from __future__ import annotations

from typing import TypeAlias, overload

import numpy as np
import numpy.typing as npt
import torch

from . import Quat, Vec3

_Float32Array: TypeAlias = npt.NDArray[np.float32]

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
    def rotation(self, index: int) -> Quat:
        """Return one stored node rotation."""
        ...
    @overload
    def set_rotation(self, index: int, rotation: npt.ArrayLike | torch.Tensor) -> None:
        """Set one node rotation from an XYZW array with shape ``(4,)``."""
        ...
    @overload
    def set_rotation(self, index: int, rotation: Quat) -> None: ...
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
