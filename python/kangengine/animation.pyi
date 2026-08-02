"""Animation typing. Shapes use F=frames, S=skeleton nodes, B=skin bones, V=vertices."""

from __future__ import annotations

from typing import Any, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import torch

_Float32Array: TypeAlias = npt.NDArray[np.float32]

class SkeletonTree:
    """Read-only skeleton hierarchy and local bind transforms."""

    def num_joints(self) -> int:
        """Return S, the number of skeleton nodes."""
        ...
    def node_name(self, index: int) -> str:
        """Return a skeleton node name."""
        ...
    def parent_index(self, index: int) -> int:
        """Return the parent node index, or -1 for the root."""
        ...
    def local_translation(self, index: int) -> Any:
        """Return a node's local bind translation."""
        ...
    def local_rotation(self, index: int) -> Any:
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
    def frame(self, frame_index: int) -> SkeletonState:
        """Return the pose at one frame."""
        ...
    def sample(self, time: float, loop: bool = True) -> SkeletonState:
        """Sample a pose at time in seconds."""
        ...
    def root_translation(self, frame: int) -> Any:
        """Return the root translation at one frame."""
        ...
    def local_rotation(self, frame: int, joint: int) -> Any:
        """Return one node's local rotation at one frame."""
        ...
    def root_translations_flat(self) -> list[float]:
        """Return flattened root translations with logical shape ``(F, 3)``."""
        ...
    def local_rotations_wxyz_flat(self) -> list[float]:
        """Return flattened WXYZ rotations with logical shape ``(F, S, 4)``."""
        ...

class Transform:
    """Forward-kinematics transform result."""

    @property
    def rotation(self) -> Any:
        """Return the transform rotation."""
        ...
    @property
    def translation(self) -> Any:
        """Return the transform translation."""
        ...

class SkeletonState:
    """Skeleton pose stored as root translation and S node rotations."""

    @staticmethod
    def from_rotation_and_root_translation(
        tree: SkeletonTree,
        rotations: npt.ArrayLike | torch.Tensor,
        root_translation: npt.ArrayLike | torch.Tensor,
        is_local: bool = True,
    ) -> SkeletonState:
        """Create a pose from XYZW rotations ``(S, 4)`` and root ``(3,)``."""
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
    def compute_global_positions(self) -> list[Any]:
        """Return S global node positions."""
        ...
    def rotation(self, index: int) -> Any:
        """Return one stored node rotation."""
        ...
    @overload
    def set_rotation(self, index: int, rotation: npt.ArrayLike | torch.Tensor) -> None:
        """Set one node rotation from an XYZW array with shape ``(4,)``."""
        ...
    @overload
    def set_rotation(self, index: int, rotation: Any) -> None: ...
    def root_translation(self) -> Any:
        """Return the root translation."""
        ...
    @overload
    def set_root_translation(self, translation: npt.ArrayLike | torch.Tensor) -> None:
        """Set the root translation from shape ``(3,)``."""
        ...
    @overload
    def set_root_translation(self, translation: Any) -> None: ...
    def print_global_positions(self) -> None:
        """Print global node positions for debugging."""
        ...

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
