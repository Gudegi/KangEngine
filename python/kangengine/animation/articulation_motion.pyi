"""Native articulation motion coordinate metadata."""

from __future__ import annotations

from enum import Enum
import numpy as np
import numpy.typing as npt

from .. import Quat, Vec3
from . import SkeletonMotion, SkeletonState
from ..asset import ArticulationDesc

class ArticulationCoordinateType(Enum):
    FIXED: ArticulationCoordinateType
    REVOLUTE: ArticulationCoordinateType
    PRISMATIC: ArticulationCoordinateType
    SPHERICAL: ArticulationCoordinateType
    FREE: ArticulationCoordinateType

class ArticulationCoordinateBlock:
    body_index: int
    parent_body_index: int
    joint_index_in_body: int
    body_name: str
    joint_name: str
    type: ArticulationCoordinateType
    axis: Vec3
    reference_translation: Vec3
    reference_rotation: Quat
    lower_limit: float
    upper_limit: float
    q_offset: int
    q_size: int
    qd_offset: int
    qd_size: int

class ArticulationCoordinateLayout:
    @staticmethod
    def from_data(
        data: ArticulationDesc,
        free_root: bool = False,
    ) -> ArticulationCoordinateLayout: ...
    @property
    def nq(self) -> int: ...
    @property
    def nv(self) -> int: ...
    @property
    def has_free_root(self) -> bool: ...
    @property
    def traversal_order(self) -> str: ...
    @property
    def model_signature(self) -> str: ...
    @property
    def blocks(self) -> list[ArticulationCoordinateBlock]: ...

class ArticulationMappingResult:
    q: list[float]
    residual_angles: list[float]

class ArticulationMotionMappingResult:
    motion: ArticulationMotion
    residual_angles: list[float]

class ArticulationMotionMapper:
    def __init__(self, layout: ArticulationCoordinateLayout) -> None: ...
    def to_articulation_coordinates(
        self, state: SkeletonState, clamp_to_limits: bool = False
    ) -> ArticulationMappingResult: ...
    def to_skeleton_state(self, q: npt.ArrayLike) -> SkeletonState: ...
    def to_articulation_motion(
        self, motion: SkeletonMotion, clamp_to_limits: bool = False
    ) -> ArticulationMotionMappingResult: ...
    def to_skeleton_motion(self, motion: ArticulationMotion) -> SkeletonMotion: ...
    @property
    def layout(self) -> ArticulationCoordinateLayout: ...

class ArticulationMotion:
    @staticmethod
    def from_arrays(
        layout: ArticulationCoordinateLayout,
        q: npt.ArrayLike,
        qd: npt.ArrayLike,
        fps: float,
        motion_name: str = "Motion",
    ) -> ArticulationMotion: ...
    def num_frames(self) -> int: ...
    def fps(self) -> float: ...
    def duration(self) -> float: ...
    def motion_name(self) -> str: ...
    @property
    def layout(self) -> ArticulationCoordinateLayout: ...
    @property
    def q(self) -> npt.NDArray[np.float32]: ...
    @property
    def qd(self) -> npt.NDArray[np.float32]: ...
