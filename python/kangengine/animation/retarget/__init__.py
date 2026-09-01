"""Angle and IK motion-retargeting APIs."""

from .angle import AngleRetargetConfig, AngleRetargeter, retarget_angle_motion
from .angle_processor import AngleRetargetProcessor, AngleRetargetResult
from .ik import (
    AuxiliaryRetargetPoint,
    IKEffectorMapping,
    IKTargetMotion,
    build_ik_target_motion,
    detect_foot_contacts,
)
from .ik_processor import IKRetargetProcessor, IKRetargetResult
from .ik_profile import IKRetargetConfig, IKTargetProfile
from .io import load_retarget_motion, scale_skeleton_motion
from .profile import AngleTargetProfile, MotionSourceProfile

__all__ = [
    "AngleRetargetConfig",
    "AngleRetargetProcessor",
    "AngleRetargetResult",
    "AngleRetargeter",
    "AngleTargetProfile",
    "AuxiliaryRetargetPoint",
    "IKEffectorMapping",
    "IKRetargetConfig",
    "IKRetargetProcessor",
    "IKRetargetResult",
    "IKTargetMotion",
    "IKTargetProfile",
    "MotionSourceProfile",
    "build_ik_target_motion",
    "detect_foot_contacts",
    "load_retarget_motion",
    "retarget_angle_motion",
    "scale_skeleton_motion",
]
