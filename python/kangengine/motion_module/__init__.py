"""Motion-editor modules and their shared lifecycle contract."""

from __future__ import annotations

from .editor import (
    ContactData,
    MotionCameraFollower,
    MotionEditor,
    MotionPlayer,
    MotionSampleData,
    RootTrajectoryData,
    TrackingData,
)
from .modules import (
    ContactModule,
    MotionModule,
    RootTrajectoryModule,
    TargetModule,
    TrackingModule,
)
from .._public import set_public_module

for _type in (
    ContactData,
    ContactModule,
    MotionCameraFollower,
    MotionEditor,
    MotionModule,
    MotionPlayer,
    MotionSampleData,
    RootTrajectoryData,
    RootTrajectoryModule,
    TargetModule,
    TrackingData,
    TrackingModule,
):
    set_public_module(_type, __name__)

del _type

__all__ = [
    "ContactData",
    "ContactModule",
    "MotionCameraFollower",
    "MotionEditor",
    "MotionModule",
    "MotionPlayer",
    "MotionSampleData",
    "RootTrajectoryData",
    "RootTrajectoryModule",
    "TargetModule",
    "TrackingData",
    "TrackingModule",
]
