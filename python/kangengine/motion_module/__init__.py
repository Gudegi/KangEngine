"""Motion-editor modules and their shared lifecycle contract."""

from __future__ import annotations

from .modules import (
    ContactModule,
    MotionModule,
    RootTrajectoryModule,
    TargetModule,
    TrackingModule,
)
from .._public import set_public_module

for _type in (
    ContactModule,
    MotionModule,
    RootTrajectoryModule,
    TargetModule,
    TrackingModule,
):
    set_public_module(_type, __name__)

del _type

__all__ = [
    "ContactModule",
    "MotionModule",
    "RootTrajectoryModule",
    "TargetModule",
    "TrackingModule",
]
