"""Recording utilities independent of simulation and rendering ownership."""

from .app_video_capture import VideoCaptureController
from .video_recorder import VideoRecorder

__all__ = ["VideoCaptureController", "VideoRecorder"]
