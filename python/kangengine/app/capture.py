"""Copied scene-frame results shared by local automation clients."""

from __future__ import annotations

from dataclasses import dataclass


class AutomationError(RuntimeError):
    """An engine operation failed with a stable client-facing code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Postprocessed RGB PNG, top-left origin, excluding application UI.

    ``frame_id`` is the native renderer frame index within ``app_session_id``.
    Capturing observes a normal frame and does not add simulation updates.
    """

    png: bytes
    width: int
    height: int
    frame_id: int
    app_session_id: str
    output_path: str | None = None
