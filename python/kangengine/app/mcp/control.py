"""Owner-thread controls shared by local automation transports."""

import math
from pathlib import Path

import numpy as np

from ..capture import AutomationError
from .inspection import _resolve_prim


def camera_get(app):
    camera = app.get_camera()
    position = camera.get_camera_pos()
    target = camera.get_target_pos()
    return {
        "position": [position.x, position.y, position.z],
        "target": [target.x, target.y, target.z],
        "coordinate_frame": "world",
        "units": "scene_unit",
        "active_scene_camera_path": app.active_scene_camera_path(),
        "camera": "navigation",
    }


def camera_set(app, *, position, target):
    if app.has_active_scene_camera():
        raise AutomationError(
            "UNSUPPORTED", "Clear the active scene camera before controlling navigation"
        )
    # Validate float32 coordinates before either setter can mutate camera state.
    with np.errstate(over="ignore", invalid="ignore"):
        position = np.asarray(position, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
    if (
        position.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(target).all()
    ):
        raise AutomationError(
            "INVALID_ARGUMENT", "position and target need three finite values"
        )
    direction = target.astype(float) - position.astype(float)
    distance = float(np.linalg.norm(direction))
    # Native camera normalization squares float32 components.
    safe_component = math.sqrt(np.finfo(np.float32).max) / 4
    if (
        np.max(np.abs(position)) > safe_component
        or np.max(np.abs(target)) > safe_component
    ):
        raise AutomationError(
            "INVALID_ARGUMENT", "Camera coordinates exceed float32 view precision"
        )
    if distance < 1e-4:
        raise AutomationError(
            "INVALID_ARGUMENT", "Camera position and target must be distinct"
        )
    app.set_camera_view(position=position, target=target)
    return camera_get(app)


def scene_set_visibility(app, *, path, visible):
    if type(visible) is not bool:
        raise AutomationError("INVALID_ARGUMENT", "visible must be a boolean")
    prim = _resolve_prim(app, path)
    previous = prim.is_visible()
    prim.set_visible(visible)
    return {
        "path": path,
        "previous_visible": previous,
        "visible": prim.is_visible(),
        "visible_in_hierarchy": prim.is_visible_in_hierarchy(),
        "active": prim.is_active(),
        "effective_visible": prim.is_visible_in_hierarchy()
        and prim.is_active_in_hierarchy(),
    }


def recording_status(app):
    path = app.get_video_recording_path()
    return {
        "recording": app.is_video_recording(),
        "output_path": str(path) if path is not None else None,
    }


def recording_start(app, *, output_dir, output_path, fps=30.0):
    if app.run_config is not None and not app.run_config.render_enabled:
        raise AutomationError("UNSUPPORTED", "Recording requires rendering")
    if app.is_video_recording():
        raise AutomationError("BUSY", "Video recording is already active")
    if app.frame_index == 0:
        raise AutomationError("NOT_READY", "Render a frame before starting recording")
    if isinstance(fps, bool) or not math.isfinite(fps) or not 1 <= fps <= 240:
        raise AutomationError("INVALID_ARGUMENT", "fps must be finite and in 1..240")
    root = Path(output_dir).resolve()
    path = Path(output_path).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".mp4":
        raise AutomationError(
            "INVALID_ARGUMENT",
            "output_path must be an MP4 inside the recording directory",
        )
    if path.exists():
        raise AutomationError("INVALID_ARGUMENT", "Recording output already exists")
    try:
        app.start_video_recording(output_path=path, fps=fps)
    except Exception:
        # The common controller may activate before the encoder fails to open.
        app.stop_video_recording()
        raise
    return recording_status(app)


def recording_stop(app):
    app.stop_video_recording()
    return recording_status(app)
