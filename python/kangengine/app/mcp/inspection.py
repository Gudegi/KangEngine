"""Copy bounded scene and renderer observations on the App owner thread."""

from __future__ import annotations

import numpy as np

from ... import scene
from ..capture import AutomationError


_CAPABILITY_FIELDS = (
    "cpu_scopes",
    "gpu_timestamps",
    "pass_timestamps",
    "command_timestamps",
    "external_timestamps",
    "in_pass_timestamps",
    "debug_groups",
    "timestamp_capacity",
)
_COUNTER_FIELDS = (
    "draw_calls",
    "indexed_draw_calls",
    "instances",
    "triangles",
    "buffer_upload_bytes",
    "texture_upload_bytes",
    "external_buffer_bytes",
    "buffer_allocations",
    "buffer_allocated_bytes",
)


def _bounded_integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise AutomationError(
            "INVALID_ARGUMENT", f"{name} must be an integer in {minimum}..{maximum}"
        )


def _fields(value, names: tuple[str, ...]) -> dict:
    return {name: getattr(value, name) for name in names}


def engine_status(app) -> dict:
    """Describe configured axes and backend capabilities without guessing asset units."""
    return {
        "graphics_backend": app.graphics_backend_type.name,
        "scene_backend": app.scene_backend_type.name,
        "up_axis": app.up_axis.name,
        "units": {
            "length": "scene_unit",
            "meters_per_unit": None,
            "time": "second",
            "angle": "radian",
        },
        "headless": app.headless,
        "render_enabled": app.run_config.render_enabled if app.run_config else True,
        "capabilities": {
            "scene_inspect": app.scene.get_root_prim() is not None,
            "scene_authoring": app.scene.get_root_prim() is not None,
            "renderer_profile": True,
            "profile_start": app.run_config.render_enabled if app.run_config else True,
            "camera_get": True,
            "camera_set": not app.has_active_scene_camera(),
            "scene_set_visibility": app.scene.get_root_prim() is not None,
            "recording_start": app.run_config.render_enabled
            if app.run_config
            else True,
            "recording_status": True,
            "recording_stop": True,
            "scenario_targets": True,
            "scenario_start": bool(getattr(app, "_mcp_scenario_targets", {})),
        },
    }


def _prim_snapshot(app, prim, depth: int) -> dict:
    parent = prim.get_parent()
    render_component = prim.get_render_component()
    mesh_component = prim.get_mesh_component()
    resource_component = prim.get_resource_component()
    material_handle = app._existing_resource_handle(prim.get_material())
    mesh_handle = mesh_component.resource_handle if mesh_component else None
    if mesh_handle == scene.InvalidResourceHandle:
        mesh_handle = None
    has_transform = prim.has_transform_component()
    transform_source = "SceneGraph" if has_transform else None
    if render_component is not None:
        transform_source = {
            "SCENE_GRAPH": "SceneGraph",
            "EXTERNAL_BUFFER": "ExternalBuffer",
        }[render_component.transform_source.name]
    effective_active = prim.is_active_in_hierarchy()
    effective_visible = prim.is_visible_in_hierarchy()
    return {
        "path": prim.get_path(),
        "parent": parent.get_path() if parent else None,
        "type": prim.get_type().name,
        "depth": depth,
        "active": prim.is_active(),
        "visible": prim.is_visible(),
        "effective_active": effective_active,
        "effective_visible": effective_visible,
        "effective_render_state": effective_active and effective_visible,
        # The Mat4 buffer exposes rows even though GLM's indexing exposes columns.
        "local_matrix": np.asarray(prim.compute_local_matrix()).tolist()
        if has_transform
        else None,
        "world_matrix": np.asarray(prim.compute_world_matrix()).tolist()
        if has_transform
        else None,
        "transform_source": transform_source,
        "matrix_scope": (
            ("batch_owner_prim" if transform_source == "ExternalBuffer" else "prim")
            if has_transform
            else None
        ),
        "resources": {"mesh": mesh_handle, "material": material_handle},
        "resource": (
            {
                "handle": resource_component.handle,
                "type": resource_component.type.name,
                "uri": resource_component.uri,
                "display_name": resource_component.display_name,
            }
            if resource_component
            else None
        ),
    }


def _resolve_prim(app, path):
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or (
            path != "/" and any(part in ("", ".", "..") for part in path[1:].split("/"))
        )
    ):
        raise AutomationError(
            "INVALID_ARGUMENT", "path must be an absolute scene prim path"
        )
    scene_root = app.scene.get_root_prim()
    if scene_root is None:
        raise AutomationError(
            "UNSUPPORTED", "this scene backend does not expose Prim traversal"
        )
    root = scene_root if path == "/" else app.scene.get_prim_at_path(path)
    if root is None:
        raise AutomationError("NOT_FOUND", f"scene prim does not exist: {path}")
    return root


def scene_inspect(
    app, *, path: str = "/", max_depth: int = 3, max_prims: int = 200
) -> dict:
    """Copy a subtree; ExternalBuffer matrices describe its owner, not its instances."""
    _bounded_integer(max_depth, "max_depth", 0, 16)
    _bounded_integer(max_prims, "max_prims", 1, 1000)
    root = _resolve_prim(app, path)

    prims = []
    truncated = False

    def visit(prim, depth: int) -> None:
        nonlocal truncated
        prims.append(_prim_snapshot(app, prim, depth))
        children = prim.get_children()
        if depth == max_depth:
            truncated |= bool(children)
            return
        for child in children:
            if len(prims) == max_prims:
                truncated = True
                return
            visit(child, depth + 1)

    visit(root, 0)
    return {
        "path": path,
        "max_depth": max_depth,
        "max_prims": max_prims,
        "matrix_layout": "row_major",
        "prims": prims,
        "truncated": truncated,
    }


def _frame_snapshot(frame) -> dict:
    result = _fields(
        frame,
        (
            "capture_id",
            "frame_index",
            "revision",
            "finalized",
            "gpu_latency_frames",
            "dropped_samples",
        ),
    )
    result.update(
        backend=frame.backend.name,
        counters=_fields(frame.counters, _COUNTER_FIELDS),
        metadata=dict(frame.metadata),
        samples=[
            {
                **_fields(
                    sample,
                    (
                        "sample_id",
                        "parent_sample_id",
                        "path",
                        "duration_ms",
                        "available",
                    ),
                ),
                "domain": sample.domain.name,
                "status": sample.status.name,
            }
            for sample in frame.samples
        ],
    )
    return result


def renderer_profile(app, *, max_frames: int = 60) -> dict:
    """Read retained revisions without enabling capture or waiting for GPU results."""
    _bounded_integer(max_frames, "max_frames", 1, 240)
    renderer = app.get_renderer()
    frames = renderer.frame_profile_history()[-max_frames:]
    summary = renderer.profile_summary(max_frames=max_frames)
    summary_data = _fields(
        summary,
        (
            "capture_id",
            "frame_count",
            "first_frame_index",
            "last_frame_index",
        ),
    )
    summary_data["scopes"] = [
        {
            **_fields(
                scope,
                (
                    "path",
                    "sample_count",
                    "ready_frames",
                    "pending_frames",
                    "unavailable_frames",
                    "mean_ms",
                    "median_ms",
                    "p95_ms",
                    "max_ms",
                ),
            ),
            "domain": scope.domain.name,
        }
        for scope in summary.scopes
    ]
    return {
        "enabled": renderer.profiler_enabled,
        "data_status": "retained" if frames else "empty",
        "max_frames": max_frames,
        "capabilities": _fields(renderer.profiler_capabilities, _CAPABILITY_FIELDS),
        "frames": [_frame_snapshot(frame) for frame in frames],
        "summary": summary_data,
        "summary_scope": "latest_capture",
    }
