"""Translate MCP calls into owner-thread automation requests."""

import asyncio
import base64
import json
from typing import Literal

from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations

from ...app.capture import AutomationError
from ._tool_groups import DEFAULT_GROUPS, enabled_tools

MAX_JSON_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 11.0


def _error_result(code: str, message: str) -> CallToolResult:
    data = {"error": {"code": code, "message": message}}
    return CallToolResult(
        content=[TextContent(text=json.dumps(data))],
        structured_content=data,
        is_error=True,
    )


def _json_result(data) -> CallToolResult:
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        return _error_result(
            "OUTPUT_TOO_LARGE",
            f"JSON output exceeds {MAX_JSON_BYTES} bytes; request fewer results.",
        )
    return CallToolResult(content=[TextContent(text=text)], structured_content=data)


def _capture_result(result) -> CallToolResult:
    if isinstance(result, CallToolResult):
        return result
    metadata = {
        "app_session_id": result.app_session_id,
        "frame_id": result.frame_id,
        "width": result.width,
        "height": result.height,
        "orientation": "top_left",
        "include_ui": False,
        "mime_type": "image/png",
        "output_path": result.output_path,
    }
    response = _json_result(metadata)
    response.content.append(
        ImageContent(
            data=base64.b64encode(result.png).decode("ascii"), mime_type="image/png"
        )
    )
    return response


def register_tools(server, automation, *, groups=DEFAULT_GROUPS):
    available = enabled_tools(groups)
    if "docs" in groups:
        from . import documentation

        # Freeze exposure at startup; status and discovery share this same set.
        if not documentation.GUIDE_ROOT.is_dir():
            available -= {"search_docs", "get_doc"}

    def tool(**kwargs):
        def register(function):
            if function.__name__ in available:
                return server.tool(**kwargs)(function)
            return function

        return register

    async def invoke(command, **kwargs):
        future = None
        try:
            future = automation.submit(command, **kwargs)
            return await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=REQUEST_TIMEOUT_SECONDS
            )
        except AutomationError as error:
            return _error_result(error.code, str(error))
        except TimeoutError:
            return _error_result("TIMEOUT", "The App did not complete the request.")
        except asyncio.CancelledError:
            if future is not None:
                future.cancel()
            # A disconnected request has no response channel left to write to.
            if asyncio.current_task().cancelling():
                raise
            return _error_result("CANCELLED", "The App cancelled the request.")

    read_only = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, open_world_hint=False
    )
    control = ToolAnnotations(
        read_only_hint=False, destructive_hint=False, open_world_hint=False
    )

    async def json_call(command, **arguments):
        result = await invoke(command, **arguments)
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @tool(annotations=control)
    async def scene_create_primitive(
        path: str,
        shape: Literal["box", "sphere", "plane"] = "box",
        translation: list[float] = [0, 0, 0],
        rotation_wxyz: list[float] = [1, 0, 0, 0],
        scale: list[float] = [1, 1, 1],
        color: list[float] = [0.7, 0.7, 0.7],
    ) -> CallToolResult:
        """Create a static visual at a new path beneath an existing root/Xform.

        Box side and sphere diameter are one scene unit before scale. Plane side
        is one unit, normal along App up axis. TRS is parent-relative; quaternion
        is wxyz. Scale is positive. RGB is linear 0..1. No physics is created.
        """
        return await json_call(
            "scene_create_primitive",
            path=path,
            shape=shape,
            translation=translation,
            rotation_wxyz=rotation_wxyz,
            scale=scale,
            color=color,
        )

    @tool(annotations=control)
    async def scene_set_transform(
        path: str,
        translation: list[float] | None = None,
        rotation_wxyz: list[float] | None = None,
        scale: list[float] | None = None,
    ) -> CallToolResult:
        """Edit a static leaf's LOCAL transform; omitted or null fields stay unchanged.

        Supply at least one field. Translation-only edits preserve the linear
        matrix; sheared/reflected transforms require both rotation and scale
        when changing either. Rotation/scale values are absolute, not increments.
        Does not move physics/ExternalBuffer objects or groups with children.
        Values use scene units, a normalized wxyz quaternion, and positive scale.
        """
        return await json_call(
            "scene_set_transform",
            path=path,
            translation=translation,
            rotation_wxyz=rotation_wxyz,
            scale=scale,
        )

    @tool(annotations=control)
    async def scene_set_material(
        path: str,
        color: list[float],
        shininess: float = 32.0,
    ) -> CallToolResult:
        """Assign an untextured Phong material to a static renderable.

        RGB is linear 0..1, shininess is 1..256. Replaces the material and resets
        its instance color multiplier to white; other users of the old material
        are unchanged. Existing render texture overrides and alpha mode remain.
        Materials are reused, with at most 128 per App session.
        """
        return await json_call(
            "scene_set_material", path=path, color=color, shininess=shininess
        )

    @tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            open_world_hint=False,
        )
    )
    async def scene_delete(path: str) -> CallToolResult:
        """Delete a childless object created by this App's automation service.

        Existing application objects and physics-owned objects cannot be deleted.
        """
        return await json_call("scene_delete", path=path)

    @tool(annotations=read_only)
    async def object_inspect(path: str) -> CallToolResult:
        """Read transforms, resource handles, render flags and physics metadata of one prim."""
        return await json_call("object_inspect", path=path)

    @tool(annotations=read_only)
    async def resource_inspect(handle: int, max_paths: int = 100) -> CallToolResult:
        """Read resource identity, URI and bounded usage paths without copying buffers."""
        return await json_call("resource_inspect", handle=handle, max_paths=max_paths)

    @tool(annotations=control)
    async def camera_focus(path: str, distance: float | None = None) -> CallToolResult:
        """Frame an automation-created primitive's world bounds, retaining view direction.

        Omit distance for automatic fit using FOV, aspect and clipping planes.
        Explicit distance (scene units) instead targets a static prim's world
        origin and also works on existing application prims. No group/animated
        geometry fit or visibility change. Inspect the resulting viewport PNG.
        """
        return await json_call("camera_focus", path=path, distance=distance)

    async def docs_call(method, **arguments):
        from . import documentation

        try:
            data = await asyncio.to_thread(getattr(documentation, method), **arguments)
            return _json_result(data)
        except AutomationError as error:
            return _error_result(error.code, str(error))

    @tool(annotations=read_only)
    async def search_docs(query: str, max_results: int = 10) -> CallToolResult:
        """Find literal text in public guide Markdown/RST in a source checkout.

        Returns paths, line numbers and short excerpts. No private docs or
        arbitrary filesystem paths are searched. Unavailable in a docs-free wheel.
        """
        return await docs_call("search_docs", query=query, max_results=max_results)

    @tool(annotations=read_only)
    async def get_doc(
        path: str, start_line: int = 1, max_lines: int = 80
    ) -> CallToolResult:
        """Read a public guide page returned by search_docs, at most 120 lines."""
        return await docs_call(
            "get_doc", path=path, start_line=start_line, max_lines=max_lines
        )

    @tool(annotations=read_only)
    async def get_api_docs(symbol: str) -> CallToolResult:
        """Read a public KangEngine symbol's signature/docstring, e.g. App.scene or geometry.create_box_data.

        Does not execute code, instantiate objects, or import arbitrary modules.
        """
        return await docs_call("get_api_docs", symbol=symbol)

    @tool(annotations=control)
    async def profile_start(frame_count: int = 60) -> CallToolResult:
        """Profile the next 1..240 normal frames and return a job ID.

        Restores the previous profiler setting on completion, cancellation or close.
        Does not advance simulation separately or wait for GPU timestamps.
        """
        return await json_call("profile_start", frame_count=frame_count)

    @tool(annotations=read_only)
    async def profile_status(job_id: str) -> CallToolResult:
        """Read progress of a retained profiling job."""
        return await json_call("profile_status", job_id=job_id)

    @tool(annotations=read_only)
    async def profile_result(
        job_id: str, offset: int = 0, max_frames: int = 30
    ) -> CallToolResult:
        """Read copied job frames in pages of 1..60; GPU samples may be pending.

        Completed, cancelled and failed jobs retain results until evicted by the
        four-job limit. Results are frozen at job end, including sample status.
        """
        return await json_call(
            "profile_result", job_id=job_id, offset=offset, max_frames=max_frames
        )

    @tool(annotations=control)
    async def profile_cancel(job_id: str) -> CallToolResult:
        """Stop profiling at the next frame boundary and restore its previous setting."""
        return await json_call("profile_cancel", job_id=job_id)

    @tool(annotations=read_only)
    async def camera_get() -> CallToolResult:
        """Read the navigation camera's world position, target and scene camera selection."""
        return await json_call("camera_get")

    @tool(annotations=control)
    async def camera_set(position: list[float], target: list[float]) -> CallToolResult:
        """Set navigation camera position and target, each three world scene-unit values.

        Requires no active scene camera. App callbacks may subsequently move it.
        """
        return await json_call("camera_set", position=position, target=target)

    @tool(annotations=control)
    async def scene_set_visibility(path: str, visible: bool) -> CallToolResult:
        """Set one prim's local visibility without changing active or child flags.

        Parent visibility still applies. ExternalBuffer visibility is batch-level.
        """
        return await json_call("scene_set_visibility", path=path, visible=visible)

    @tool(annotations=read_only)
    async def recording_status() -> CallToolResult:
        """Read the common App recorder state and current or last output path."""
        return await json_call("recording_status")

    @tool(annotations=control)
    async def recording_start(output_path: str, fps: float = 30.0) -> CallToolResult:
        """Start common App recording to a new MP4 under tmp/recordings.

        Requires a previously rendered frame. fps is in 1..240. Recording follows
        the App run mode and continues until recording_stop or App close.
        """
        return await json_call("recording_start", output_path=output_path, fps=fps)

    @tool(annotations=control)
    async def recording_stop() -> CallToolResult:
        """Stop and finalize the common App recording, including one started through UI."""
        return await json_call("recording_stop")

    @tool(annotations=read_only)
    async def engine_status() -> CallToolResult:
        """Read the App session, readiness, coordinates and supported operations."""
        result = await invoke("engine_status")
        if isinstance(result, CallToolResult):
            return result
        return _json_result(
            {
                **result,
                "tool_groups": sorted(groups),
                "available_tools": sorted(available),
            }
        )

    @tool(annotations=read_only)
    async def scene_inspect(
        path: str = "/", max_depth: int = 3, max_prims: int = 200
    ) -> CallToolResult:
        """Inspect a bounded prim hierarchy, visibility, transforms and resources.

        Args:
            path: Absolute KangEngine scene path.
            max_depth: Maximum child depth below the requested prim.
            max_prims: Maximum number of prim records, including the root.
        """
        result = await invoke(
            "scene_inspect", path=path, max_depth=max_depth, max_prims=max_prims
        )
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        )
    )
    async def viewport_capture(output_path: str | None = None) -> CallToolResult:
        """Return the next rendered scene as PNG, excluding UI and extra simulation steps.

        To capture an edit, wait for each dependent scene/camera tool to succeed
        before calling this tool. Do not run those edits and capture in parallel.
        App callbacks and animation may still change the scene before rendering.

        Args:
            output_path: Optional filename beneath the App's capture output directory.
                Existing files are never overwritten.
        """
        result = await invoke("viewport_capture", output_path=output_path)
        return _capture_result(result)

    @tool(annotations=read_only)
    async def renderer_profile(max_frames: int = 60) -> CallToolResult:
        """Read retained profiler results without enabling profiling or waiting for GPU.

        Args:
            max_frames: Number of recent frames to summarize, between 1 and 240.
        """
        result = await invoke("renderer_profile", max_frames=max_frames)
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @tool(annotations=read_only)
    async def scenario_targets() -> CallToolResult:
        """List opt-in playback targets, supported inputs and data/configuration IDs."""
        result = await invoke("scenario_targets")
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, open_world_hint=False
        )
    )
    async def scenario_start(
        target: str, scenario: str, duration: float, output_path: str | None = None
    ) -> CallToolResult:
        """Reset a registered target, advance logical time, and capture its final pose.

        Args:
            target: Name from scenario_targets; never inferred from the scene.
            scenario: Named input policy supported by the target.
            duration: Logical seconds after reset, in 0..120. Uses 1/60 s steps
                and one shorter remainder step. Returns a job ID immediately.
            output_path: Optional new PNG beneath the capture directory.
        """
        result = await invoke(
            "scenario_start",
            target=target,
            scenario=scenario,
            duration=duration,
            output_path=output_path,
        )
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @tool(annotations=read_only)
    async def scenario_status(job_id: str) -> CallToolResult:
        """Read progress, final state and capture metadata for a retained job."""
        result = await invoke("scenario_status", job_id=job_id)
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        )
    )
    async def scenario_cancel(job_id: str) -> CallToolResult:
        """Stop at a step boundary and restore live control; completed steps remain.

        A PNG file commit already in progress finishes before releasing control.
        """
        result = await invoke("scenario_cancel", job_id=job_id)
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @tool(annotations=read_only)
    async def scenario_capture(job_id: str) -> CallToolResult:
        """Return the retained completed-job PNG without rendering another frame."""
        return _capture_result(await invoke("scenario_capture", job_id=job_id))
