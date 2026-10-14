"""Translate MCP calls into owner-thread automation requests."""

import asyncio
import base64
import json

from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations

from ...app.capture import AutomationError

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


def register_tools(server, automation):
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

    @server.tool(annotations=read_only)
    async def engine_status() -> CallToolResult:
        """Read the App session, readiness, coordinates and supported operations."""
        result = await invoke("engine_status")
        return result if isinstance(result, CallToolResult) else _json_result(result)

    @server.tool(annotations=read_only)
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

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        )
    )
    async def viewport_capture(output_path: str | None = None) -> CallToolResult:
        """Return the next rendered scene as PNG, excluding UI and extra simulation steps.

        Args:
            output_path: Optional filename beneath the App's capture output directory.
                Existing files are never overwritten.
        """
        result = await invoke("viewport_capture", output_path=output_path)
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
            "output_path": (
                None if result.output_path is None else str(result.output_path)
            ),
        }
        response = _json_result(metadata)
        response.content.append(
            ImageContent(
                data=base64.b64encode(result.png).decode("ascii"),
                mime_type="image/png",
            )
        )
        return response

    @server.tool(annotations=read_only)
    async def renderer_profile(max_frames: int = 60) -> CallToolResult:
        """Read retained profiler results without enabling profiling or waiting for GPU.

        Args:
            max_frames: Number of recent frames to summarize, between 1 and 240.
        """
        result = await invoke("renderer_profile", max_frames=max_frames)
        return result if isinstance(result, CallToolResult) else _json_result(result)
