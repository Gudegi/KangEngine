"""Validate real OpenGL frames through the official MCP HTTP client.

Run from the repository root with the ``mcp`` extra and a working GLFW display.
No external assets are required; offscreen mode still needs a display server.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import httpx2
import imageio.v3 as iio
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
import numpy as np

import kangengine as ke


class ObservationApp(ke.App):
    def __init__(self, stop):
        super().__init__()
        self.stop = stop
        self.updates = 0
        self.cleanup_count = 0
        self.video_writer = None

    def setup(self):
        materials = self.create_standard_materials()
        mesh = ke.geometry.create_cube_data(scale=1.0)
        self.scene.add_mesh("/subject", mesh, material=materials.common)
        self.set_camera_view(position=[3, 2, 3], target=[0, 0, 0])
        self.get_renderer().set_profiler_enabled(enabled=True)

    def pre_update(self):
        # Deliberately omit super(): private automation hooks must still run.
        self.updates += 1
        if self.stop.is_set():
            self.request_close()

    def cleanup(self):
        self.cleanup_count += 1


async def _observe(url, token, session_id, output_dir):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx2.AsyncClient(headers=headers, timeout=15.0) as http:
        for connection in range(2):
            async with Client(
                streamable_http_client(url, http_client=http), cache=None
            ) as client:
                tools = await client.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "engine_status",
                    "scene_inspect",
                    "viewport_capture",
                    "renderer_profile",
                    "profile_start",
                    "profile_status",
                    "profile_result",
                    "profile_cancel",
                    "camera_get",
                    "camera_set",
                    "scene_set_visibility",
                    "recording_status",
                    "recording_start",
                    "recording_stop",
                    "scene_create_primitive",
                    "scene_set_transform",
                    "scene_set_material",
                    "scene_delete",
                    "object_inspect",
                    "resource_inspect",
                    "camera_focus",
                    "search_docs",
                    "get_doc",
                    "get_api_docs",
                }
                for _ in range(100):
                    status = await client.call_tool("engine_status", {})
                    if status.structured_content["state"] == "ready":
                        break
                    await asyncio.sleep(0.01)
                assert status.structured_content["state"] == "ready"
                assert status.structured_content["app_session_id"] == session_id
                if connection:
                    continue
                scene = await client.call_tool("scene_inspect", {"path": "/subject"})
                assert not scene.is_error, scene
                subject = scene.structured_content["prims"][0]
                assert subject["path"] == "/subject"
                assert subject["resources"]["mesh"] is not None
                capture = await client.call_tool(
                    "viewport_capture", {"output_path": "frame.png"}
                )
                assert not capture.is_error, capture
                image = next(
                    block for block in capture.content if block.type == "image"
                )
                png = base64.b64decode(image.data)
                pixels = iio.imread(png, extension=".png")
                metadata = capture.structured_content
                assert pixels.shape == (metadata["height"], metadata["width"], 3)
                assert np.ptp(pixels) > 0
                assert (output_dir / "frame.png").read_bytes() == png
                assert not metadata["include_ui"]
                profile = await client.call_tool("renderer_profile", {"max_frames": 4})
                assert not profile.is_error, profile
                assert profile.structured_content["data_status"] == "retained"
                assert len(profile.structured_content["frames"]) <= 4
                missing = await client.call_tool("scene_inspect", {"path": "/missing"})
                assert missing.is_error
                assert missing.structured_content["error"]["code"] == "NOT_FOUND"


def main():
    stop = threading.Event()
    errors = []
    app: ke.App = ObservationApp(stop)
    app.initialize(width=320, height=240, hide_ui=True, headless=True)
    app.set_vsync(enabled=False)
    app.set_render_hz(hz=60.0)
    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)
        app.automation.output_dir = output_dir
        server = app.enable_mcp(port=0)
        record_path = app._mcp_connection_file
        record = json.loads(record_path.read_text())
        assert record_path.stat().st_mode & 0o777 == 0o600
        assert record["app_session_id"] == app.automation.app_session_id

        def client_worker():
            try:
                asyncio.run(
                    _observe(
                        server.url, server.token, record["app_session_id"], output_dir
                    )
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "kangengine.adapters.mcp",
                        "capture",
                        "--output",
                        "cli.png",
                    ],
                    cwd=output_dir,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
                    },
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                assert completed.returncode == 0, completed.stderr + completed.stdout
                assert server.token not in completed.stdout + completed.stderr
                assert (output_dir / "cli.png").is_file()
            except BaseException as error:
                errors.append(error)
            finally:
                stop.set()

        worker = threading.Thread(target=client_worker, name="mcp-smoke-client")
        worker.start()
        watchdog = threading.Timer(35.0, stop.set)
        watchdog.start()
        try:
            app.start()
        finally:
            stop.set()
            watchdog.cancel()
            app.close()
            worker.join(timeout=20)
        assert not worker.is_alive()
        assert not record_path.exists()
        assert app.closed and app.cleanup_count == 1
        assert app.updates > 1
        if errors:
            raise errors[0]
    print("MCP OpenGL observation smoke passed")


if __name__ == "__main__":
    main()
