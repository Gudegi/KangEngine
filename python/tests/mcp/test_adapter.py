"""Exercise the SDK boundary over a real loopback HTTP connection."""

import asyncio
import base64
from concurrent.futures import Future
import json
import subprocess
import sys
import threading

import pytest

pytest.importorskip("mcp")

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from kangengine.adapters.mcp import McpServer
from kangengine.app.capture import AutomationError, CaptureResult


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aDXsAAAAASUVORK5CYII="
)


class FakeAutomation:
    def __init__(self):
        self.calls = []
        self.pending = None
        self.error = None

    def submit(self, command, **kwargs):
        self.calls.append((command, kwargs, threading.get_ident()))
        if self.pending is not None:
            return self.pending
        future = Future()
        if self.error is not None:
            future.set_exception(self.error)
        elif command == "viewport_capture":
            future.set_result(
                CaptureResult(
                    png=PNG,
                    width=1,
                    height=1,
                    frame_id=42,
                    app_session_id="test-app",
                    output_path=kwargs["output_path"],
                )
            )
        else:
            future.set_result({"app_session_id": "test-app", "command": command})
        return future


@pytest.fixture
def server():
    automation = FakeAutomation()
    with McpServer(automation, port=0) as server:
        yield server, automation


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_tools_and_image_use_official_sdk(server, mode):
    server, automation = server
    main_thread = threading.get_ident()

    async def run():
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {server.token}"}
        ) as http:
            async with Client(
                streamable_http_client(server.url, http_client=http),
                mode=mode,
                cache=None,
            ) as client:
                listing = await client.list_tools()
                tools = {tool.name: tool for tool in listing.tools}
                assert set(tools) == {
                    "engine_status",
                    "scene_inspect",
                    "viewport_capture",
                    "renderer_profile",
                }
                assert tools["scene_inspect"].annotations.read_only_hint
                assert not tools["viewport_capture"].annotations.read_only_hint
                status = await client.call_tool("engine_status", {})
                assert status.structured_content["app_session_id"] == "test-app"
                scene = await client.call_tool(
                    "scene_inspect", {"path": "/robot", "max_depth": 1, "max_prims": 9}
                )
                assert not scene.is_error
                profile = await client.call_tool("renderer_profile", {"max_frames": 7})
                assert not profile.is_error
                capture = await client.call_tool("viewport_capture", {})
                image = next(
                    block for block in capture.content if block.type == "image"
                )
                assert image.mime_type == "image/png"
                assert base64.b64decode(image.data) == PNG
                assert capture.structured_content["frame_id"] == 42
                assert capture.structured_content["include_ui"] is False
                assert capture.structured_content["orientation"] == "top_left"

    asyncio.run(run())
    assert all(thread != main_thread for _, _, thread in automation.calls)
    assert automation.calls[1][1] == {"path": "/robot", "max_depth": 1, "max_prims": 9}
    assert automation.calls[2][1] == {"max_frames": 7}


def test_token_and_dns_rebinding_checks(server):
    server, automation = server
    body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    with httpx2.Client() as http:
        assert http.post(server.url, json=body).status_code == 401
        assert (
            http.post(
                server.url, json=body, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        headers = {
            "Authorization": f"Bearer {server.token}",
            "Accept": "application/json, text/event-stream",
            "Host": "attacker.example",
        }
        assert http.post(server.url, json=body, headers=headers).status_code == 421
        headers.pop("Host")
        headers["Origin"] = "https://attacker.example"
        assert http.post(server.url, json=body, headers=headers).status_code == 403
    assert automation.calls == []


def test_errors_timeout_and_cancelled_future(server, monkeypatch):
    from kangengine.adapters.mcp import tools

    server, automation = server
    monkeypatch.setattr(tools, "REQUEST_TIMEOUT_SECONDS", 0.03)

    async def run():
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {server.token}"}
        ) as http:
            async with Client(
                streamable_http_client(server.url, http_client=http), cache=None
            ) as client:
                automation.error = AutomationError("NOT_READY", "App is initializing")
                result = await client.call_tool("engine_status", {})
                assert result.is_error
                assert result.structured_content["error"]["code"] == "NOT_READY"
                automation.error = None
                automation.pending = Future()
                result = await client.call_tool("scene_inspect", {})
                assert result.structured_content["error"]["code"] == "TIMEOUT"
                assert automation.pending.cancelled()
                result = await client.call_tool("scene_inspect", {})
                assert result.structured_content["error"]["code"] == "CANCELLED"

    asyncio.run(run())


def test_output_bound(server, monkeypatch):
    from kangengine.adapters.mcp import tools

    server, _ = server
    monkeypatch.setattr(tools, "MAX_JSON_BYTES", 1)

    async def run():
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {server.token}"}
        ) as http:
            async with Client(
                streamable_http_client(server.url, http_client=http), cache=None
            ) as client:
                result = await client.call_tool("engine_status", {})
                assert result.is_error
                assert result.structured_content["error"]["code"] == "OUTPUT_TOO_LARGE"

    asyncio.run(run())


def test_cli_checks_session_without_printing_token(server, tmp_path):
    server, _ = server
    path = tmp_path / "connection.json"
    data = {
        "url": server.url,
        "token": server.token,
        "app_session_id": "test-app",
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "kangengine.adapters.mcp",
        "--connection-file",
        str(path),
        "status",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["app_session_id"] == "test-app"
    assert server.token not in result.stdout + result.stderr
    data["app_session_id"] = "old-app"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "App session changed" in result.stderr
    assert server.token not in result.stdout + result.stderr


def test_import_without_optional_sdk():
    script = """
import importlib.abc
import sys

class BlockSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mcp":
            raise ModuleNotFoundError("blocked optional SDK", name="mcp")

sys.meta_path.insert(0, BlockSdk())
from kangengine.adapters.mcp import McpServer, McpUnavailableError
assert "mcp" not in sys.modules
try:
    McpServer(None)
except McpUnavailableError as error:
    assert "kangengine[mcp]" in str(error)
else:
    raise AssertionError("missing SDK was not reported")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


def test_close_and_port_reuse(server):
    server, automation = server
    port = server.port
    server.close()
    server.close()
    assert not server._thread.is_alive()
    with McpServer(automation, port=port):
        pass


def test_disconnect_cancels_unstarted_work(server):
    server, automation = server
    automation.pending = Future()

    async def run():
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {server.token}"}
        ) as http:
            async with Client(
                streamable_http_client(server.url, http_client=http), cache=None
            ) as client:
                call = asyncio.create_task(client.call_tool("scene_inspect", {}))
                async with asyncio.timeout(3):
                    while not automation.calls:
                        await asyncio.sleep(0.01)
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
                async with asyncio.timeout(3):
                    while not automation.pending.cancelled():
                        await asyncio.sleep(0.01)

    asyncio.run(run())


def test_server_close_cancels_pending_work(server):
    server, automation = server
    automation.pending = Future()

    async def run():
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {server.token}"}
        ) as http:
            async with Client(
                streamable_http_client(server.url, http_client=http), cache=None
            ) as client:
                call = asyncio.create_task(client.call_tool("scene_inspect", {}))
                async with asyncio.timeout(3):
                    while not automation.calls:
                        await asyncio.sleep(0.01)
                await asyncio.to_thread(server.close)
                result = (await asyncio.gather(call, return_exceptions=True))[0]
                assert isinstance(result, BaseException)
                assert automation.pending.cancelled()
                assert not server._thread.is_alive()

    asyncio.run(run())
