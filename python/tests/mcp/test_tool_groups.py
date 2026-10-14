"""Disabled tools must be absent from discovery and unavailable for invocation."""

import asyncio
from concurrent.futures import Future

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from kangengine.adapters.mcp import McpServer
from kangengine.adapters.mcp import documentation
from kangengine.adapters.mcp._tool_groups import configured_groups, enabled_tools
from kangengine.adapters.mcp.tools import register_tools


@pytest.mark.parametrize("selection", [None, "scene,camera", "scenario,capture"])
def test_tool_groups_limit_registration_and_calls(monkeypatch, selection):
    if selection is None:
        monkeypatch.delenv("KANGENGINE_MCP_TOOLS", raising=False)
    else:
        monkeypatch.setenv("KANGENGINE_MCP_TOOLS", selection)
    groups = configured_groups()
    sdk = MCPServer("group-test")
    register_tools(sdk, None, groups=groups)

    async def check():
        listed = await sdk.list_tools()
        assert {tool.name for tool in listed} == enabled_tools(groups)
        hidden = "scenario_start" if "scenario" not in groups else "scene_delete"
        with pytest.raises(ToolError, match="Unknown tool"):
            await sdk.call_tool(hidden, {})

    asyncio.run(check())
    if selection is None:
        assert "scenario_start" not in enabled_tools(groups)


@pytest.mark.parametrize("selection", ["", "scene,typo", "all"])
def test_invalid_groups_fail_before_server_start(monkeypatch, selection):
    monkeypatch.setenv("KANGENGINE_MCP_TOOLS", selection)
    with pytest.raises(ValueError, match="KANGENGINE_MCP_TOOLS"):
        McpServer(None)


@pytest.mark.parametrize("token", ["", "short", "x" * 31, "x" * 32 + " ", "é" * 32])
def test_weak_or_invalid_supplied_tokens_rejected(token):
    with pytest.raises(ValueError, match="32 printable"):
        McpServer(None, token=token)


@pytest.mark.parametrize("with_guide", [True, False])
@pytest.mark.parametrize("selection", [None, "docs", "scene,camera"])
def test_guide_availability_matches_discovery_and_status(
    tmp_path, monkeypatch, with_guide, selection
):
    guide = tmp_path / "guide"
    if with_guide:
        guide.mkdir()
        (guide / "page.md").write_text("Public guide\n")
    monkeypatch.setattr(documentation, "GUIDE_ROOT", guide)
    if selection is None:
        monkeypatch.delenv("KANGENGINE_MCP_TOOLS", raising=False)
    else:
        monkeypatch.setenv("KANGENGINE_MCP_TOOLS", selection)
    groups = configured_groups()

    class Automation:
        def submit(self, command, **arguments):
            assert command == "engine_status"
            future = Future()
            future.set_result({"app_session_id": "guide-test"})
            return future

    sdk = MCPServer("guide-test")
    register_tools(sdk, Automation(), groups=groups)

    async def check():
        names = {tool.name for tool in await sdk.list_tools()}
        expected = enabled_tools(groups)
        if not with_guide:
            expected -= {"search_docs", "get_doc"}
        assert names == expected
        if selection is None:
            assert len(names) == (24 if with_guide else 22)
        status = await sdk.call_tool("engine_status", {})
        assert set(status.structured_content["available_tools"]) == names
        assert status.structured_content["tool_groups"] == sorted(groups)
        if "docs" in groups:
            api = await sdk.call_tool("get_api_docs", {"symbol": "App.initialize"})
            assert not api.is_error and "width" in api.structured_content["signature"]
        for tool, arguments in (
            ("search_docs", {"query": "Public"}),
            ("get_doc", {"path": "page.md"}),
        ):
            if tool in names:
                assert not (await sdk.call_tool(tool, arguments)).is_error
            else:
                with pytest.raises(ToolError, match="Unknown tool"):
                    await sdk.call_tool(tool, arguments)

        # Runtime filesystem changes cannot make status disagree with discovery.
        monkeypatch.setattr(documentation, "GUIDE_ROOT", tmp_path / "replacement")
        if not with_guide:
            documentation.GUIDE_ROOT.mkdir()
        elif "docs" in groups:
            result = await sdk.call_tool("get_doc", {"path": "page.md"})
            assert result.structured_content["error"]["code"] == "UNSUPPORTED"
        assert {tool.name for tool in await sdk.list_tools()} == names
        status = await sdk.call_tool("engine_status", {})
        assert set(status.structured_content["available_tools"]) == names

    asyncio.run(check())
