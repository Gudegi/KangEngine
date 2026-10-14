"""Call the MCP endpoint of an already running KangEngine App."""

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from ._dependency import load_mcp


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-file", type=Path, default=Path("tmp/mcp/session.json")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    scene = commands.add_parser("scene")
    scene.add_argument("path", nargs="?", default="/")
    scene.add_argument("--max-depth", type=int, default=3)
    scene.add_argument("--max-prims", type=int, default=200)
    capture = commands.add_parser("capture")
    capture.add_argument("--output", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--max-frames", type=int, default=60)
    return parser


async def _run(args):
    load_mcp()
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    connection = json.loads(args.connection_file.read_text(encoding="utf-8"))
    url = connection["url"]
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("the connection file must name a loopback HTTP endpoint")
    headers = {"Authorization": f"Bearer {connection['token']}"}
    async with httpx2.AsyncClient(headers=headers, timeout=15.0) as http:
        async with Client(
            streamable_http_client(
                url, http_client=http, max_sse_event_size=16 * 1024 * 1024
            ),
            read_timeout_seconds=15.0,
            cache=None,
        ) as client:
            status = await client.call_tool("engine_status", {})
            if status.is_error:
                _print_result(status)
                return 1
            actual_session = status.structured_content["app_session_id"]
            if actual_session != connection["app_session_id"]:
                raise RuntimeError(
                    "App session changed; use its current connection file"
                )
            if args.command == "status":
                result = status
            elif args.command == "scene":
                result = await client.call_tool(
                    "scene_inspect",
                    {
                        "path": args.path,
                        "max_depth": args.max_depth,
                        "max_prims": args.max_prims,
                    },
                )
            elif args.command == "capture":
                result = await client.call_tool(
                    "viewport_capture", {"output_path": args.output}
                )
            else:
                result = await client.call_tool(
                    "renderer_profile", {"max_frames": args.max_frames}
                )
            _print_result(result)
            return int(result.is_error)


def _print_result(result):
    if result.structured_content is not None:
        print(json.dumps(result.structured_content, ensure_ascii=False, indent=2))
    else:
        for block in result.content:
            if block.type == "text":
                print(block.text)


def main(argv=None):
    args = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
