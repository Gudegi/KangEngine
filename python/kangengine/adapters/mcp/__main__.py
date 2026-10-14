"""Start a blank KangEngine scene or call an already running App's MCP endpoint."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from ._dependency import load_mcp
from ._connection import list_connections, process_state, select_connection_file


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-file",
        type=Path,
        help="Select a connection file; otherwise discover a single user session",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "sessions", help="List saved session paths and endpoints without tokens"
    )
    commands.add_parser(
        "token", help="Print the selected session's bearer token to stdout"
    )
    config = commands.add_parser(
        "print-config", help="Print connection settings without secrets"
    )
    config.add_argument("--format", choices=("json", "codex"), default="json")
    serve = commands.add_parser("serve", help="Open a blank scene with MCP enabled")
    serve.add_argument("--port", type=int, default=8765)
    commands.add_parser("status")
    scene = commands.add_parser("scene")
    scene.add_argument("path", nargs="?", default="/")
    scene.add_argument("--max-depth", type=int, default=3)
    scene.add_argument("--max-prims", type=int, default=200)
    capture = commands.add_parser("capture")
    capture.add_argument("--output", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--max-frames", type=int, default=60)
    start = commands.add_parser("profile-start")
    start.add_argument("--frame-count", type=int, default=60)
    for command in ("profile-status", "profile-cancel", "profile-result"):
        job = commands.add_parser(command)
        job.add_argument("job_id")
        if command == "profile-result":
            job.add_argument("--offset", type=int, default=0)
            job.add_argument("--max-frames", type=int, default=30)
    commands.add_parser("camera")
    camera = commands.add_parser("camera-set")
    camera.add_argument("--position", type=float, nargs=3, required=True)
    camera.add_argument("--target", type=float, nargs=3, required=True)
    visibility = commands.add_parser("visibility")
    visibility.add_argument("path")
    visibility.add_argument("visible", choices=("true", "false"))
    recording = commands.add_parser("recording-start")
    recording.add_argument("--output", required=True)
    recording.add_argument("--fps", type=float, default=30.0)
    commands.add_parser("recording-status")
    commands.add_parser("recording-stop")
    commands.add_parser("targets")
    run = commands.add_parser("scenario")
    run.add_argument("target")
    run.add_argument("--scenario", required=True)
    run.add_argument("--duration", type=float, required=True)
    run.add_argument("--output")
    for command in ("job-status", "job-cancel", "job-capture"):
        commands.add_parser(command).add_argument("job_id")
    return parser


def _read_connection(args):
    path = select_connection_file(args.connection_file)
    connection = json.loads(path.read_text(encoding="utf-8"))
    if process_state(connection) == "dead":
        raise ValueError("Selected App process has stopped; choose a current session")
    parsed = urlsplit(connection["url"])
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("the connection file must name a loopback HTTP endpoint")
    return connection


async def _run(args):
    load_mcp()
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    connection = _read_connection(args)
    url = connection["url"]
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
            elif args.command == "profile":
                result = await client.call_tool(
                    "renderer_profile", {"max_frames": args.max_frames}
                )
            elif args.command == "targets":
                result = await client.call_tool("scenario_targets", {})
            elif args.command.startswith("profile-"):
                arguments = {
                    k: v
                    for k, v in vars(args).items()
                    if k not in {"command", "connection_file"}
                }
                result = await client.call_tool(
                    args.command.replace("-", "_"), arguments
                )
            elif args.command == "camera":
                result = await client.call_tool("camera_get", {})
            elif args.command == "camera-set":
                result = await client.call_tool(
                    "camera_set", {"position": args.position, "target": args.target}
                )
            elif args.command == "visibility":
                result = await client.call_tool(
                    "scene_set_visibility",
                    {"path": args.path, "visible": args.visible == "true"},
                )
            elif args.command.startswith("recording-"):
                arguments = (
                    {"output_path": args.output, "fps": args.fps}
                    if args.command == "recording-start"
                    else {}
                )
                result = await client.call_tool(
                    args.command.replace("-", "_"), arguments
                )
            elif args.command == "scenario":
                result = await client.call_tool(
                    "scenario_start",
                    {
                        "target": args.target,
                        "scenario": args.scenario,
                        "duration": args.duration,
                        "output_path": args.output,
                    },
                )
            else:
                tool = {
                    "job-status": "scenario_status",
                    "job-cancel": "scenario_cancel",
                    "job-capture": "scenario_capture",
                }[args.command]
                result = await client.call_tool(tool, {"job_id": args.job_id})
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
    if args.command == "sessions":
        print(json.dumps(list_connections(), ensure_ascii=False, indent=2))
        return 0
    if args.command in {"token", "print-config"}:
        connection = _read_connection(args)
        if args.command == "token":
            print(connection["token"])
        elif args.format == "codex":
            print("[mcp_servers.kangengine]")
            print(f"url = {json.dumps(connection['url'])}")
            print('bearer_token_env_var = "KANGENGINE_MCP_TOKEN"')
        else:
            print(
                json.dumps(
                    {
                        "transport": "streamable-http",
                        "url": connection["url"],
                        "headers": {"Authorization": "Bearer <token>"},
                        "token_command": "python -m kangengine.adapters.mcp token",
                    },
                    indent=2,
                )
            )
        return 0
    if args.command == "serve":
        from ...app import App

        load_mcp()
        app = App()
        try:
            app.initialize(width=960, height=640)
            app.set_camera_view(position=[4, 3, 4], target=[0, 0, 0])
            app.enable_mcp(port=args.port, token=os.environ.get("KANGENGINE_MCP_TOKEN"))
            app.start()
        finally:
            app.close()
        return 0
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
