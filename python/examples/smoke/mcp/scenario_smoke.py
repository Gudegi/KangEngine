"""Compare fixed-time capture through MCP and CLI using an asset-free GL scene."""

import asyncio
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import weakref

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

import kangengine as ke


class AnimatedTarget:
    def __init__(self, app):
        self.app = weakref.ref(app)
        self.owned = False
        self.reset()

    def describe(self):
        return {
            "scenarios": ["orbit"],
            "data_sha256": hashlib.sha256(b"orbit-v1").hexdigest(),
        }

    def begin(self, scenario):
        self.owned = True

    def reset(self):
        self.elapsed = 0.0
        self.steps = 0

    def step(self, dt):
        self.elapsed += dt
        self.steps += 1

    def sync_visuals(self):
        self.app().subject.set_local_translation(ke.Vec3(*self.position()))

    def position(self):
        return [math.sin(self.elapsed), 0.5, math.cos(self.elapsed)]

    def snapshot(self):
        return {"time": self.elapsed, "position": self.position(), "steps": self.steps}

    def end(self):
        self.owned = False


class FixtureApp(ke.App):
    def __init__(self, stop):
        super().__init__()
        self.stop = stop
        self.target = AnimatedTarget(self)
        self.register_mcp_scenario_target("fixture", self.target)

    def setup(self):
        materials = self.create_standard_materials()
        mesh = ke.geometry.create_cube_data(scale=0.5)
        self.subject = self.scene.add_mesh("/subject", mesh, material=materials.common)
        self.set_camera_view(position=[4, 3, 4], target=[0, 0.5, 0])

    def pre_render(self):
        if self.stop.is_set():
            self.request_close()
        if not self.target.owned:
            self.target.step(1 / 37)
            self.target.sync_visuals()


async def observe(server, directory):
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {server.token}"}
    ) as http:
        async with Client(
            streamable_http_client(server.url, http_client=http), cache=None
        ) as client:

            async def call(tool, **arguments):
                result = await client.call_tool(tool, arguments)
                assert not result.is_error, result
                return result

            def cli(*arguments):
                result = subprocess.run(
                    [sys.executable, "-m", "kangengine.adapters.mcp", *arguments],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                return json.loads(result.stdout)

            listing = await call("scenario_targets")
            assert listing.structured_content["targets"][0]["name"] == "fixture"
            for duration in (13, 24, 0.025):
                snapshots = []
                images = []
                for transport in ("mcp", "cli"):
                    filename = f"{transport}-{duration}.png"
                    if transport == "mcp":
                        response = await call(
                            "scenario_start",
                            target="fixture",
                            scenario="orbit",
                            duration=duration,
                            output_path=filename,
                        )
                        job = response.structured_content
                    else:
                        job = cli(
                            "scenario",
                            "fixture",
                            "--scenario",
                            "orbit",
                            "--duration",
                            str(duration),
                            "--output",
                            filename,
                        )
                    job_id = job["job_id"]
                    for _ in range(2000):
                        result = await call("scenario_status", job_id=job_id)
                        job = result.structured_content
                        if job["state"] in ("completed", "failed", "cancelled"):
                            break
                        await asyncio.sleep(0.005)
                    assert job["state"] == "completed", job
                    assert abs(job["snapshot"]["time"] - duration) < 1e-9
                    assert job["snapshot"]["steps"] == job["steps_total"]
                    snapshots.append(job["snapshot"])
                    result = await call("scenario_capture", job_id=job_id)
                    image = next(
                        block for block in result.content if block.type == "image"
                    )
                    png = base64.b64decode(image.data)
                    assert png == (directory / filename).read_bytes()
                    images.append(png)
                assert snapshots[0] == snapshots[1]
                assert images[0] == images[1]
            result = await call(
                "scenario_start", target="fixture", scenario="orbit", duration=120
            )
            job_id = result.structured_content["job_id"]
            cancelled = cli("job-cancel", job_id)
            assert cancelled["state"] == "cancelled", cancelled
            assert cancelled["elapsed_seconds"] < 120


def main():
    os.environ["KANGENGINE_MCP_TOOLS"] = "scenario,capture"
    stop = threading.Event()
    errors = []
    app: ke.App = FixtureApp(stop)
    app.initialize(width=160, height=120, hide_ui=True, headless=True)
    app.set_vsync(enabled=False)
    app.set_render_hz(hz=120)
    with tempfile.TemporaryDirectory() as directory:
        app.automation.output_dir = Path(directory)
        server = app.enable_mcp(port=0)

        def worker():
            try:
                asyncio.run(observe(server, Path(directory)))
            except BaseException as error:
                errors.append(error)
            finally:
                stop.set()

        thread = threading.Thread(target=worker)
        watchdog = threading.Timer(90, stop.set)
        thread.start()
        watchdog.start()
        try:
            app.start()
        finally:
            stop.set()
            watchdog.cancel()
            app.close()
            thread.join(timeout=20)
        assert not thread.is_alive()
        if errors:
            raise errors[0]
        assert not app.target.owned
    print(
        "PASS: MCP/CLI 13 s, 24 s, fractional-step poses and PNGs match; cancellation releases control"
    )


if __name__ == "__main__":
    main()
