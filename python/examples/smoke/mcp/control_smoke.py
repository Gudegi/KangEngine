"""Exercise generic MCP controls against real GL frames and an MP4 encoder."""

import asyncio
import base64
import json
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


class ControlApp(ke.App):
    def __init__(self, stop):
        super().__init__()
        self.stop = stop

    def setup(self):
        materials = self.create_standard_materials()
        self.scene.add_mesh(
            "/subject",
            ke.geometry.create_cube_data(scale=1.0),
            material=materials.common,
        )
        self.set_camera_view(position=[3, 2, 3], target=[0, 0, 0])

    def pre_update(self):
        if self.stop.is_set():
            self.request_close()


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
                return result.structured_content

            def cli(*arguments):
                result = subprocess.run(
                    [sys.executable, "-m", "kangengine.adapters.mcp", *arguments],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                assert server.token not in result.stdout + result.stderr
                return json.loads(result.stdout)

            await call("viewport_capture")
            # Build through MCP only; no example-specific construction callbacks.
            created = await call(
                "scene_create_primitive",
                path="/created",
                shape="box",
                translation=[1.5, 0.5, 0],
                color=[1, 0, 0],
            )
            assert created["automation_created"]
            mesh_handle = created["resources"]["mesh"]
            material_handle = created["resources"]["material"]
            resource = await call("resource_inspect", handle=mesh_handle)
            assert "/created" in resource["usage_paths"]
            await call("scene_create_primitive", path="/second", shape="box")
            second = await call("object_inspect", path="/second")
            assert second["resources"]["mesh"] == mesh_handle
            await call("scene_set_visibility", path="/second", visible=False)
            await call("scene_set_visibility", path="/subject", visible=False)
            changed = await call(
                "scene_set_transform",
                path="/created",
                translation=[2, 1, 0],
                scale=[1, 2, 1],
                rotation_wxyz=[1, 0, 0, 0],
            )
            np.testing.assert_allclose(
                np.asarray(changed["local_matrix"])[:3, 3], [2, 1, 0]
            )
            focused = await call("camera_focus", path="/created", distance=4)
            np.testing.assert_allclose(focused["target"], [2, 1, 0])
            moved = await call(
                "scene_set_transform", path="/created", translation=[2, 1.5, 0]
            )
            np.testing.assert_allclose(np.diag(moved["local_matrix"]), [1, 2, 1, 1])
            focused = await call("camera_focus", path="/created")
            assert focused["focus_mode"] == "bounds"
            np.testing.assert_allclose(focused["target"], [2, 1.5, 0])

            async def pixels():
                result = await client.call_tool("viewport_capture", {})
                assert not result.is_error, result
                assert (
                    result.structured_content["frame_id"]
                    >= moved["processed_before_frame_id"]
                )
                image = next(block for block in result.content if block.type == "image")
                return iio.imread(base64.b64decode(image.data), extension=".png")

            red = await pixels()
            changed = await call("scene_set_material", path="/created", color=[0, 1, 0])
            assert changed["resources"]["material"] != material_handle
            green = await pixels()
            assert np.mean(np.abs(red.astype(float) - green.astype(float))) > 1
            # Record the authored scene, then verify the generated MP4 can be decoded.
            await call("recording_start", output_path="authored.mp4")
            await pixels()
            await pixels()
            await call("recording_stop")
            assert iio.imread(directory / "authored.mp4", index=0).ndim == 3
            assert (await call("scene_delete", path="/created"))["removed"]
            assert (await call("scene_delete", path="/second"))["removed"]
            assert not (await call("resource_inspect", handle=mesh_handle))[
                "usage_paths"
            ]
            rejected = await client.call_tool("scene_delete", {"path": "/subject"})
            assert (
                rejected.is_error
                and rejected.structured_content["error"]["code"] == "UNSUPPORTED"
            )
            # Recreate after deletion, exercise every primitive and resource reuse.
            recreated = await call(
                "scene_create_primitive", path="/created", shape="box"
            )
            assert recreated["resources"]["mesh"] == mesh_handle
            await call("scene_delete", path="/created")
            for shape in ("sphere", "plane"):
                await call("scene_create_primitive", path="/created", shape=shape)
                await pixels()
                await call("scene_delete", path="/created")
            await call("scene_set_visibility", path="/subject", visible=True)
            await call("camera_set", position=[3, 2, 3], target=[0, 0, 0])
            original = await call("camera_get")
            changed = await call("camera_set", position=[4, 3, 4], target=[0, 0, 0])
            assert changed["position"] == [4, 3, 4]
            assert cli("camera")["position"] == [4, 3, 4]
            cli(
                "camera-set",
                "--position",
                *map(str, original["position"]),
                "--target",
                *map(str, original["target"]),
            )
            hidden = await call("scene_set_visibility", path="/subject", visible=False)
            assert not hidden["visible"] and hidden["active"]
            assert cli("visibility", "/subject", "true")["effective_visible"]

            job = cli("profile-start", "--frame-count", "8")
            job_id = job["job_id"]
            for _ in range(200):
                status = await call("profile_status", job_id=job_id)
                if status["state"] != "running":
                    break
                await asyncio.sleep(0.01)
            assert status["state"] == "completed", status
            assert status["frames_collected"] == 8
            result = cli("profile-result", job_id, "--max-frames", "3")
            assert len(result["frames"]) == 3 and result["next_offset"] == 3
            frames = (await call("profile_result", job_id=job_id))["frames"]
            assert [f["frame_index"] for f in frames] == list(
                range(job["first_frame_index"], job["first_frame_index"] + 8)
            )
            assert not (await call("renderer_profile"))["enabled"]
            assert cli("profile-status", job_id)["state"] == "completed"
            job = await call("profile_start", frame_count=240)
            cancelled = await call("profile_cancel", job_id=job["job_id"])
            assert cancelled["state"] == "cancelled"
            assert not (await call("renderer_profile"))["enabled"]
            assert cli("profile-cancel", job["job_id"])["state"] == "cancelled"

            assert not (await call("recording_status"))["recording"]
            assert (await call("recording_start", output_path="sdk.mp4", fps=30))[
                "recording"
            ]
            for _ in range(4):
                await call("viewport_capture")
            assert not (await call("recording_stop"))["recording"]
            assert cli("recording-start", "--output", "cli.mp4")["recording"]
            assert cli("recording-status")["recording"]
            assert not cli("recording-stop")["recording"]
            for name in ("sdk.mp4", "cli.mp4"):
                pixels = iio.imread(directory / name, index=0)
                assert pixels.ndim == 3 and pixels.shape[-1] == 3
                assert np.ptp(pixels) > 0
            # App shutdown must finalize an unfinished recording as well.
            await call("recording_start", output_path="close.mp4")
            await call("viewport_capture")


def main():
    stop = threading.Event()
    errors = []
    app: ke.App = ControlApp(stop)
    app.initialize(width=160, height=120, hide_ui=True, headless=True)
    app.set_render_hz(hz=60)
    app.set_vsync(enabled=False)
    with tempfile.TemporaryDirectory() as directory:
        app.automation.recording_dir = Path(directory)
        server = app.enable_mcp(port=0)

        def worker():
            try:
                asyncio.run(observe(server, Path(directory)))
            except BaseException as exc:
                errors.append(exc)
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
        assert iio.imread(Path(directory) / "close.mp4", index=0).ndim == 3
    print(
        "PASS: MCP scene authoring/resources, camera, visibility, profiling and MP4 recording/close"
    )


if __name__ == "__main__":
    main()
