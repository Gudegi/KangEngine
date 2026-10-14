"""Bounded requests executed at the App owner's frame boundaries."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import threading
import time
import uuid
import weakref

from .capture import AutomationError, CaptureResult
from .mcp.authoring import COMMANDS as AUTHORING_COMMANDS

MAX_PENDING = 16
REQUEST_TIMEOUT = 10.0
MAX_PNG_BYTES = 8 * 1024 * 1024
MAX_CAPTURE_PIXELS = 4096 * 4096
REQUESTS_PER_FRAME = 4
MAX_ENCODING_JOBS = 2
CONTROL_COMMANDS = {
    "camera_get",
    "camera_set",
    "scene_set_visibility",
    "recording_status",
    "recording_start",
    "recording_stop",
}
PROFILE_COMMANDS = {
    "profile_start",
    "profile_status",
    "profile_result",
    "profile_cancel",
}


@dataclass(eq=False)
class _Request:
    command: str
    arguments: dict
    future: Future
    deadline: float
    timer: threading.Timer | None = None


class AutomationService:
    """Queue observations without moving engine objects to worker threads.

    Construct on the App thread. ``submit`` and cancellation are thread-safe;
    frame callbacks and ``close`` belong to the App thread. Futures must never
    be waited on from an App callback because the next frame fulfills them.
    """

    def __init__(self, app, *, output_dir: str | Path = "tmp/captures"):
        self._app = weakref.ref(app)
        self._owner = threading.get_ident()
        self.app_session_id = uuid.uuid4().hex
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.recording_dir = Path("tmp/recordings").resolve()
        self._state = "initializing"
        self._metadata = {}
        self._last_frame_id = None
        self._lock = threading.RLock()
        self._pending: set[_Request] = set()
        self._queue = deque()
        self._captures = []
        self._frame_active = False
        self._encoder = None
        self._mcp_scenario_runner = None
        self._profile_runner = None
        self._scene_authoring = None
        self._encoding_slots = threading.BoundedSemaphore(MAX_ENCODING_JOBS)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def submit(self, command: str, **kwargs) -> Future:
        """Return a future with a ten-second deadline, including queue time."""
        future = Future()
        with self._lock:
            if self._state in {"closing", "closed"}:
                future.set_exception(AutomationError("CLOSED", "App is closed"))
                return future
            if command == "engine_status" and self._state == "initializing":
                future.set_result(self._status())
                return future
            if (
                command
                not in {
                    "engine_status",
                    "scene_inspect",
                    "viewport_capture",
                    "renderer_profile",
                    "scenario_targets",
                    "scenario_start",
                    "scenario_status",
                    "scenario_cancel",
                    "scenario_capture",
                }
                | CONTROL_COMMANDS
                | PROFILE_COMMANDS
                | AUTHORING_COMMANDS
            ):
                future.set_exception(
                    AutomationError("INVALID_ARGUMENT", "Unknown operation")
                )
                return future
            if len(self._pending) >= MAX_PENDING:
                future.set_exception(
                    AutomationError("BUSY", "App request queue is full")
                )
                return future
            request = _Request(
                command, kwargs, future, time.monotonic() + REQUEST_TIMEOUT
            )
            request.timer = threading.Timer(
                REQUEST_TIMEOUT,
                self._fail,
                args=(request, "TIMEOUT", "No result before the request deadline"),
            )
            request.timer.daemon = True
            self._pending.add(request)
            self._queue.append(request)
            self._app()._request_automation_frame()
            future.add_done_callback(lambda _: self._forget(request))
            request.timer.start()
        return future

    def _status(self):
        return {
            **self._metadata,
            "app_session_id": self.app_session_id,
            "state": self._state,
            "last_frame_id": self._last_frame_id,
            "contract_version": 1,
            "capabilities": {
                **self._metadata.get("capabilities", {}),
                "engine_status": True,
                "viewport_capture": self._metadata.get("render_enabled"),
            },
            "limits": {
                "max_pending": MAX_PENDING,
                "request_timeout_seconds": REQUEST_TIMEOUT,
                "max_png_bytes": MAX_PNG_BYTES,
                "max_capture_pixels": MAX_CAPTURE_PIXELS,
                "max_encoding_jobs": MAX_ENCODING_JOBS,
            },
        }

    def _forget(self, request):
        with self._lock:
            self._pending.discard(request)
            request.timer.cancel()
            # Remove expired requests even when the App stops producing frames.
            try:
                self._queue.remove(request)
            except ValueError:
                pass

    def _fail(self, request, code, message):
        with self._lock:
            if not request.future.done() and not request.future.running():
                try:
                    request.future.set_exception(AutomationError(code, message))
                except InvalidStateError:
                    # Future.cancel() uses its own lock, independent of this service.
                    pass

    def _alive(self, request):
        if time.monotonic() >= request.deadline:
            self._fail(request, "TIMEOUT", "No result before the request deadline")
        return not request.future.done()

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("App frame operations must run on the creating thread")

    def _mcp_scenario_command(self, app, command, arguments):
        if self._mcp_scenario_runner is None:
            from .mcp.scenario import McpScenarioRunner

            self._mcp_scenario_runner = McpScenarioRunner(app._mcp_scenario_targets)
        runner = self._mcp_scenario_runner
        if command == "scenario_targets":
            return runner.list_targets()
        if command == "scenario_start":
            if app.run_config is not None and not app.run_config.render_enabled:
                raise AutomationError(
                    "UNSUPPORTED", "Scenario capture requires rendering"
                )
            arguments = {
                **arguments,
                "output_path": self._output_path(arguments.get("output_path")),
            }
            return runner.start(**arguments)
        if command == "scenario_status":
            return runner.status(**arguments)
        if command == "scenario_cancel":
            return runner.cancel(**arguments)
        return runner.capture(**arguments)

    def on_frame_started(self):
        self._check_owner()
        from .mcp import inspection

        app = self._app()
        with self._lock:
            if self._state in {"closing", "closed"}:
                return
            initializing = self._state == "initializing"
            self._state = "ready"
            self._frame_active = True
            requests = [
                self._queue.popleft()
                for _ in range(min(REQUESTS_PER_FRAME, len(self._queue)))
            ]
            if self._queue:
                app._request_automation_frame()
        # The preceding frame's profiler scope has finalized before this boundary.
        if self._profile_runner is not None:
            self._profile_runner.tick(app)
        if initializing or any(
            request.command == "engine_status" for request in requests
        ):
            metadata = inspection.engine_status(app)
            with self._lock:
                self._metadata = metadata
                self._last_frame_id = (
                    int(app.frame_index) - 1 if app.frame_index else None
                )
        for request in requests:
            if not self._alive(request):
                continue
            try:
                if (
                    request.command.startswith("scenario_")
                    or request.command
                    in CONTROL_COMMANDS | PROFILE_COMMANDS | AUTHORING_COMMANDS
                ):
                    # Mutation commits once owner-thread execution starts. A transport
                    # timeout cannot then report a cancelled start that already ran.
                    with self._lock:
                        if not self._alive(request):
                            continue
                        if not request.future.set_running_or_notify_cancel():
                            continue
                        request.timer.cancel()
                    try:
                        result = self._control_command(
                            app, request.command, request.arguments
                        )
                        if isinstance(result, dict):
                            result = {
                                **result,
                                "app_session_id": self.app_session_id,
                                "processed_before_frame_id": int(app.frame_index),
                            }
                        request.future.set_result(result)
                    except Exception as exc:
                        code = (
                            exc.code
                            if isinstance(exc, AutomationError)
                            else "INVALID_ARGUMENT"
                            if isinstance(exc, (TypeError, ValueError))
                            else "INTERNAL_ERROR"
                        )
                        request.future.set_exception(AutomationError(code, str(exc)))
                    continue
                if request.command == "engine_status":
                    with self._lock:
                        if self._alive(request):
                            request.future.set_result(self._status())
                    continue
                if request.command == "viewport_capture":
                    if app.run_config is not None and not app.run_config.render_enabled:
                        raise AutomationError(
                            "UNSUPPORTED", "Rendering is disabled in HEADLESS_FAST mode"
                        )
                    if request.arguments.keys() - {"output_path"}:
                        raise AutomationError(
                            "INVALID_ARGUMENT", "Unsupported capture argument"
                        )
                    self._captures.append(request)
                    continue
                result = getattr(inspection, request.command)(app, **request.arguments)
                result = {
                    **result,
                    "app_session_id": self.app_session_id,
                    "observed_before_frame_id": int(app.frame_index),
                }
                with self._lock:
                    if self._alive(request):
                        request.future.set_result(result)
            except AutomationError as exc:
                self._fail(request, exc.code, str(exc))
            except (TypeError, ValueError) as exc:
                self._fail(request, "INVALID_ARGUMENT", str(exc))
            except Exception as exc:
                self._fail(request, "INTERNAL_ERROR", str(exc))
        if self._mcp_scenario_runner is not None:
            self._mcp_scenario_runner.tick(
                lambda **kwargs: self.submit("viewport_capture", **kwargs)
            )
            if self._mcp_scenario_runner.active is not None:
                app._request_automation_frame()
        if self._profile_runner is not None and self._profile_runner.active is not None:
            app._request_automation_frame()

    def _control_command(self, app, command, arguments):
        if command in AUTHORING_COMMANDS:
            if self._scene_authoring is None:
                from .mcp.authoring import SceneAuthoring

                self._scene_authoring = SceneAuthoring()
            return getattr(self._scene_authoring, command)(app, **arguments)
        if command.startswith("scenario_"):
            return self._mcp_scenario_command(app, command, arguments)
        if command in PROFILE_COMMANDS:
            if self._profile_runner is None:
                from .mcp.profiling import ProfileRunner

                self._profile_runner = ProfileRunner()
            method = getattr(self._profile_runner, command.removeprefix("profile_"))
            if command in {"profile_start", "profile_cancel"}:
                return method(app, **arguments)
            return method(**arguments)
        from .mcp import control

        if command == "recording_start":
            return control.recording_start(
                app, output_dir=self.recording_dir, **arguments
            )
        return getattr(control, command)(app, **arguments)

    def _output_path(self, output_path):
        if output_path is None:
            return None
        path = Path(output_path).expanduser()
        path = (path if path.is_absolute() else self.output_dir / path).resolve()
        if not path.is_relative_to(self.output_dir) or path.suffix.lower() != ".png":
            raise AutomationError(
                "INVALID_ARGUMENT",
                "output_path must be a PNG inside the capture directory",
            )
        if path.exists():
            raise AutomationError("INVALID_ARGUMENT", "Capture output already exists")
        return path

    def on_frame_rendered(self):
        self._check_owner()
        if not self._frame_active:
            return
        self._frame_active = False
        app = self._app()
        with self._lock:
            self._last_frame_id = int(app.frame_index)
        captures, self._captures = self._captures, []
        captures = [request for request in captures if self._alive(request)]
        if not captures:
            return
        if not self._encoding_slots.acquire(blocking=False):
            for request in captures:
                self._fail(request, "BUSY", "PNG encoder queue is full")
            return
        try:
            if app.get_width() * app.get_height() > MAX_CAPTURE_PIXELS:
                raise AutomationError(
                    "OUTPUT_TOO_LARGE", f"Capture exceeds {MAX_CAPTURE_PIXELS} pixels"
                )
            pixels = app.read_rgb_pixels(flip_y=True)
            height, width, _ = pixels.shape
            if width * height > MAX_CAPTURE_PIXELS:
                raise AutomationError(
                    "OUTPUT_TOO_LARGE", f"Capture exceeds {MAX_CAPTURE_PIXELS} pixels"
                )
            if self._encoder is None:
                self._encoder = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="kangengine-capture"
                )
            job = self._encoder.submit(
                self._encode, captures, pixels, self._last_frame_id
            )
            job.add_done_callback(lambda _: self._encoding_slots.release())
        except Exception as exc:
            self._encoding_slots.release()
            code = exc.code if isinstance(exc, AutomationError) else "CAPTURE_FAILED"
            for request in captures:
                self._fail(request, code, str(exc))

    def _encode(self, requests, pixels, frame_id):
        captures = []
        for request in requests:
            if not self._alive(request):
                continue
            try:
                request.arguments["output_path"] = self._output_path(
                    request.arguments.get("output_path")
                )
                captures.append(request)
            except (AutomationError, TypeError, ValueError, OSError) as exc:
                code = (
                    exc.code if isinstance(exc, AutomationError) else "INVALID_ARGUMENT"
                )
                self._fail(request, code, str(exc))
        requests = captures
        try:
            from PIL import Image

            if not any(self._alive(request) for request in requests):
                return
            with BytesIO() as buffer:
                Image.fromarray(pixels).save(buffer, format="PNG")
                png = buffer.getvalue()
            if len(png) > MAX_PNG_BYTES:
                raise AutomationError(
                    "OUTPUT_TOO_LARGE", f"PNG exceeds {MAX_PNG_BYTES} bytes"
                )
        except Exception as exc:
            code = exc.code if isinstance(exc, AutomationError) else "CAPTURE_FAILED"
            for request in requests:
                self._fail(request, code, str(exc))
            return
        height, width, _ = pixels.shape
        for request in requests:
            with self._lock:
                if not self._alive(request):
                    continue
                # Commit owns the result from here. Shutdown waits for it;
                # deadline/cancellation may not report failure after a file is saved.
                if not request.future.set_running_or_notify_cancel():
                    continue
                request.timer.cancel()
            path = request.arguments["output_path"]
            try:
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(png)
                request.future.set_result(
                    CaptureResult(
                        png=png,
                        width=width,
                        height=height,
                        frame_id=frame_id,
                        app_session_id=self.app_session_id,
                        output_path=str(path) if path is not None else None,
                    )
                )
            except OSError as exc:
                request.future.set_exception(
                    AutomationError("CAPTURE_FAILED", str(exc))
                )

    def close(self):
        self._check_owner()
        with self._lock:
            if self._state == "closed":
                return
            self._state = "closing"
            for request in tuple(self._pending):
                self._fail(request, "CLOSED", "App closed before the request completed")
            self._captures.clear()
        if self._encoder is not None:
            self._encoder.shutdown(wait=True, cancel_futures=True)
        if self._mcp_scenario_runner is not None:
            self._mcp_scenario_runner.close()
        if self._profile_runner is not None:
            self._profile_runner.close(self._app())
        with self._lock:
            self._state = "closed"
