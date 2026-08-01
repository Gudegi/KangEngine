"""Application-level video recording orchestration."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import queue
import threading
import time

from .video_recorder import VideoRecorder


class VideoCaptureController:
    """Connect an App framebuffer and run mode to :class:`VideoRecorder`."""

    def __init__(
        self,
        output_dir: str | Path = "tmp/recordings",
        fps: float | None = None,
        *,
        clock=None,
        max_resolution: tuple[int, int] | None = (1920, 1080),
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.default_fps = None if fps is None else float(fps)
        self.max_resolution = self._validate_resolution(max_resolution)
        self._clock = time.perf_counter if clock is None else clock
        self._recorder: VideoRecorder | None = None
        self._output_path: Path | None = None
        self._run_mode = "paced"
        self._wall_start_time: float | None = None
        self._last_paced_frame = None
        self._scheduled_frame_count = 0
        self._encode_queue = None
        self._encode_thread = None
        self._encode_error: BaseException | None = None
        self._capture_resolution: tuple[int, int] | None = None

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def recorder(self) -> VideoRecorder | None:
        return self._recorder

    def configure(
        self,
        *,
        run_mode,
        output_dir: str | Path | None = None,
        fps: float | None = None,
    ):
        if self.is_recording:
            raise RuntimeError("cannot reconfigure video capture while recording")
        self._run_mode = self._mode_value(run_mode)
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser()
        if fps is not None:
            self.default_fps = float(fps)
        return self

    def set_max_resolution(self, width: int | None, height: int | None = None):
        if self.is_recording:
            raise RuntimeError("cannot change video resolution while recording")
        resolution = None if width is None else (width, height)
        self.max_resolution = self._validate_resolution(resolution)
        return self

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_active()

    @property
    def is_offscreen(self) -> bool:
        return self._run_mode == "offscreen_fast"

    def start(
        self, app, output_path: str | Path | None = None, fps: float | None = None
    ) -> Path:
        if self._run_mode == "headless_fast":
            raise RuntimeError(
                "HEADLESS_FAST disables rendering and cannot record video"
            )
        if self.is_recording:
            raise RuntimeError("video recording is already active")

        path = (
            self._default_output_path()
            if output_path is None
            else Path(output_path).expanduser()
        )
        recording_fps = self._resolve_fps(app, fps)
        self._recorder = VideoRecorder(path, fps=recording_fps)
        self._recorder.start()
        self._output_path = path
        self._wall_start_time = self._clock() if not self.is_offscreen else None
        self._last_paced_frame = None
        self._scheduled_frame_count = 0
        self._encode_error = None
        self._capture_resolution = self._resolve_capture_resolution(app)
        if not self.is_offscreen:
            self._encode_queue = queue.Queue(maxsize=8)
            self._encode_thread = threading.Thread(
                target=self._paced_encode_worker,
                args=(self._recorder, self._encode_queue),
                name="kangengine-video-encoder",
                daemon=True,
            )
            self._encode_thread.start()
        app.set_frame_capture_active(True)
        print(f"[INFO]: Started video recording: {path}")
        return path

    def stop(self, app) -> Path | None:
        if self._recorder is None:
            app.set_frame_capture_active(False)
            return self._output_path

        was_active = self._recorder.is_active()
        app.set_frame_capture_active(False)
        self._schedule_final_paced_frames()
        self._finish_paced_encoding()
        self._recorder.stop()
        self._wall_start_time = None
        self._last_paced_frame = None
        self._scheduled_frame_count = 0
        self._capture_resolution = None
        if was_active and self._output_path is not None:
            print(f"[INFO]: Saved video recording: {self._output_path}")
        if self._encode_error is not None:
            error = self._encode_error
            self._encode_error = None
            raise RuntimeError("video encoding failed") from error
        return self._output_path

    def toggle(self, app) -> Path | None:
        if self.is_recording:
            return self.stop(app)
        return self.start(app)

    def on_frame_rendered(self, app):
        toggle_requested = app.consume_video_recording_toggle_requested()
        if toggle_requested and self.is_recording:
            self._capture_frame(app)
            self.stop(app)
            return
        if toggle_requested:
            self.start(app)
        if not self.is_recording:
            return

        self._capture_frame(app)

    def close(self, app):
        self.stop(app)

    def _resolve_fps(self, app, fps: float | None) -> float:
        if fps is not None:
            return float(fps)
        if self.default_fps is not None:
            return self.default_fps
        if self.is_offscreen and app.timing_config is not None:
            return float(app.timing_config.fixed_update_hz)
        return 60.0

    def _capture_frame(self, app):
        repeat_count = self._paced_frame_count() if not self.is_offscreen else 1
        if repeat_count <= 0:
            return
        if not self.is_offscreen and self._encode_queue.full():
            return
        frame = self._read_frame(app)
        if self.is_offscreen:
            self._recorder.write(frame)
            return

        current_frame = frame.copy()
        held_frame = (
            current_frame if self._last_paced_frame is None else self._last_paced_frame
        )
        self._encode_queue.put_nowait((held_frame, current_frame, repeat_count))
        self._last_paced_frame = current_frame
        self._scheduled_frame_count += repeat_count

    def _paced_frame_count(self) -> int:
        if self._wall_start_time is None:
            self._wall_start_time = self._clock()
        elapsed = max(0.0, self._clock() - self._wall_start_time)
        target_count = max(1, math.ceil(elapsed * self._recorder.get_fps() - 1.0e-9))
        return max(0, target_count - self._scheduled_frame_count)

    def _paced_encode_worker(self, recorder, encode_queue):
        while True:
            item = encode_queue.get()
            if item is None:
                return
            held_frame, current_frame, repeat_count = item
            try:
                for _ in range(repeat_count - 1):
                    recorder.write(held_frame)
                recorder.write(current_frame)
            except BaseException as exc:
                if self._encode_error is None:
                    self._encode_error = exc

    def _finish_paced_encoding(self):
        if self._encode_queue is None:
            return
        self._encode_queue.put(None)
        self._encode_thread.join()
        self._encode_queue = None
        self._encode_thread = None

    def _schedule_final_paced_frames(self):
        if self._encode_queue is None or self._last_paced_frame is None:
            return
        repeat_count = self._paced_frame_count()
        if repeat_count <= 0:
            return
        self._encode_queue.put(
            (self._last_paced_frame, self._last_paced_frame, repeat_count)
        )
        self._scheduled_frame_count += repeat_count

    def _default_output_path(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return self.output_dir / f"recording_{stamp}.mp4"

    @staticmethod
    def _mode_value(run_mode) -> str:
        return str(getattr(run_mode, "value", run_mode))

    def _resolve_capture_resolution(self, app) -> tuple[int, int] | None:
        if self.max_resolution is None:
            return None
        get_width = getattr(app, "get_width", None)
        get_height = getattr(app, "get_height", None)
        if get_width is None or get_height is None:
            return None
        source_width = int(get_width())
        source_height = int(get_height())
        max_width, max_height = self.max_resolution
        scale = min(1.0, max_width / source_width, max_height / source_height)
        width = max(2, int(source_width * scale) // 2 * 2)
        height = max(2, int(source_height * scale) // 2 * 2)
        return width, height

    def _read_frame(self, app):
        if self._capture_resolution is None:
            return app.read_rgb_pixels(True)
        width, height = self._capture_resolution
        return app.read_rgb_pixels_resized(width, height, True)

    @staticmethod
    def _validate_resolution(resolution):
        if resolution is None:
            return None
        width, height = resolution
        try:
            width = int(width)
            height = int(height)
        except (TypeError, ValueError):
            raise ValueError("video resolution must contain integer dimensions")
        if width < 2 or height < 2:
            raise ValueError("video resolution dimensions must be at least 2")
        return width, height
