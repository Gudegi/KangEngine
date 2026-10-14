from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import imageio.v3 as iio
import numpy as np

from kangengine.app.automation import AutomationService, MAX_PENDING
from kangengine.app.capture import AutomationError


class _App:
    def __init__(self):
        self.frame_index = 7
        self.run_config = None
        self.reads = 0
        self.automation_requested = True
        self.pixels = np.zeros((3, 5, 3), dtype=np.uint8)
        self.pixels[0, :, 0] = 255
        self.pixels[-1, :, 2] = 255

    def _request_automation_frame(self):
        self.automation_requested = True

    def get_width(self):
        return 5

    def get_height(self):
        return 3

    def read_rgb_pixels(self, *, flip_y):
        assert flip_y
        self.reads += 1
        return self.pixels.copy()


class AutomationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.app = _App()
        self.service = AutomationService(self.app, output_dir=self.directory.name)
        self.addCleanup(self.service.close)
        patcher = mock.patch(
            "kangengine.app.mcp.inspection.engine_status",
            return_value={"render_enabled": True},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def frame(self):
        requested = self.app.automation_requested
        self.app.automation_requested = False
        if requested:
            self.service.on_frame_started()
            self.service.on_frame_rendered()
        self.app.frame_index += 1

    def assert_error(self, future, code):
        with self.assertRaises(AutomationError) as raised:
            future.result(timeout=2)
        self.assertEqual(raised.exception.code, code)

    def test_capture_observes_one_normal_frame_and_shares_readback(self):
        first = self.service.submit("viewport_capture", output_path="first.png")
        second = self.service.submit("viewport_capture")
        self.assertEqual(self.app.reads, 0)
        self.frame()
        result = first.result(timeout=2)
        self.assertEqual((result.width, result.height, result.frame_id), (5, 3, 7))
        self.assertEqual(result.app_session_id, self.service.app_session_id)
        self.assertEqual(self.app.reads, 1)
        self.assertEqual(result.png, second.result(timeout=2).png)
        np.testing.assert_array_equal(
            iio.imread(result.png, extension=".png"), self.app.pixels
        )
        self.assertEqual(Path(result.output_path).read_bytes(), result.png)
        self.frame()
        self.assertEqual(self.app.reads, 1)

    def test_cancellation_and_deadline_remove_queued_work(self):
        cancelled = self.service.submit("viewport_capture", output_path="cancelled.png")
        self.assertTrue(cancelled.cancel())
        with mock.patch("kangengine.app.automation.REQUEST_TIMEOUT", 0.02):
            expired = self.service.submit("viewport_capture", output_path="expired.png")
        self.assert_error(expired, "TIMEOUT")
        self.frame()
        self.assertEqual(self.app.reads, 0)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        self.assertEqual(len(self.service._queue), 0)

    def test_queue_is_bounded_and_close_rejects_waiting_requests(self):
        pending = [self.service.submit("viewport_capture") for _ in range(MAX_PENDING)]
        self.assert_error(self.service.submit("viewport_capture"), "BUSY")
        self.service.close()
        self.service.close()
        for future in pending:
            self.assert_error(future, "CLOSED")
        self.assert_error(self.service.submit("viewport_capture"), "CLOSED")

    def test_output_paths_cannot_escape_or_overwrite(self):
        existing = Path(self.directory.name) / "existing.png"
        existing.write_bytes(b"keep")
        for path in (
            "../outside.png",
            "/tmp/outside.png",
            "capture.jpg",
            "existing.png",
        ):
            with self.subTest(path=path):
                future = self.service.submit("viewport_capture", output_path=path)
                self.frame()
                self.assert_error(future, "INVALID_ARGUMENT")
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_headless_fast_and_failed_readback_are_explicit(self):
        self.app.run_config = SimpleNamespace(render_enabled=False)
        future = self.service.submit("viewport_capture")
        self.frame()
        self.assert_error(future, "UNSUPPORTED")
        self.app.run_config = None
        future = self.service.submit("viewport_capture")
        with mock.patch.object(
            self.app, "read_rgb_pixels", side_effect=RuntimeError("no frame")
        ):
            self.frame()
        self.assert_error(future, "CAPTURE_FAILED")

    def test_oversize_png_is_neither_returned_nor_written(self):
        future = self.service.submit("viewport_capture", output_path="too-large.png")
        with mock.patch("kangengine.app.automation.MAX_PNG_BYTES", 1):
            self.frame()
            self.assert_error(future, "OUTPUT_TOO_LARGE")
        self.assertFalse((Path(self.directory.name) / "too-large.png").exists())

    def test_status_is_cached_and_engine_access_stays_on_owner(self):
        result = []

        def worker():
            result.append(self.service.submit("engine_status").result())
            try:
                self.service.on_frame_started()
            except RuntimeError as error:
                result.append(str(error))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["state"], "initializing")
        self.assertIn("creating thread", result[1])
        self.frame()
        status_request = self.service.submit("engine_status")
        self.assertFalse(status_request.done())
        self.frame()
        status = status_request.result()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["last_frame_id"], 7)

    def test_cancelled_captures_cannot_accumulate_pixel_buffers(self):
        entered = threading.Event()
        release = threading.Event()
        encode = self.service._encode

        def slow_encode(*args):
            entered.set()
            release.wait(timeout=3)
            encode(*args)

        try:
            with mock.patch.object(self.service, "_encode", side_effect=slow_encode):
                first = self.service.submit("viewport_capture")
                self.frame()
                self.assertTrue(entered.wait(timeout=2))
                first.cancel()
                second = self.service.submit("viewport_capture")
                self.frame()
                second.cancel()
                for _ in range(20):
                    rejected = self.service.submit("viewport_capture")
                    self.frame()
                    self.assert_error(rejected, "BUSY")
                self.assertEqual(self.app.reads, 2)
        finally:
            release.set()

    def test_close_tolerates_future_cancel_during_failure_delivery(self):
        future = self.service.submit("viewport_capture")
        done = future.done

        def concurrent_cancel():
            was_done = done()
            future.cancel()
            return was_done

        with mock.patch.object(future, "done", side_effect=concurrent_cancel):
            self.service.close()
        self.assertTrue(future.cancelled())
        self.assertEqual(self.service.state, "closed")

    def test_idle_frames_skip_hooks_and_status_refreshes_on_demand(self):
        with mock.patch(
            "kangengine.app.mcp.inspection.engine_status", return_value={}
        ) as inspect:
            self.frame()
            self.assertEqual(inspect.call_count, 1)
            with mock.patch.object(self.service, "on_frame_started") as started:
                for _ in range(50):
                    self.frame()
                started.assert_not_called()
            status = self.service.submit("engine_status")
            self.frame()
            self.assertEqual(inspect.call_count, 2)
            self.assertEqual(status.result()["last_frame_id"], self.app.frame_index - 2)
            capture = self.service.submit("viewport_capture")
            self.frame()
            capture.result(timeout=2)
            self.assertEqual(inspect.call_count, 2)
            self.assertFalse(self.app.automation_requested)

    def test_queue_rearms_after_frame_budget_and_late_submission(self):
        self.frame()
        pending = [self.service.submit("engine_status") for _ in range(6)]
        self.frame()
        self.assertEqual(sum(future.done() for future in pending), 4)
        self.assertTrue(self.app.automation_requested)
        self.frame()
        self.assertTrue(all(future.done() for future in pending))
        self.assertFalse(self.app.automation_requested)
        # A request arriving after the frame-start drain belongs to the next frame.
        self.service.on_frame_started()
        late = self.service.submit("viewport_capture")
        self.service.on_frame_rendered()
        self.assertFalse(late.done())
        self.frame()
        late.result(timeout=2)

    def test_file_commit_does_not_block_frames_or_fail_during_close(self):
        entered = threading.Event()
        release = threading.Event()
        resumed = threading.Event()
        original_open = Path.open

        def slow_open(path, *args, **kwargs):
            if path.name == "slow.png":
                entered.set()
                release.wait(timeout=3)
                resumed.set()
            return original_open(path, *args, **kwargs)

        try:
            with mock.patch.object(Path, "open", slow_open):
                capture = self.service.submit(
                    "viewport_capture", output_path="slow.png"
                )
                self.frame()
                self.assertTrue(entered.wait(timeout=2))
                self.assertTrue(capture.running())
                self.assertFalse(capture.cancel())
                status = self.service.submit("engine_status")
                self.frame()
                self.assertTrue(status.done())
                self.assertFalse(resumed.is_set(), "render waited on file I/O")
                shutdown = self.service._encoder.shutdown

                def release_commit(**kwargs):
                    self.assertEqual(self.service.state, "closing")
                    self.assertFalse(capture.done())
                    release.set()
                    shutdown(**kwargs)

                with mock.patch.object(
                    self.service._encoder, "shutdown", side_effect=release_commit
                ):
                    self.service.close()
                self.assertTrue(Path(capture.result().output_path).is_file())
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
