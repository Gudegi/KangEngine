from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from kangengine.recording import VideoCaptureController, VideoRecorder


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


class _FakeApp:
    def __init__(self, *, render_hz=60.0, fixed_update_hz=30.0):
        self._render_hz = render_hz
        self.timing_config = mock.Mock(fixed_update_hz=fixed_update_hz)
        self.capture_active = False
        self.toggle_requested = False

    def set_frame_capture_active(self, active):
        self.capture_active = bool(active)

    def consume_video_recording_toggle_requested(self):
        requested = self.toggle_requested
        self.toggle_requested = False
        return requested

    def read_rgb_pixels(self, flip_y):
        return np.zeros((4, 8, 3), dtype=np.uint8)

    def get_render_hz(self):
        return self._render_hz


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class VideoRecorderTest(unittest.TestCase):
    def test_requires_at_least_one_output(self):
        with self.assertRaisesRegex(ValueError, "requires output_path"):
            VideoRecorder()

    def test_retains_frames_and_optionally_dumps_ppm(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(
                fps=24,
                retain_frames=True,
                frame_dir=directory,
            )
            frame = np.full((3, 5, 3), 127, dtype=np.uint8)

            recorder.start()
            recorder.write(frame)
            recorder.write(frame + 1)
            recorder.stop()

            self.assertEqual(recorder.get_fps(), 24.0)
            self.assertEqual(recorder.get_num_frames(), 2)
            self.assertEqual(recorder.get_resolution(), (3, 5))
            self.assertEqual(len(recorder.get_frames()), 2)
            self.assertTrue((Path(directory) / "frame_000000.ppm").is_file())
            self.assertTrue((Path(directory) / "frame_000001.ppm").is_file())

    def test_rejects_resolution_changes(self):
        recorder = VideoRecorder(retain_frames=True)
        recorder.start()
        recorder.write(np.zeros((3, 5, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "resolution changed"):
            recorder.write(np.zeros((4, 5, 3), dtype=np.uint8))

    def test_streams_without_retaining_frames(self):
        writer = _FakeWriter()
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "video.mp4"
            recorder = VideoRecorder(output_path, fps=60)
            with mock.patch.object(
                recorder,
                "_open_writer",
                return_value=writer,
            ):
                with recorder:
                    recorder.write(np.zeros((2, 4, 3), dtype=np.uint8))
                saved_path = recorder.save()

            self.assertEqual(saved_path, output_path)
            self.assertEqual(len(writer.frames), 1)
            self.assertTrue(writer.closed)
            with self.assertRaisesRegex(RuntimeError, "not retained"):
                recorder.get_frames()

    def test_inactive_write_is_ignored(self):
        recorder = VideoRecorder(retain_frames=True)
        result = recorder.write(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertIsNone(result)
        self.assertEqual(recorder.get_num_frames(), 0)


class VideoCaptureControllerTest(unittest.TestCase):
    def test_offscreen_uses_fixed_update_rate_without_status_overlay(self):
        writer = _FakeWriter()
        app = _FakeApp(render_hz=0.0, fixed_update_hz=30.0)
        with tempfile.TemporaryDirectory() as directory:
            controller = VideoCaptureController(directory)
            controller.configure(run_mode="offscreen_fast")
            with mock.patch.object(VideoRecorder, "_open_writer", return_value=writer):
                controller.start(app)
                controller.on_frame_rendered(app)
                controller.stop(app)

        self.assertEqual(controller.recorder.get_fps(), 30.0)
        self.assertEqual(controller.recorder.get_num_frames(), 1)
        self.assertFalse(app.capture_active)
        self.assertTrue(writer.closed)

    def test_paced_hotkey_request_toggles_recording(self):
        writer = _FakeWriter()
        app = _FakeApp(render_hz=72.0)
        clock = _FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            controller = VideoCaptureController(directory, clock=clock)
            controller.configure(run_mode="paced")
            app.toggle_requested = True
            with mock.patch.object(VideoRecorder, "_open_writer", return_value=writer):
                controller.on_frame_rendered(app)
                self.assertTrue(controller.is_recording)
                self.assertTrue(app.capture_active)
                self.assertEqual(controller.recorder.get_fps(), 60.0)

                clock.now = 1.0 / 60.0
                app.toggle_requested = True
                controller.on_frame_rendered(app)

        self.assertFalse(controller.is_recording)
        self.assertFalse(app.capture_active)
        self.assertEqual(controller.recorder.get_num_frames(), 1)

    def test_paced_subsamples_144_hz_rendering_to_60_fps(self):
        writer = _FakeWriter()
        app = _FakeApp(render_hz=144.0)
        clock = _FakeClock()
        controller = VideoCaptureController(fps=60.0, clock=clock)
        controller.configure(run_mode="paced")

        with mock.patch.object(VideoRecorder, "_open_writer", return_value=writer):
            controller.start(app, "paced.mp4")
            for frame_index in range(1, 145):
                clock.now = frame_index / 144.0
                controller.on_frame_rendered(app)
            controller.stop(app)

        self.assertEqual(controller.recorder.get_num_frames(), 60)
        self.assertEqual(len(writer.frames), 60)

    def test_paced_duplicates_frames_when_rendering_is_slow(self):
        writer = _FakeWriter()
        app = _FakeApp(render_hz=30.0)
        app.read_rgb_pixels = mock.Mock(
            side_effect=[
                np.full((4, 8, 3), value, dtype=np.uint8) for value in range(30)
            ]
        )
        clock = _FakeClock()
        controller = VideoCaptureController(fps=60.0, clock=clock)
        controller.configure(run_mode="paced")

        with mock.patch.object(VideoRecorder, "_open_writer", return_value=writer):
            controller.start(app, "paced.mp4")
            for frame_index in range(1, 31):
                clock.now = frame_index / 30.0
                controller.on_frame_rendered(app)
            controller.stop(app)

        self.assertEqual(controller.recorder.get_num_frames(), 60)
        self.assertEqual(len(writer.frames), 60)
        np.testing.assert_array_equal(writer.frames[2], writer.frames[1])
        self.assertEqual(int(writer.frames[3][0, 0, 0]), 1)

    def test_headless_fast_rejects_recording(self):
        controller = VideoCaptureController()
        controller.configure(run_mode="headless_fast")
        with self.assertRaisesRegex(RuntimeError, "disables rendering"):
            controller.start(_FakeApp())

    def test_default_resolution_caps_4k_at_fhd_without_upscaling(self):
        app = _FakeApp()
        app.get_width = mock.Mock(return_value=3840)
        app.get_height = mock.Mock(return_value=2160)
        controller = VideoCaptureController()
        controller.configure(run_mode="offscreen_fast")
        controller.start(app, "offscreen.mp4")
        self.assertEqual(controller._capture_resolution, (1920, 1080))
        controller.stop(app)

        app.get_width.return_value = 1280
        app.get_height.return_value = 720
        controller.start(app, "offscreen.mp4")
        self.assertEqual(controller._capture_resolution, (1280, 720))
        controller.stop(app)


if __name__ == "__main__":
    unittest.main()
