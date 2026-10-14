"""Exercise common capture, video coexistence and failure cleanup without MCP."""

from pathlib import Path
import tempfile

import kangengine as ke
from kangengine.app.capture import AutomationError


class FailingApp(ke.App):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.cleaned = 0

    def setup(self):
        if self.stage == "setup":
            raise RuntimeError("expected setup failure")

    def pre_render(self):
        if self.stage == "render":
            raise RuntimeError("expected render failure")

    def cleanup(self):
        self.cleaned += 1


def main():
    for stage in ("setup", "render"):
        app: ke.App = FailingApp(stage)
        app.initialize(width=96, height=64, hide_ui=True, headless=True)
        waiting = app.capture_frame()
        try:
            app.start()
        except RuntimeError as error:
            assert str(error) == f"expected {stage} failure"
        else:
            raise AssertionError("callback failure was lost")
        assert app.closed and app.cleaned == 1
        app.close()
        assert app.cleaned == 1
        try:
            waiting.result()
        except AutomationError as error:
            assert error.code == "CLOSED"
        else:
            raise AssertionError("capture unexpectedly completed")

    with tempfile.TemporaryDirectory() as directory:
        app = ke.App()
        try:
            app.initialize(width=96, height=64, hide_ui=True, headless=True)
            app.set_vsync(enabled=False)
            app.update_and_draw()
            app.automation.output_dir = Path(directory)
            app.set_video_recording_resolution(width=96, height=64)
            app.start_video_recording(output_path=Path(directory) / "video.mp4", fps=30)
            cancelled = app.capture_frame(output_path="cancelled.png")
            assert cancelled.cancel()
            capture = app.capture_frame(output_path="during-video.png")
            app.draw_frame()
            result = capture.result(timeout=5)
            assert result.frame_id == app.frame_index - 1
            assert Path(result.output_path).is_file()
            assert not (Path(directory) / "cancelled.png").exists()
            app.stop_video_recording()
            capture = app.capture_frame(output_path="after-video.png")
            app.update_and_draw()
            assert capture.result(timeout=5).frame_id == app.frame_index - 1
            assert (Path(directory) / "video.mp4").stat().st_size > 0
        finally:
            app.close()
        app.close()
    print("Common capture lifecycle and video smoke passed")


if __name__ == "__main__":
    main()
