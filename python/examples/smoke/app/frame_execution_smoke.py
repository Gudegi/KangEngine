"""Check frame scheduling ownership and capture with a real hidden GL window."""

from pathlib import Path
import tempfile
from unittest import mock

import kangengine as ke
from kangengine.recording import VideoRecorder


class ProbeApp(ke.App):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.close_after_draw = False

    def setup(self):
        self.calls.append("setup")

    def pre_update(self):
        self.calls.append("pre_update")

    def fixed_update(self, fixed_dt):
        assert abs(fixed_dt - 1.0 / 60.0) < 1e-9
        self.calls.append("fixed_update")

    def pre_render(self):
        self.calls.append("pre_render")

    def render(self):
        self.calls.append("render")

    def post_render(self):
        self.calls.append("post_render")
        if self.close_after_draw:
            self.request_close()


def main():
    app: ke.App = ProbeApp()
    app.initialize(width=64, height=64, hide_ui=True, headless=True)
    app.set_vsync(enabled=False)
    app.set_fixed_update_hz(update_hz=60.0)
    app.set_simulation_paused(paused=True)
    renderer: ke.render.Renderer = app.get_renderer()
    renderer.set_profiler_enabled(enabled=True)
    render_calls = ["pre_render", "render", "post_render"]

    # Real capture reads the rendered framebuffer; only the encoder is mocked.
    writer = mock.Mock()
    with tempfile.TemporaryDirectory() as directory:
        with mock.patch.object(VideoRecorder, "_open_writer", return_value=writer):
            app.start_video_recording(output_path=Path(directory) / "frames.mp4")
            app.request_simulation_step()
            app.process_events()
            assert app.calls == []
            for _ in range(2):
                app.draw_frame()
            assert app.calls == render_calls * 2
            assert writer.append_data.call_count == 2
            frame = renderer.latest_frame_profile()
            assert frame.metadata["frame_execution"] == "draw_frame"
            assert frame.metadata["fixed_update_count"] == "0"
            assert not {"event_poll", "simulation/pre_update", "simulation/fixed_update"} & {
                sample.path for sample in frame.samples
            }
            assert app.get_dropped_wall_time() == 0.0

            # Drawing and event handling leave the pending update untouched.
            app.calls.clear()
            app.update_and_draw()
            assert app.calls == ["pre_update", "fixed_update"] + render_calls
            frame = renderer.latest_frame_profile()
            assert frame.metadata["frame_execution"] == "update_and_draw"
            assert frame.metadata["fixed_update_count"] == "1"
            assert "event_poll" in {sample.path for sample in frame.samples}
            assert not app.consume_simulation_step()
            app.calls.clear()
            app.update_and_draw()
            assert app.calls == ["pre_update"] + render_calls
            assert writer.append_data.call_count == 4
            rgb = writer.append_data.call_args.args[0]
            assert rgb.shape == (app.get_height(), app.get_width(), 3)
            assert rgb.dtype.name == "uint8"
            screenshot = Path(directory) / "frame.png"
            assert app.write_pixels_png(path=str(screenshot), flip_y=True)
            assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            app.stop_video_recording()
            writer.close.assert_called_once()

    # An external scheduler needs neither a fixed-update rate nor callbacks.
    app.set_fixed_update_hz(update_hz=0.0)
    app.calls.clear()
    app.request_simulation_step()
    app.process_events()
    app.draw_frame()
    assert app.calls == render_calls
    assert app.consume_simulation_step()
    assert not app.consume_simulation_step()
    app.request_simulation_step()
    app.set_simulation_paused(paused=False)
    assert not app.consume_simulation_step()

    app.set_fixed_update_hz(update_hz=60.0)
    app.set_max_frame_delta(seconds=0.000001)
    app.calls.clear()
    app.draw_frame()
    app.process_events()
    app.draw_frame()
    assert app.calls == render_calls * 2
    assert app.get_dropped_wall_time() == 0.0
    app.set_fixed_update_hz(update_hz=0.0)

    # start() owns setup and uses the combined path, without duplicate hooks.
    app.calls.clear()
    app.close_after_draw = True
    app.start()
    assert app.calls == ["setup", "pre_update"] + render_calls
    print("PASS: frame scheduling, callbacks, video/PNG capture, profiling, external stepping and start")


if __name__ == "__main__":
    main()
