"""Check scheduled animation and material preservation across environment toggles."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import kangengine as ke

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rendering"))
from view_scene_renderer import SceneRendererApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = SceneRendererApp()
    try:
        app.initialize(width=960, height=640, up_axis=ke.UpAxis.Y, render_backend=args.backend)
        app.setup()
        app.set_vsync(enabled=False)
        app.update_and_draw()
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            app.update_and_draw()
            time.sleep(0.01)
        assert app.angle > 0, "Scheduled animation never ran"
        app.animate = False
        angle = app.angle
        for _ in range(4):
            time.sleep(0.02)
            app.update_and_draw()
        assert app.angle == angle, "Paused animation kept advancing"

        def capture():
            for _ in range(32):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        before = capture()
        settled = capture()
        noise = np.abs(settled - before).mean()
        for _ in range(2):
            app.environment.set_visible(False)
            hidden = capture()
            assert np.abs(hidden - settled).mean() > 1
            app.environment.set_visible(True)
            restored = capture()
            error = np.abs(restored - settled).mean()
            assert error < max(2, noise * 3), f"Environment toggle changed the scene: {error=}, {noise=}"
    finally:
        app.close()
    print(f"PASS: {args.backend} scheduled animation, pause, and environment toggle restoration")


if __name__ == "__main__":
    main()
