"""Check native demand gating and worker wakeups with a real MCP-enabled App."""

import threading

import kangengine as ke


class CountingApp(ke.App):
    def __init__(self):
        super().__init__()
        self.started_hooks = 0
        self.rendered_hooks = 0
        self.capture_in_render = False
        self.late_capture = None
        self.updates = 0

    def pre_update(self):
        self.updates += 1

    def _on_frame_started_internal(self):
        self.started_hooks += 1
        super()._on_frame_started_internal()

    def _on_frame_rendered_internal(self):
        self.rendered_hooks += 1
        super()._on_frame_rendered_internal()

    def pre_render(self):
        if self.capture_in_render:
            self.capture_in_render = False
            self.late_capture = self.capture_frame()


def main():
    app: ke.App = CountingApp()
    try:
        app.initialize(width=160, height=90, hide_ui=True, headless=True)
        app.set_vsync(enabled=False)
        for _ in range(100):
            app.update_and_draw()
        assert (app.started_hooks, app.rendered_hooks) == (0, 0)

        app.enable_mcp(port=0)
        app.update_and_draw()
        assert (app.started_hooks, app.rendered_hooks) == (1, 1)
        for _ in range(100):
            app.draw_frame()
        assert (app.started_hooks, app.rendered_hooks) == (1, 1)

        service = app.automation
        pending = []
        worker = threading.Thread(
            target=lambda: pending.append(service.submit("viewport_capture"))
        )
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        updates_before_capture = app.updates
        app.set_simulation_paused(paused=True)
        app.request_simulation_step()
        app.process_events()
        assert not pending[0].done()
        app.draw_frame()
        assert pending[0].result(timeout=5).frame_id == app.frame_index - 1
        assert app.updates == updates_before_capture
        assert app.consume_simulation_step()
        assert (app.started_hooks, app.rendered_hooks) == (2, 2)
        for _ in range(100):
            app.update_and_draw()
        assert (app.started_hooks, app.rendered_hooks) == (2, 2)

        app.capture_in_render = True
        app.draw_frame()
        assert not app.late_capture.done()
        assert (app.started_hooks, app.rendered_hooks) == (2, 2)
        app.update_and_draw()
        assert app.late_capture.result(timeout=5).frame_id == app.frame_index - 1
        assert (app.started_hooks, app.rendered_hooks) == (3, 3)
    finally:
        app.close()
    print("Idle MCP: 0 automation hooks across 100 frames before/after capture")
    print("Worker and mid-frame requests wake the next frame: passed")


if __name__ == "__main__":
    main()
