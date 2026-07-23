# Hello App

`ke.App` owns the window, render loop, camera, scene facade, and renderer. A
Python application subclasses it and implements lifecycle callbacks.

```python
import kangengine as ke


class HelloApp(ke.App):
    def setup(self):
        self.set_camera_view([3.0, -4.0, 2.0], [0.0, 0.0, 0.5])

    def preRender(self):
        pass

    def render(self):
        pass


app = HelloApp()
app.initialize(1280, 720, False, ke.UpAxis.Y)
app.start()
```

Save this as `hello_app.py` and run:

```bash
python hello_app.py
```

Lifecycle callback names retain their C++ virtual spelling, while normal
Python methods use snake_case.

- `setup()`: create resources and scene objects once.
- `preRender()`: update simulation or animation before drawing.
- `render()`: build ImGui panels or other per-frame UI.
- `cleanup()`: release explicitly owned simulation resources.

The complete scene example used by the next page is
`python/examples/render_prim_scene.py`.

Next: [First Scene](FIRST_SCENE.md).
