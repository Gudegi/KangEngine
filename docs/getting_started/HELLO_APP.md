# Hello App

After `pip install kangengine`, save this as `hello_app.py`. It creates a cube
and ground plane using built-in geometry and materials.

```python
import kangengine as ke


class HelloApp(ke.App):
    def setup(self):
        self.materials = self.create_standard_materials()
        self.scene.add_ground("/ground", scale=10.0)
        box = self.scene.add_mesh(
            "/box",
            ke.geometry.create_cube_data(1.0),
            self.materials.common,
            color=ke.Vec4(0.8, 0.3, 0.02, 1.0),
        )
        box.set_local_translation(ke.Vec3(0.0, 0.5, 0.0))
        self.set_camera_view([3.0, 2.5, 4.0], [0.0, 0.5, 0.0])


app = HelloApp()
app.initialize(1280, 720, False, ke.UpAxis.Y)
app.start()
```

Run in a graphical session:

```bash
python hello_app.py
```

`setup()` creates the scene once; `start()` runs the window and render loop.
Close the window to exit. No external assets or repository checkout are needed.

## Lifecycle callbacks

Override these methods in your `ke.App` subclass using the Python names below.
Only implement the callbacks your application needs.

| Callback | When it runs and what to put there |
| --- | --- |
| `setup()` | Once before the main loop; create scene objects and resources. |
| `pre_update()` | Once per frame, before fixed updates; handle input and per-frame state. |
| `fixed_update(fixed_dt)` | Zero or more times per frame; advance control and physics by the supplied duration in seconds. |
| `pre_render()` | Before scene rendering; synchronize visuals with the latest state. |
| `render()` | Each frame; add ImGui UI and custom per-frame drawing. |
| `post_render()` | After rendering; perform end-of-frame work. |
| `cleanup()` | When `start()` exits, including on exceptions; release resources owned by your application. |

Within each frame, callbacks run in this order:
`pre_update()` → `fixed_update()` (zero or more calls) → `pre_render()` →
`render()` → `post_render()`.

Advance physics in `fixed_update()`. See
[Fixed Timestep and Rendering](../simulation/FIXED_TIMESTEP.md) for timing
configuration and simulation examples.

Next: [First Scene](FIRST_SCENE.md) or [First Simulation](FIRST_SIMULATION.md).
