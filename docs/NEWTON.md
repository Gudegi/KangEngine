# Newton Viewer

KangEngine can be used as a viewer for Newton simulations. Newton
owns the simulation while `NewtonViewer` handles rendering and interaction.

![Newton mjcf-test](../../images/newton/mjcf_test.png)

## Install

```bash
uv pip install "kangengine[newton]"
```

From a source checkout, use `uv pip install -e "./python[newton]"` instead.

## Basic example

```python
import newton

from kangengine.adapters.newton import NewtonViewer

builder = newton.ModelBuilder()
body = builder.add_body()
builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)

model = builder.finalize(device="cpu")  # or "cuda:0"
state = model.state()

viewer: NewtonViewer = NewtonViewer()
viewer.set_model(model=model)
try:
    while viewer.is_running():
        # Step Newton and update state here.
        viewer.begin_frame(0.0)
        viewer.log_state(state=state)
        viewer.end_frame()
finally:
    viewer.close()
```

## Use an existing App

Use `NewtonVisualizer` with an initialized App. Your application controls
simulation, drawing, and App cleanup.

```python
from kangengine.adapters.newton import NewtonVisualizer

visualizer: NewtonVisualizer = NewtonVisualizer(app)
visualizer.set_model(model=model)
try:
    # Repeat with the current state after each simulation step.
    visualizer.log_state(state=state)
    app.process_events()
    app.draw_frame()
finally:
    visualizer.release()  # The App remains open.
```

Keep the model and state alive until drawing finishes. On CUDA, log the state on
the solver's Warp stream. Use one visualizer per App.

Both classes support the viewport eye menu for colliders, centers of mass,
body/joint axes, and contacts.
After `log_state()`, call `log_contacts(contacts, state)` or
`log_mujoco_contacts(solver, state)` to display contacts. Forces and estimated
contact states require solver-provided forces; otherwise only normals are shown.

## Examples

Run from the repository root:

```bash
python python/examples/adapters/newton/newton_basic_shapes.py --device cpu
python python/examples/adapters/newton/newton_mjcf_articulation.py --device cpu
python python/examples/adapters/newton/newton_franka_ik.py --device cpu
```

Use `--device cpu` or `--device cuda` to select the simulation device.

<details>
<summary>Complete source: <code>newton_basic_shapes.py</code></summary>

```{literalinclude} ../../../python/examples/adapters/newton/newton_basic_shapes.py
:language: python
:linenos:
```

</details>

The basic-shapes example shows spheres, capsules, boxes, cylinders, cones, and
meshes. Shift + left drag applies a force.

![Newton rigid-body shapes](../../images/newton/basic_shapes.png)
