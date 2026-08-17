# Newton Viewer

KangEngine can be used as a viewer for Newton simulations. Newton
owns the simulation while `ViewerKE` handles rendering and interaction.

![Newton mjcf-test](../../images/newton/mjcf_test.png)

## Install

```bash
uv pip install newton
```

## Basic example

```python
import newton

from kangengine.adapters.newton import ViewerKE

builder = newton.ModelBuilder()
body = builder.add_body()
builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)

model = builder.finalize(device="cpu")  # or "cuda:0"
state = model.state()

viewer = ViewerKE()
viewer.set_model(model)
try:
    while viewer.is_running():
        # Step Newton and update state here.
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
finally:
    viewer.close()
```

Run the complete rigid-body example:

```bash
python ./python/examples/adapters/newton/newton_basic_shapes.py
```

<details>
<summary>Complete source: <code>newton_basic_shapes.py</code></summary>

```{literalinclude} ../../../python/examples/adapters/newton/newton_basic_shapes.py
:language: python
:linenos:
```

</details>

The example simulates sphere, capsule, box, cylinder, cone, and mesh shapes in
Newton and displays them through `ViewerKE`. Shift + left drag applies a picking force.

![Newton rigid-body shapes](../../images/newton/basic_shapes.png)
