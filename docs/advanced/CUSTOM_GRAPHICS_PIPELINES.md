# Custom Graphics Pipelines

Use a custom graphics pipeline when you need to draw geometry that does not fit
the normal scene-mesh path, such as an experimental particle or procedural
draw. Most applications do not need this API.

A render-hook draw does not receive automatic scene transforms, batching,
visibility, or frustum culling. The caller owns its buffers, pipeline, draw
commands, and resource lifetime.

Examples on this page assume `import kangengine as ke` and run inside an `App`
subclass after graphics initialization.

## Minimal triangle

Create a GPU vertex buffer containing three clip-space positions:

```python
import numpy as np
import kangengine as ke

vertices = np.array(
    [[-0.5, -0.4], [0.5, -0.4], [0.0, 0.5]], dtype=np.float32
)
self.custom_vertex_buffer = self.get_renderer().device().create_buffer(
    vertices,
    ke.render.BufferUsage.VERTEX,
    label="custom_triangle",
)
```

Describe how one vertex is stored. Each vertex contains two `float32` values,
so its stride is 8 bytes. The attribute starts at byte offset 0 and is passed
to shader location 0:

```python
layout = ke.render.VertexBufferLayout()
layout.array_stride = 2 * np.dtype(np.float32).itemsize
layout.attributes = [
    ke.render.VertexAttributeDesc(
        ke.render.VertexFormat.FLOAT32_X2,
        0,  # byte offset inside one vertex
        0,  # shader layout location
    )
]
```

Provide the vertex and fragment stages:

```python
shader = ke.render.ShaderDesc()
shader.name = "custom_triangle"
shader.stages = [
    ke.render.ShaderStage(
        """#version 410 core
layout(location = 0) in vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
""",
        ke.render.ShaderType.VERTEX,
    ),
    ke.render.ShaderStage(
        """#version 410 core
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(0.15, 0.8, 1.0, 1.0); }
""",
        ke.render.ShaderType.FRAGMENT,
    ),
]
```

Create a pipeline compatible with the scene render targets:

```python
desc = ke.render.SceneHookPipelineDesc()
desc.label = "custom_triangle"
desc.shader = shader
desc.vertex_buffers = [layout]
desc.depth_test = False
self.custom_pipeline = self.create_scene_hook_pipeline(desc)
```

Record the draw after the normal transparent pass:

```python
def record_triangle(context):
    draw = context.pass_encoder
    draw.set_viewport(0, 0, context.width, context.height)
    draw.set_pipeline(self.custom_pipeline)
    draw.set_vertex_buffer(0, self.custom_vertex_buffer)
    draw.draw(3)

self.custom_hook = self.add_render_hook(
    ke.render.RenderHookPhase.AFTER_TRANSPARENT,
    record_triangle,
    pipeline=self.custom_pipeline,
)
```

The two pipeline references have different roles:

- `draw.set_pipeline(self.custom_pipeline)` records the GPU pipeline selection
  for this draw and is required before `draw()` or `draw_indexed()`.
- `pipeline=self.custom_pipeline` on `add_render_hook()` tracks the
  hook-to-pipeline relationship in the Resource panel. It increments usage and
  records a `render-hook://...` usage path.

Omitting the tracking argument does not stop rendering, but the Resource panel
cannot infer which pipeline an arbitrary Python callback uses and reports zero
hook usage.

## Lifetime and cleanup

Keep the buffer, pipeline, and callback alive while the hook is registered.
The callback `context` and its `pass_encoder` are valid only during that
callback and must not be stored for later use.

Remove the hook during application cleanup:

```python
self.remove_render_hook(self.custom_hook)
```

Removing it also removes the tracked pipeline usage.

## Resource panel

`App.create_scene_hook_pipeline()` mirrors the authored definitions into:

```text
/.Resources/ShaderSources
/.Resources/Pipelines
```

The compiled pipeline remains caller-owned. The Resource entries contain the
shader source, stage, entry point, pipeline state summary, and usage paths for
inspection.

## Run the interactive example

```bash
python python/examples/custom_graphics_pipeline.py
```

The example displays a gradient triangle and keeps the editor open so the
ShaderSources and Pipelines folders can be inspected.
