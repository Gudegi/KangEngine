"""Draw one triangle through the Python SceneHook RHI escape hatch."""

from __future__ import annotations

import numpy as np

import kangengine as ke
from kangengine import render


class SceneHookSmoke(ke.App):
    def __init__(self):
        super().__init__()
        self.records = 0

    def setup(self):
        vertices = np.asarray([[-0.5, -0.4], [0.5, -0.4], [0.0, 0.5]], dtype=np.float32)
        self.vertex_buffer = (
            self.get_renderer()
            .device()
            .create_buffer(
                vertices,
                render.BufferUsage.VERTEX,
                label="python_scene_hook_triangle",
            )
        )

        layout = render.VertexBufferLayout()
        layout.array_stride = 8
        layout.attributes = [
            render.VertexAttributeDesc(render.VertexFormat.FLOAT32_X2, 0, 0)
        ]

        shader = render.ShaderDesc()
        shader.name = "python_scene_hook_triangle"
        shader.stages = [
            render.ShaderStage(
                """#version 410 core
layout(location = 0) in vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
""",
                render.ShaderType.VERTEX,
            ),
            render.ShaderStage(
                """#version 410 core
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(0.15, 0.8, 1.0, 1.0); }
""",
                render.ShaderType.FRAGMENT,
            ),
        ]

        desc = render.SceneHookPipelineDesc()
        desc.label = "python_scene_hook_triangle"
        desc.shader = shader
        desc.vertex_buffers = [layout]
        desc.depth_test = False
        self.pipeline = self.create_scene_hook_pipeline(desc)
        self.hook = self.add_render_hook(
            render.RenderHookPhase.AFTER_TRANSPARENT,
            self.record_draw,
            pipeline=self.pipeline,
        )

        _, pipeline_handle = self._scene_hook_resource_handles[id(self.pipeline)]
        pipeline_prim = self.resources.resource_prim(pipeline_handle)
        if pipeline_prim is None or not pipeline_prim.get_path().startswith(
            "/.Resources/Pipelines/"
        ):
            raise AssertionError("custom pipeline was not mirrored into resources")

    def record_draw(self, context):
        self.records += 1
        draw = context.pass_encoder
        draw.set_viewport(0.0, 0.0, float(context.width), float(context.height))
        draw.set_pipeline(self.pipeline)
        draw.set_vertex_buffer(0, self.vertex_buffer)
        draw.draw(3)

    def pre_render(self):
        if self.records >= 2:
            self.request_close()

    def cleanup(self):
        if hasattr(self, "hook"):
            self.remove_render_hook(self.hook)


def main():
    app = SceneHookSmoke()
    app.initialize(320, 180, True, ke.UpAxis.Y, headless=True)
    app.start()
    if app.records < 2:
        raise AssertionError("Python render hook did not record draws")
    print("PASS: Python SceneHook custom pipeline")


if __name__ == "__main__":
    main()
