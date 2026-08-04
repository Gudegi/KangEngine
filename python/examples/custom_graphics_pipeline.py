"""Interactive custom graphics-pipeline example.

The triangle is submitted by a Python render hook rather than as a scene mesh.
Open /.Resources/ShaderSources and /.Resources/Pipelines in the Scene panel to
inspect the authored shader and pipeline definitions.
"""

from __future__ import annotations

import numpy as np

import kangengine as ke


class CustomGraphicsPipelineApp(ke.App):
    def setup(self):
        # Clip-space position (xy) and color (rgb).
        vertices = np.asarray(
            [
                [-0.65, -0.55, 1.0, 0.15, 0.10],
                [0.65, -0.55, 0.10, 0.85, 1.0],
                [0.00, 0.65, 0.75, 0.20, 1.0],
            ],
            dtype=np.float32,
        )
        self.vertex_buffer = (
            self.get_renderer()
            .device()
            .create_buffer(
                vertices,
                ke.render.BufferUsage.VERTEX,
                label="custom_gradient_triangle_vertices",
            )
        )

        vertex_layout = ke.render.VertexBufferLayout()
        # One vertex is five consecutive float32 values:
        # [position.x, position.y, color.r, color.g, color.b].
        # Advance 20 bytes (5 floats * 4 bytes) to reach the next vertex.
        vertex_layout.array_stride = 5 * np.dtype(np.float32).itemsize
        vertex_layout.attributes = [
            # FLOAT32X2 at byte offset 0 -> vertex shader location 0
            # (`layout(location = 0) in vec2 aPosition`).
            ke.render.VertexAttributeDesc(ke.render.VertexFormat.FLOAT32_X2, 0, 0),
            # FLOAT32X3 at byte offset 8(4*2) -> vertex shader location 1
            # (`layout(location = 1) in vec3 aColor`).
            ke.render.VertexAttributeDesc(ke.render.VertexFormat.FLOAT32_X3, 8, 1),
        ]

        shader = ke.render.ShaderDesc()
        shader.name = "custom_gradient_triangle"
        shader.stages = [
            ke.render.ShaderStage(
                """#version 410 core
                    layout(location = 0) in vec2 aPosition;
                    layout(location = 1) in vec3 aColor;

                    out vec3 vertexColor;

                    void main() {
                        gl_Position = vec4(aPosition, 0.0, 1.0);
                        vertexColor = aColor;
                    }
                """,
                ke.render.ShaderType.VERTEX,
            ),
            ke.render.ShaderStage(
                """#version 410 core
                    layout(location = 0) out vec4 outColor;

                    in vec3 vertexColor;

                    void main() {
                        outColor = vec4(vertexColor, 1.0);
                    }
                """,
                ke.render.ShaderType.FRAGMENT,
            ),
        ]

        pipeline_desc = ke.render.SceneHookPipelineDesc()
        pipeline_desc.label = "CustomGradientTriangle"
        pipeline_desc.shader = shader
        pipeline_desc.vertex_buffers = [vertex_layout]
        pipeline_desc.depth_test = False
        pipeline_desc.depth_write = False

        # The App helper creates the GPU pipeline and mirrors its authored
        # definition into /.Resources for editor inspection.
        self.pipeline = self.create_scene_hook_pipeline(pipeline_desc)
        # `pipeline=` tracks hook -> pipeline usage in the Resource panel. The
        # callback must still record set_pipeline() as an actual draw command.
        self.render_hook = self.add_render_hook(
            ke.render.RenderHookPhase.AFTER_TRANSPARENT,
            self.record_custom_draw,
            pipeline=self.pipeline,
        )

        print("Custom gradient triangle pipeline is active.")
        print("Inspect: /.Resources/ShaderSources/custom_gradient_triangle_*")
        print("Inspect: /.Resources/Pipelines/CustomGradientTriangle_*")
        print("Press Escape to close the example.")

    def record_custom_draw(self, context):
        draw = context.pass_encoder
        draw.set_viewport(
            0.0,
            0.0,
            float(context.width),
            float(context.height),
        )
        draw.set_pipeline(self.pipeline)  # Required GPU pipeline selection.
        draw.set_vertex_buffer(0, self.vertex_buffer)
        draw.draw(3)

    def cleanup(self):
        if hasattr(self, "render_hook"):
            self.remove_render_hook(self.render_hook)


def main():
    app = CustomGraphicsPipelineApp()
    app.initialize(width=1920, height=1080, hide_ui=False, up_axis=ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
